//! Implementation of Layer Normalisation layer.

use ark_ff::PrimeField;
use dp_crypto::{
    arkyper::{
        CommitmentScheme,
        transcript::{AppendToTranscript, Transcript},
    },
    structs::IOPProof,
};

use crate::{
    NextPowerOfTwo, ProverContext,
    iop::context::ContextAux,
    layers::{
        LayerCtx,
        provable::{Evaluate, OpInfo, PadOp, ProvableOp, ProveInfo, Splittable, VerifiableCtx},
        requant::{FIXED_POINT_SCALE, Requant},
        transformer::normalisation::layernorm::verify::LayerNormLookupVerifier,
    },
    lookup::{
        context::LookupWitnessGen,
        operation::{LookupOp, decomposer::ChunkingInfo, variant::LookupVariant},
        table::SHIFT_CHECK_TABLE_BIT_SIZE,
    },
    padding::PaddingMode,
    poly_commit::verifier::VerifierCommitment,
    quantization,
    tensor::{CommitmentId, TensorTypeParam},
};
use std::collections::HashMap;

use super::*;

pub mod evaluate;
pub mod lookup;
pub mod prove;
pub mod quantise;
pub mod verify;

/// The short name used to identify the LayerNorm layer.
pub(crate) const LAYERNORM_LAYER: &str = "LNRM";

/// Struct storing all information needed to perform LayerNorm.
///
/// The `gamma` and `beta` fields are normally learned parameters that are
/// applied elementwise. The `eps` field is used for normalisation when
/// calculating the inverse square root.
///
/// # References
///
/// - PyTorch's [LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
/// - [Layer Normalization](https://arxiv.org/abs/1607.06450) paper
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "N: Serialize", deserialize = "N: DeserializeOwned"))]
pub struct LayerNorm<N: TensorTypeParam> {
    /// Each element of the normalisation dimension is multiplied elementwise by this
    pub gamma: TensorHandle<N>,
    /// Added elementwise to each element in the normalisation dimension
    pub beta: TensorHandle<N>,
    /// Normalisation factor
    pub eps: f32,
    /// The unpadded size of the normalisation dimension
    pub normalisation_dim_size: usize,
    /// The [`ScalingFactor`] used during quantized evaluation for the mean.
    pub mean_scaling_factor: Option<ScalingFactor>,
    /// The [`ScalingFactor`] for the normalisation value in standard deviation (i.e. before applying gamma and beta)
    pub normalisation_scaling_factor: Option<ScalingFactor>,
    /// Caches the current max shift seen during quantised evaluation. Needed to ensure result of running full sequence is consistent with running
    /// token by token
    cache: Arc<Mutex<NormalisationCache>>,
}

impl<N: TensorTypeParam> LayerNorm<N> {
    pub fn new(gamma: TensorHandle<N>, beta: TensorHandle<N>, eps: f32) -> Result<Self> {
        let gamma_shape = gamma.shape();
        let beta_shape = beta.shape();
        ensure!(
            gamma_shape == beta_shape,
            "Gamma and beta shape must match. gamma {:?} beta {:?}",
            gamma_shape,
            beta_shape,
        );
        ensure!(
            gamma_shape.rank() == 1,
            "Gamma and beta must be 1D. gamma {:?} beta {:?}",
            gamma_shape,
            beta_shape,
        );

        let normalisation_dim_size = gamma_shape.dim(-1);

        Ok(Self {
            gamma: gamma.wrapped_tensor_variant()?,
            beta: beta.wrapped_tensor_variant()?,
            eps,
            normalisation_dim_size,
            mean_scaling_factor: None,
            normalisation_scaling_factor: None,
            cache: Arc::new(Mutex::new(NormalisationCache::new())),
        })
    }

    /// Returns the size of the dimension normalisation occurs over.
    pub fn normalisation_dim_size(&self) -> usize {
        self.normalisation_dim_size
    }

    /// Getter for the quantisation scaling factors
    pub fn get_quantisation_scaling_factors(&self) -> Option<(&ScalingFactor, &ScalingFactor)> {
        match (
            &self.mean_scaling_factor,
            &self.normalisation_scaling_factor,
        ) {
            (Some(mean), Some(normalisation)) => Some((mean, normalisation)),
            _ => None,
        }
    }

    pub fn reset_cache(&self) {
        self.cache.lock().unwrap().reset();
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerNormCtx {
    /// The size of the dimension we normalise over (unpadded)
    normalisation_dim_size: usize,
    /// The [`CommitmentId`] for the `gamma` parameter
    gamma_key: CommitmentId,
    /// The [`CommitmentId`] for the `beta` parameter
    beta_key: CommitmentId,
    /// The [`ScalingFactor`] used for the mean calculation
    mean_scaling: ScalingFactor,
    /// The [`ScalingFactor`] used for the normalising value in standard deviation (i.e. before applying gamma and beta)
    normalisation_scaling: ScalingFactor,
}

impl LayerNormCtx {
    pub fn lookup_tables(&self) -> Vec<Table> {
        vec![
            Table::new_shift_check(),
            Table::new_normalisation(self.normalisation_scaling.bit_size() + 1),
            Table::new_signed_zero_check(),
        ]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
/// Proof for correct execution of a quantised [`LayerNorm`] operation.
pub struct LayerNormProof<F, PCS>
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    /// The LogUp proofs for LayerNorm, they are ordered `inv_sqrt_lookup`, `range_lookup`.
    pub(crate) logup_proof: LogUpBatchProof<F>,
    /// The right shift used in the lookup operation
    pub(crate) right_shift: usize,
    /// Witness commitments for this layer
    pub(crate) commitments: Vec<VerifierCommitment<PCS>>,
    /// The IO proof that links all claims to `last_claim` and the input
    pub(crate) io_proof: IOPProof<F>,
    /// The claimed evaluations of the commitments
    #[serde(with = "dp_crypto::serialization")]
    pub(crate) io_evaluations: Vec<F>,
    /// The evalautions of the mean commitments
    #[serde(with = "dp_crypto::serialization")]
    pub(crate) mean_evals: Vec<F>,
    /// The gamma evaluation
    #[serde(with = "dp_crypto::serialization")]
    pub(crate) gamma_eval: F,
    /// The beta evaluation
    #[serde(with = "dp_crypto::serialization")]
    pub(crate) beta_eval: F,
}

impl<F: PrimeField, PCS: CommitmentScheme> LayerNormProof<F, PCS> {
    pub(crate) fn write_commitment<T: Transcript>(&self, transcript: &mut T) -> Result<()> {
        self.commitments
            .iter()
            .for_each(|comm| comm.append_to_transcript(transcript));
        Ok(())
    }
}

impl OpInfo for LayerNormCtx {
    // https://docs.rs/burn/0.17.0/burn/nn/struct.LayerNorm.html#method.forward
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        _padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        Ok(input_shapes.to_vec())
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        Ok(num_inputs)
    }

    fn describe(&self) -> String {
        format!(
            "LayerNormCtx(dimension size: {})",
            self.normalisation_dim_size,
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<N: TensorTypeParam> OpInfo for LayerNorm<N> {
    // https://docs.rs/burn/0.17.0/burn/nn/struct.LayerNorm.html#method.forward
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        _padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        Ok(input_shapes.to_vec())
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        Ok(num_inputs)
    }

    fn describe(&self) -> String {
        format!(
            "LayerNorm(dimension size: {})",
            self.normalisation_dim_size()
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Stores the unpadded mean and standard deviation tensors for a LayerNorm layer, to be used in proving.
pub struct LayerNormProvingData {
    /// The mean used in the LayerNorm operation.
    pub mean: WrappedTensor<Element>,
    /// The standard deviation used in the LayerNorm operation.
    pub std_dev: WrappedTensor<Element>,
    /// Internal [`LayerNormLookupVerifier`]
    pub(crate) lookup_verifier: LayerNormLookupVerifier,
}

impl LayerNormProvingData {
    pub fn new(
        mean: WrappedTensor<Element>,
        std_dev: WrappedTensor<Element>,
        right_shift: usize,
        dim_size: usize,
        intermediate_scaling_factor: &ScalingFactor,
        intermediate_bit_size: usize,
    ) -> Self {
        let normalised_sumsq = (dim_size as f32
            / (intermediate_scaling_factor.scale() * intermediate_scaling_factor.scale()))
        .round_ties_even() as Element;
        let magnitude_error_bound = (dim_size as f32
            * (2.0f32 * intermediate_scaling_factor.scale().recip() + 1.0f32))
            .round_ties_even() as Element;

        let sum_error_bound = dim_size as Element;

        Self {
            mean,
            std_dev,
            lookup_verifier: LayerNormLookupVerifier::new_from_parts(
                right_shift as Element,
                normalised_sumsq,
                magnitude_error_bound,
                sum_error_bound,
                intermediate_bit_size,
            ),
        }
    }
}

impl LookupOp for LayerNormProvingData {
    fn intermediate_bit_size(&self) -> usize {
        self.lookup_verifier.intermediate_bit_size()
    }

    fn right_shift(&self) -> usize {
        self.lookup_verifier.right_shift()
    }

    fn fixed_point_multiplier(&self) -> Element {
        self.lookup_verifier.fixed_point_multiplier()
    }

    fn variant(&self) -> LookupVariant {
        self.lookup_verifier.variant()
    }

    fn chunking_info(&self, table: &Table) -> Result<ChunkingInfo> {
        self.lookup_verifier.chunking_info(table)
    }

    fn apply(
        &self,
        input: WrappedTensor<Element>,
        table: &Table,
    ) -> Result<WrappedTensor<Element>> {
        let rescaled_input = input.mul(self.std_dev.clone())?;
        let zero_mean_input = rescaled_input.sub(self.mean.clone())?;

        let to_lookup = zero_mean_input
            .add_scalar(self.rounding_constant())
            .bitwise_right_shift_scalar(self.right_shift() as Element);
        table.lookup_tensor(to_lookup)
    }

    fn is_signed(&self) -> bool {
        self.lookup_verifier.is_signed()
    }

    fn padding_value(&self) -> Element {
        self.lookup_verifier.padding_value()
    }
}

impl ProveInfo for LayerNorm<Element> {
    fn step_info<F: PrimeField>(&self, mut aux: ContextAux) -> Result<(LayerCtx<F>, ContextAux)> {
        // Check that the quantisation scaling factors are present
        let (mean_scaling, normalisation_scaling) = self
            .get_quantisation_scaling_factors()
            .ok_or_else(|| anyhow::anyhow!("Quantisation scaling factors not set for LayerNorm"))?;

        // Add the Gamma and Beta commitments
        let gamma_evals = self
            .gamma
            .pad_next_power_of_two()
            .wrapped_tensor()?
            .get_data();
        let beta_evals = self
            .beta
            .pad_next_power_of_two()
            .wrapped_tensor()?
            .get_data();

        aux.model_polys = {
            let mut model_polys = HashMap::new();
            model_polys.insert(CommitmentId::from(self.gamma.storage_key()), gamma_evals);
            model_polys.insert(CommitmentId::from(self.beta.storage_key()), beta_evals);
            Some(model_polys)
        };

        aux.max_poly_len = aux
            .last_output_shape
            .iter()
            .fold(aux.max_poly_len, |acc, shapes| {
                acc.max(shapes.next_power_of_two().product())
            });

        // The output shape is the same as the input shape so we don't need to update it
        // return the LayerCtx and the updated ContextAux
        Ok((
            LayerCtx::LayerNorm(LayerNormCtx {
                normalisation_dim_size: self.normalisation_dim_size,
                gamma_key: CommitmentId::from(self.gamma.storage_key()),
                beta_key: CommitmentId::from(self.beta.storage_key()),
                mean_scaling: *mean_scaling,
                normalisation_scaling: *normalisation_scaling,
            }),
            aux,
        ))
    }
}

impl<F, PCS> ProvableOp<F, PCS> for LayerNorm<Element>
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
{
    type Ctx = LayerNormCtx;

    fn prove<'a, 'b, 'c, 'd, T: Transcript>(
        &'a self,
        node_id: NodeId,
        _ctx: &'b Self::Ctx,
        last_claims: Vec<&Claim<F>>,
        step_data: &Step<Element>,
        prover: &mut Prover<'c, 'd, F, T, PCS>,
    ) -> Result<Vec<Claim<F>>> {
        self.prove_internal(node_id, last_claims[0], step_data, prover)
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        _ctx: &ProverContext<F, PCS>,
        step_data: &Step<Element>,
    ) -> Result<LookupWitnessGen<'_, F>> {
        self.lookup_witness(id, step_data)
    }
}

impl Splittable for LayerNormCtx {}

impl<F, PCS> VerifiableCtx<F, PCS> for LayerNormCtx
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
{
    type Proof = LayerNormProof<F, PCS>;

    fn verify<T: Transcript>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<F>],
        verifier: &mut Verifier<'_, F, T, PCS>,
        shape_step: &ShapeStep,
        node_id: NodeId,
    ) -> Result<Vec<Claim<F>>> {
        self.verify_internal(proof, node_id, verifier, last_claims[0], shape_step)
    }

    fn write_proof_to_transcript<T: Transcript>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        proof.write_commitment(transcript)
    }
}

impl Evaluate<f32> for LayerNorm<f32> {
    fn evaluate(&self, inputs: &[&WrappedTensor<f32>]) -> Result<LayerOut<f32>> {
        self.evaluate_float_internal(inputs)
    }
}

impl Evaluate<Element> for LayerNorm<Element> {
    fn evaluate(&self, inputs: &[&WrappedTensor<Element>]) -> Result<LayerOut<Element>> {
        self.evaluate_quantised_internal(inputs)
    }
}

impl QuantizeOp for LayerNorm<f32> {
    type QuantizedOp = LayerNorm<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        _data: &S::AuxData,
        _node_id: NodeId,
        input_scaling: &[ScalingFactor],
        _unpadded_input_shapes: &[Shape],
        output_scalings: &[ScalingFactor],
        _unpadded_output_shapes: &[Shape],
    ) -> Result<QuantizeOutput<Self::QuantizedOp>> {
        // First check we have one input_scaling
        ensure!(
            input_scaling.len() == 1,
            "Could not quantise LayerNorm, too many input scaling factors {}, expected 1",
            input_scaling.len()
        );

        ensure!(
            output_scalings.len() == 1,
            "Could not quantise LayerNorm, too many output scaling factors {}, expected 1",
            output_scalings.len()
        );

        let input_scaling_factor = input_scaling[0];

        // We want to calculate the largest quantisation domain that will allow us to fit the magnitude normalisation check into
        // a single shift check table. This means we need twice the error bound to have a bit length <= SHIFT_CHECK_TABLE_BIT_SIZE.
        // The error bound is normalisation_dim_size * (1.0 / normalisation_scaling_factor + 0.25) so we can rearrange to find the normalisation_scaling_factor that will allow us to fit into the table.
        // The rearrangement gives us (2^SHIFT_CHECK_TABLE_BIT_SIZE / normalisation_dim_size) - 0.25 >= 1.0 / normalisation_scaling_factor, so normalisation_scaling_factor >= 1.0 / ((2^SHIFT_CHECK_TABLE_BIT_SIZE / normalisation_dim_size) - 0.25).
        let minimum_normalising_scale = 2.0
            / ((1u64 << (SHIFT_CHECK_TABLE_BIT_SIZE - 1)) as f32
                / self.normalisation_dim_size as f32
                - 1.0);

        // The scale is calculated as (float_max - float_min) / (quant_max - quant_min)
        let mut normalisation_bits = *quantization::BIT_LEN;
        let norm_max = (self.normalisation_dim_size() as f32).sqrt();
        let norm_min = -norm_max;

        let mut test_scale = (norm_max - norm_min) / ((1 << normalisation_bits) as f32);
        while test_scale < minimum_normalising_scale && normalisation_bits > 1 {
            normalisation_bits -= 1;
            test_scale = (norm_max - norm_min) / ((1 << normalisation_bits) as f32);
        }

        let norm_quant_min: Element = -1 << (normalisation_bits - 1);
        let norm_quant_max: Element = (1 << (normalisation_bits - 1)) - 1;
        let normalisation_scaling_factor = ScalingFactor::from_parts(
            norm_max,
            norm_min,
            test_scale,
            (norm_quant_min, norm_quant_max),
        );

        let output_scaling = output_scalings[0];

        self.quantise(
            input_scaling_factor,
            normalisation_scaling_factor,
            output_scaling,
        )
    }
}

impl PadOp for LayerNorm<Element> {
    fn pad_node(self, _si: &mut crate::padding::ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        // LayerNorm does not need any special padding handling so
        // we just pad the keyed tensors
        Ok(Self {
            gamma: self.gamma,
            beta: self.beta,
            eps: self.eps,
            normalisation_dim_size: self.normalisation_dim_size,
            mean_scaling_factor: self.mean_scaling_factor,
            normalisation_scaling_factor: self.normalisation_scaling_factor,
            cache: Arc::new(Mutex::new(NormalisationCache::new())),
        })
    }
}

impl LayerNorm<f32> {
    pub fn from_json(l: &json::FileTensorLoader, _c: &LLMConfig) -> anyhow::Result<Self> {
        trace!("from_json: current path: {:?}", l.prefix);
        let gamma = l.get_tensor("norm.weight")?;
        let beta = l.get_tensor("norm.bias")?;
        let eps = l.metadata_to_f32("norm_epsilon")?;
        Self::new(gamma.into(), beta.into(), eps)
    }
    // Replaces from_var_builder and from_tensor_loader
    // The 'loader' passed here is expected to be pre-scoped by the caller
    // (e.g., loader.pp("attn_") or loader.pp("ffn_"))
    pub fn from_gguf(loader: &gguf::FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let gamma = loader.get_tensor("norm.weight")?;
        let beta = loader.get_tensor("norm.bias")?;
        ensure!(
            gamma.shape().as_ref() == &[c.embedding_size],
            "norm_gamma must have shape [{}] vs given {:?}",
            c.embedding_size,
            gamma.shape()
        );
        ensure!(
            beta.shape().as_ref() == &[c.embedding_size],
            "norm_beta must have shape [{}] vs given {:?}",
            c.embedding_size,
            beta.shape()
        );
        let eps = loader
            .metadata::<f32>(&loader.norm_epsilon_key(&c.model_name, NormType::LayerNorm))
            .context("norm_epsilon not found")?;
        Self::new(gamma.into(), beta.into(), eps)
    }

    pub fn from_safetensors(
        loader: &safe::FileTensorLoader,
        config: &safe::ConfigJSON,
        c: &LLMConfig,
    ) -> anyhow::Result<Self> {
        let gamma = loader.get_tensor("norm.weight")?;
        let beta = loader.get_tensor("norm.bias")?;
        let eps = config
            .get::<f32, _>("norm_epsilon")
            .context("norm_epsilon not found")?;
        ensure!(
            gamma.shape().as_ref() == &[c.embedding_size],
            "norm_gamma must have shape [{}] vs given {:?}",
            c.embedding_size,
            gamma.shape()
        );
        ensure!(
            beta.shape().as_ref() == &[c.embedding_size],
            "norm_beta must have shape [{}] vs given {:?}",
            c.embedding_size,
            beta.shape()
        );
        Self::new(gamma.into(), beta.into(), eps)
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::Rng;
    use tenstore::GenStore;

    use crate::{
        init_test_logging,
        layers::{Layer, einsum::EinSum},
        model::{Model, test::prove_model},
        rng_from_env_or_random,
        tensor::KeyedTensor,
    };

    use super::*;

    #[test]
    fn test_layernorm_proving() {
        init_test_logging("debug");
        for _ in 0..25 {
            let Input {
                weight,
                bias,
                input: random_input,
            } = Input::random(25, 25);

            let input_rank = random_input.shape().rank();
            let equation = match input_rank {
                1 => "I(j)@W(ij)->O(i)+BIAS(i)",
                2 => "I(aj)@W(ij)->O(ai)+BIAS(i)",
                3 => "I(abj)@W(ij)->O(abi)+BIAS(i)",
                4 => "I(abcj)@W(ij)->O(abci)+BIAS(i)",
                _ => panic!("Input rank too high for test"),
            }
            .to_string();
            let dense = EinSum::<f32>::new(
                equation.to_owned(),
                vec![Some(weight.into())],
                vec![Some(bias.into())],
            )
            .unwrap()
            .no_requant();

            let output_shape = dense
                .output_shapes(
                    std::slice::from_ref(random_input.shape()),
                    PaddingMode::NoPadding,
                )
                .unwrap()[0]
                .clone();

            let final_dims = output_shape.dim(-1);
            let weight_shape = Shape::new(vec![final_dims]);
            let gamma = Tensor::<f32>::random(&weight_shape);
            let beta = Tensor::<f32>::random(&weight_shape);

            let layernorm = LayerNorm::<f32>::new(
                KeyedTensor::new("LNRM_GAMMA".to_string(), gamma).into(),
                KeyedTensor::new("LNRM_BETA".to_string(), beta).into(),
                1e-6,
            )
            .unwrap();

            let mut model = Model::new_from_input_shapes(vec![random_input.shape().clone()]);

            let dense_id = model
                .add_consecutive_layer(Layer::EinSum(dense), None)
                .unwrap();

            let _ = model
                .add_consecutive_layer(Layer::LayerNorm(layernorm), Some(dense_id))
                .unwrap();

            model.automatic_output_labelling().unwrap();
            model.describe();
            prove_model(model, &mut GenStore::default()).unwrap();
        }
    }

    #[derive(Clone, Debug)]
    struct Input {
        weight: KeyedTensor<f32>,
        bias: KeyedTensor<f32>,
        input: Tensor<f32>,
    }

    impl Input {
        fn random(rows_max: usize, columns_max: usize) -> Input {
            let mut rng = rng_from_env_or_random();
            let rows = rng.gen_range(8..rows_max);
            let columns = rng.gen_range(8..columns_max);
            let matrix_size = rows * columns;
            let weight_data: Vec<f32> = (0..matrix_size)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect();
            let bias_data: Vec<f32> = (0..rows).map(|_| rng.gen_range(-10.0..10.0)).collect();

            let input_rank = rng.gen_range(2usize..=4);

            let mut all_dims: Vec<usize> =
                (0..(input_rank - 1)).map(|_| rng.gen_range(3..8)).collect();
            all_dims.push(columns);

            let total_data_size = all_dims.iter().product::<usize>();
            let input_shape = Shape::from(all_dims);
            let input_data: Vec<f32> = (0..total_data_size)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect();

            Input {
                weight: KeyedTensor::new(
                    "W".to_string(),
                    Tensor::new(vec![rows, columns].into(), weight_data).unwrap(),
                ),
                bias: KeyedTensor::new(
                    "BIAS".to_string(),
                    Tensor::new(vec![rows].into(), bias_data).unwrap(),
                ),
                input: Tensor::new(input_shape, input_data).unwrap(),
            }
        }
    }
}
