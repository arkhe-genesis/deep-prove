//! Module defining the [`RMSNorm`] layer.
use super::*;

use std::collections::HashMap;

use crate::{
    NextPowerOfTwo,
    iop::context::ContextAux,
    layers::{
        LayerCtx,
        provable::{Evaluate, OpInfo, PadOp, ProvableOp, ProveInfo, VerifiableCtx},
        requant::FIXED_POINT_SCALE,
        transformer::normalisation::rmsnorm::verify::RMSNormLookupVerifier,
    },
    lookup::{
        operation::{LookupOp, decomposer::ChunkingInfo, variant::LookupVariant},
        table::{SHIFT_CHECK_TABLE_BIT_SIZE, Table},
    },
    padding::PaddingMode,
    quantization,
    tensor::{CommitmentId, TensorTypeParam},
};

pub mod evaluate;
mod lookup;
pub mod prove;
pub mod quantise;
mod verify;

/// The short name used to identify the RMSNorm layer.
pub(crate) const RMSNORM_LAYER: &str = "RMSN";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "N: Serialize", deserialize = "N: DeserializeOwned"))]
/// Struct storing all information needed to perform RMSNorm. The `alpha` field
/// is normally learned parameters that are applied elementwise. The `eps` field is used for normalisation when calculating
/// the inverse square root.
pub struct RMSNorm<N: TensorTypeParam> {
    /// Each element of the normalisation dimension is multiplied elementwise by this, we use
    /// an [`Option`] because it may be the case the weights are all 1 and then we don't want to apply this tensor.
    pub alpha: Option<TensorHandle<N>>,
    /// Normalisation factor
    pub eps: f32,
    /// The unpadded size of the normalisation dimension
    pub normalisation_dim_size: usize,
    /// The [`ScalingFactor`] used during quantized evaluation for the input.
    pub input_scaling_factor: Option<ScalingFactor>,
    /// The [`ScalingFactor`] used during quantized evaluation for the output of the normalisation.
    pub normalisation_scaling_factor: Option<ScalingFactor>,
    /// Caches the current max shift seen during quantised evaluation. Needed to ensure result of running full sequence is consistent with running
    /// token by token
    cache: Arc<Mutex<NormalisationCache>>,
}

impl<N: TensorTypeParam> RMSNorm<N> {
    /// Create a new [`RMSNorm`] layer with the given `alpha` and `eps` values.
    pub fn new(
        alpha: Option<TensorHandle<N>>,
        eps: f32,
        normalisation_dim_size: Option<usize>,
    ) -> Result<Self> {
        if alpha.is_none() && normalisation_dim_size.is_none() {
            return Err(anyhow::anyhow!(
                "Must provide either alpha or normalisation_dim_size"
            ));
        }
        // Unwrap is safe because we check one of alpha or normalisation_dim_size is Some
        let normalisation_dim_size =
            normalisation_dim_size.unwrap_or_else(|| alpha.as_ref().map(|a| a.shape()[0]).unwrap());
        let alpha = alpha
            .map(|inner| inner.wrapped_tensor_variant())
            .transpose()?;
        Ok(Self {
            alpha,
            eps,
            normalisation_dim_size,
            input_scaling_factor: None,
            normalisation_scaling_factor: None,
            cache: Arc::new(Mutex::new(NormalisationCache::new())),
        })
    }

    /// Returns the size of the dimension we normalise over.
    pub fn normalisation_dim_size(&self) -> usize {
        self.normalisation_dim_size
    }

    /// Getter for the quantisation scaling factors
    pub fn get_quantisation_scaling_factors(&self) -> Option<(&ScalingFactor, &ScalingFactor)> {
        match (
            &self.input_scaling_factor,
            &self.normalisation_scaling_factor,
        ) {
            (Some(input), Some(output)) => Some((input, output)),
            _ => None,
        }
    }

    pub fn reset_cache(&self) {
        self.cache.lock().unwrap().reset();
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RMSNormCtx {
    /// The [`NodeId`] for this operation
    node_id: NodeId,
    /// The size of the dimension we normalise over (unpadded)
    normalisation_dim_size: usize,
    /// The [`CommitmentId`] for the `alpha` parameter
    alpha_key: Option<CommitmentId>,
    /// The [`ScalingFactor`] used for the input
    input_scaling: ScalingFactor,
    /// The [`ScalingFactor`] used for the intermediate normalised values
    normalisation_scaling: ScalingFactor,
}

impl RMSNormCtx {
    /// Getter for the lookup tables needed for RMSNorm
    pub fn lookup_tables(&self) -> Vec<Table> {
        vec![
            Table::new_shift_check(),
            Table::new_normalisation(self.normalisation_scaling().bit_size() + 1),
            Table::new_signed_zero_check(),
        ]
    }

    /// Getter for the normalisation dimension size
    pub fn normalisation_dim_size(&self) -> usize {
        self.normalisation_dim_size
    }

    /// Getter for the input scaling factor
    pub fn input_scaling(&self) -> &ScalingFactor {
        &self.input_scaling
    }

    /// Getter for the normalisation scaling factor
    pub fn normalisation_scaling(&self) -> &ScalingFactor {
        &self.normalisation_scaling
    }

    /// Getter for the alpha commitment id
    pub fn alpha_key(&self) -> Option<&CommitmentId> {
        self.alpha_key.as_ref()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Stores the unpadded normalisation tensor for a RMSNorm layer, to be used in proving.
pub struct RMSNormProvingData {
    /// This is a tensor containing the normalisation values.
    pub normalisation: WrappedTensor<Element>,
    /// Internal [`RMSNormLookupVerifier`]
    pub(crate) lookup_verifier: RMSNormLookupVerifier,
}

impl RMSNormProvingData {
    pub fn new(
        normalisation: WrappedTensor<Element>,
        right_shift: usize,
        normalised_sumsq: Element,
        error_bound: Element,
        intermediate_bit_size: usize,
        has_weight: bool,
    ) -> Self {
        Self {
            normalisation,
            lookup_verifier: RMSNormLookupVerifier::new_from_parts(
                right_shift as Element,
                normalised_sumsq,
                error_bound,
                intermediate_bit_size,
                has_weight,
            ),
        }
    }
}

impl LookupOp for RMSNormProvingData {
    fn intermediate_bit_size(&self) -> usize {
        self.lookup_verifier.intermediate_bit_size()
    }

    fn right_shift(&self) -> usize {
        self.lookup_verifier.right_shift()
    }

    fn apply(
        &self,
        input: WrappedTensor<Element>,
        table: &Table,
    ) -> Result<WrappedTensor<Element>> {
        // Apply the fixed point multipliers, rounding and right shift
        let rescaled_input = input
            .mul(self.normalisation.clone())?
            .add_scalar(self.rounding_constant())
            .bitwise_right_shift_scalar(self.right_shift() as Element);
        // Perform the lookup, the table will handle clamping internally
        table.lookup_tensor(rescaled_input)
    }

    fn variant(&self) -> LookupVariant {
        self.lookup_verifier.variant()
    }

    fn chunking_info(&self, table: &Table) -> Result<ChunkingInfo> {
        self.lookup_verifier.chunking_info(table)
    }

    fn fixed_point_multiplier(&self) -> Element {
        self.lookup_verifier.fixed_point_multiplier()
    }

    fn is_signed(&self) -> bool {
        self.lookup_verifier.is_signed()
    }

    fn padding_value(&self) -> Element {
        self.lookup_verifier.padding_value()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Proof for correct execution of a quantised [`RMSNorm`] operation.
pub struct RMSNormProof<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    /// The LogUp proofs for RMSNorm, this is the error check on the normalised output.
    pub(crate) logup_proof: LogUpBatchProof<E>,
    /// This is the size of the right shift for the RMSNorm operation
    pub(crate) right_shift: Element,
    /// Witness commitments for this layer
    pub(crate) commitment: PCS::Commitment,
    /// The IO proof that links all claims to `last_claim` and the input
    pub(crate) io_proof: IOPProof<E>,
    /// The claimed evaluations of the commitments
    pub(crate) io_evaluations: Vec<E>,
    /// The (Optional) evaluation of alpha if it is used
    pub(crate) alpha_evaluation: Option<E>,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> RMSNormProof<E, PCS> {
    pub(crate) fn write_commitment<T: Transcript<E>>(&self, transcript: &mut T) -> Result<()> {
        PCS::write_commitment(&self.commitment, transcript).map_err(|e| anyhow!("{e:?}"))
    }
}

impl OpInfo for RMSNormCtx {
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
            "RMSNormCtx(dimension size: {})",
            self.normalisation_dim_size,
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<N: TensorTypeParam> OpInfo for RMSNorm<N> {
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
        format!("RMSNorm(dimension size: {})", self.normalisation_dim_size())
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl ProveInfo for RMSNorm<Element> {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeId,
        mut aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)> {
        // Check that the quantisation scaling factors are present
        let (input_scaling, normalisation_scaling) = self
            .get_quantisation_scaling_factors()
            .ok_or_else(|| anyhow::anyhow!("Quantisation scaling factors not set for RMSNorm"))?;

        let alpha_key = match self.alpha.as_ref() {
            Some(alpha) => {
                let padded_alpha = alpha.pad_next_power_of_two();
                let alpha_tensor = padded_alpha.wrapped_tensor()?;
                let alpha_evals = alpha_tensor.get_data();
                let mut model_polys = HashMap::new();
                model_polys.insert(CommitmentId::from(alpha.storage_key()), alpha_evals);
                aux.model_polys = Some(model_polys);
                Some(CommitmentId::from(alpha.storage_key()))
            }
            None => None,
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
            LayerCtx::RMSNorm(RMSNormCtx {
                node_id: id,
                normalisation_dim_size: self.normalisation_dim_size,
                alpha_key,
                input_scaling: *input_scaling,
                normalisation_scaling: *normalisation_scaling,
            }),
            aux,
        ))
    }
}

impl<E, PCS> ProvableOp<E, PCS> for RMSNorm<Element>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = RMSNormCtx;

    fn prove<'a, 'b, 'c, 'd, T: transcript::Transcript<E>>(
        &'a self,
        node_id: NodeId,
        _ctx: &'b Self::Ctx,
        last_claims: Vec<&crate::Claim<E>>,
        step_data: &crate::model::Step<Element>,
        prover: &mut Prover<'c, 'd, E, T, PCS>,
    ) -> Result<Vec<crate::Claim<E>>> {
        self.prove_internal(node_id, last_claims[0], step_data, prover)
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &crate::ProverContext<E, PCS>,
        step_data: &crate::model::Step<Element>,
    ) -> Result<crate::lookup::context::LookupWitnessGen<E, PCS>> {
        self.lookup_witness(id, ctx, step_data)
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for RMSNormCtx
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = RMSNormProof<E, PCS>;

    fn verify<T: transcript::Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<crate::Claim<E>>> {
        self.verify_internal(proof, self.node_id, verifier, last_claims[0], shape_step)
    }

    fn write_proof_to_transcript<T: transcript::Transcript<E>>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        proof.write_commitment(transcript)
    }
}

impl Evaluate<f32> for RMSNorm<f32> {
    fn evaluate(&self, inputs: &[&WrappedTensor<f32>]) -> Result<LayerOut<f32>> {
        self.evaluate_float_internal(inputs)
    }
}

impl Evaluate<Element> for RMSNorm<Element> {
    fn evaluate(&self, inputs: &[&WrappedTensor<Element>]) -> Result<LayerOut<Element>> {
        self.evaluate_quantised_internal(inputs)
    }
}

impl QuantizeOp for RMSNorm<f32> {
    type QuantizedOp = RMSNorm<Element>;

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

impl PadOp for RMSNorm<Element> {
    fn pad_node(self, _si: &mut crate::padding::ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        // RMSNorm does not need any special padding handling
        Ok(Self {
            alpha: self.alpha,
            eps: self.eps,
            normalisation_dim_size: self.normalisation_dim_size,
            input_scaling_factor: self.input_scaling_factor,
            normalisation_scaling_factor: self.normalisation_scaling_factor,
            cache: Arc::new(Mutex::new(NormalisationCache::new())),
        })
    }
}

impl RMSNorm<f32> {
    pub fn from_json(l: &json::FileTensorLoader, _c: &LLMConfig) -> anyhow::Result<Self> {
        trace!("from_json: current path: {:?}", l.prefix);
        let alpha = l.get_tensor("norm.weight")?;
        let eps = l.metadata_to_f32("norm_epsilon")?;

        // If alpha is all ones we can just set it to None
        let trivial_alpha = alpha.data().iter().all(|&x| x == 1.0);

        if trivial_alpha {
            Self::new(None, eps, Some(alpha.shape().dim(-1)))
        } else {
            Self::new(Some(alpha.into()), eps, None)
        }
    }

    // Replaces from_var_builder and from_tensor_loader
    // The 'loader' passed here is expected to be pre-scoped by the caller
    // (e.g., loader.pp("attn_") or loader.pp("ffn_"))
    pub fn from_gguf(loader: &gguf::FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let alpha = loader.get_tensor("norm.weight")?;

        // we can have any checks on the shape alpha here since it depends of the context
        // a RMSNorm after  Q doesn't have the same shape as a RMSNorm after K or inside FeedForward etc
        let eps = loader
            .metadata::<f32>(&loader.norm_epsilon_key(&c.model_name, NormType::RMSNorm))
            .unwrap_or_default();

        // If alpha is all ones or zeroes we can just set it to None
        let trivial_alpha = alpha.data().iter().all(|&x| x == 1.0 || x == 0.0f32);

        if trivial_alpha {
            Self::new(None, eps, Some(alpha.shape().dim(-1)))
        } else {
            Self::new(Some(alpha.into()), eps, None)
        }
    }

    pub fn from_safetensors(
        loader: &safe::FileTensorLoader,
        config: &safe::ConfigJSON,
    ) -> anyhow::Result<Self> {
        let alpha = loader.get_tensor("norm.weight")?;

        let eps = config
            .get::<f32, _>("rms_norm_eps")
            .context("norm_epsilon not found")?;
        // If alpha is all ones or zeroes we can just set it to None
        let trivial_alpha = alpha.data().iter().all(|&x| x == 1.0 || x == 0.0f32);

        if trivial_alpha {
            Self::new(None, eps, Some(alpha.shape().dim(-1)))
        } else {
            Self::new(Some(alpha.into()), eps, None)
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::Rng;
    use tenstore::GenStore;

    use crate::{
        layers::{Layer, einsum::EinSum},
        model::{Model, test::prove_model},
        rng_from_env_or_random,
        tensor::KeyedTensor,
    };

    use super::*;

    #[test]
    fn test_rmsnorm_proving() {
        for i in 0..25 {
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
            let rmsnorm = if i % 2 == 0 {
                RMSNorm::<f32>::new(None, 1e-6, Some(final_dims)).unwrap()
            } else {
                let alpha_shape = Shape::new(vec![final_dims]);
                let alpha = Tensor::<f32>::random(&alpha_shape);
                RMSNorm::<f32>::new(
                    Some(KeyedTensor::new("RMSN_ALPHA".to_string(), alpha).into()),
                    1e-6,
                    None,
                )
                .unwrap()
            };

            let mut model = Model::new_from_input_shapes(
                vec![random_input.shape().clone()],
            );

            let dense_id = model
                .add_consecutive_layer(Layer::EinSum(dense), None)
                .unwrap();

            let _ = model
                .add_consecutive_layer(Layer::RMSNorm(rmsnorm), Some(dense_id))
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
