use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use crate::{
    parser::llm::transformer::expand,
    tensor::{KeyedTensor, TensorKey},
};
use anyhow::{Ok, bail, ensure};
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tenstore::GenStore;
use transcript::Transcript;

use crate::{
    Claim, Element, Prover, ScalingFactor, ScalingStrategy, Shape, Tensor,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx,
        add::Add,
        provable::{
            Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp,
            QuantizeOutput, VerifiableCtx,
        },
        transformer::positional::{
            absolute::{Absolute, AbsoluteCtx, AbsoluteProof},
            rope::{Rope, RopeCtx, RopeProof},
        },
    },
    model::StepData,
    number::Number,
    padding::{PaddingMode, ShapeInfo},
    parser::{
        gguf::FileTensorLoader,
        json,
        llm::{LLMConfig, LLMVariant},
    },
    quantization::{Fieldizer, TensorFielder},
    tensor::{IntoBTensor, TensorSlice},
};

pub(crate) mod absolute;
pub(crate) mod rope;

/// The short name used to identify the positional layer.
pub const POSITIONAL_LAYER: &str = "POSI";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PositionalCtx {
    Absolute(AbsoluteCtx),
    Rope(RopeCtx),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub enum PositionalProof<E: ExtensionField> {
    // Proofs for all the inputs of the positional encoding layer
    Absolute(AbsoluteProof<E>),
    Rope(RopeProof<E>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PositionalCache {
    seq_len: usize,
    initialized: bool,
}

impl PositionalCache {
    fn new() -> Self {
        Self {
            seq_len: 0,
            initialized: false,
        }
    }
    fn set_seq_len(&mut self, seq_len: usize) -> anyhow::Result<()> {
        if !self.initialized {
            self.seq_len = seq_len;
            self.initialized = true;
        } else {
            ensure!(
                self.seq_len + 1 == seq_len,
                "with caching, new seq_len should be old one + 1 = {} + 1 vs given {}",
                self.seq_len,
                seq_len
            );
            self.seq_len = seq_len;
        }
        Ok(())
    }
    fn reset(&mut self) {
        self.seq_len = 0;
        self.initialized = false;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Positional<N> {
    cache: Option<Arc<Mutex<PositionalCache>>>,
    pub(crate) variant: PositionalVariant<N>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum PositionalVariant<N> {
    Absolute(Absolute<N>),
    Rope(Rope<N>),
}

impl<N: Number> Positional<N> {
    pub fn shape(&self) -> Shape {
        match &self.variant {
            PositionalVariant::Absolute(abs) => abs.positional.shape().clone(),
            PositionalVariant::Rope(rope) => rope.cosine_matrix.shape().clone(),
        }
    }

    pub(crate) fn reset_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.lock().unwrap().reset();
        }
    }

    pub(crate) fn new_from_variant(variant: PositionalVariant<N>) -> Self {
        let cache = Arc::new(Mutex::new(PositionalCache::new()));
        Self {
            cache: Some(cache),
            variant,
        }
    }

    pub fn new_absolute(matrix: KeyedTensor<N>) -> Self {
        Self::new_from_variant(PositionalVariant::Absolute(Absolute::new(matrix)))
    }

    pub fn new_rope(
        angles: Vec<f32>,
        base_frequency_id: TensorKey,
        max_content_length: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self::new_from_variant(PositionalVariant::Rope(
            Rope::build_from_angles(angles, base_frequency_id, max_content_length)?,
        )))
    }

    pub fn new_rope_from_frequency(
        base_frequency: f32,
        base_frequency_id: TensorKey,
        head_size: usize,
        max_content_length: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self::new_from_variant(PositionalVariant::Rope(
            Rope::build_from_frequency(
                base_frequency,
                base_frequency_id,
                head_size,
                max_content_length,
            )?,
        )))
    }

    pub fn new_rope_from_matrices(
        cosine_matrix: KeyedTensor<N>,
        sine_matrix: KeyedTensor<N>,
    ) -> anyhow::Result<Self> {
        Ok(Self::new_from_variant(PositionalVariant::Rope(Rope::new(
            cosine_matrix,
            sine_matrix,
        )?)))
    }

    /// Disable caching for `self` positional layer
    pub fn with_no_cache(self) -> Self {
        Self {
            cache: None,
            ..self
        }
    }

    // Sample from the transcript `t` `num_coordinates` random coordinates to be employed
    // for proving. It requires as input also the claim related to the output polynomial and
    // the claim about the sub-matrix of positional matrix being actually added to the
    // input polynomial
    fn sample_random_coordinates<E: ExtensionField, T: Transcript<E>>(
        num_coordinates: usize,
        t: &mut T,
        output_claim: &Claim<E>,
        sub_matrix_claim: &Claim<E>,
    ) -> Vec<E> {
        // first, we add `output_claim` and `sub_pos_claim` to the transcript
        t.append_field_element_exts(&output_claim.point);
        t.append_field_element_ext(&output_claim.eval);
        t.append_field_element_exts(&sub_matrix_claim.point);
        t.append_field_element_ext(&sub_matrix_claim.eval);

        // then, we get `num_coordinates` challenges
        (0..num_coordinates)
            .map(|_| t.read_challenge().elements)
            .collect()
    }

    // Compute a claim about the positional encoding matrix at point `evaluation_point`.
    // The claim is computed from the following evaluations:
    // - `sub_pos_eval`: Evaluation of the sub-matrix that is added to the input tensor
    // - `sub_matrix_evals`: Evaluations of all the sub-matrices employed to derive the claim
    // about the positional encoding matrix
    // Note that the sub-matrices the evaluations are referred to corresponds to a partition
    // of the positional encoding matrix, that is their concatenation yields the positional
    // encoding matrix itself
    fn compute_positional_matrix_claim<E: ExtensionField>(
        evaluation_point: Vec<E>,
        sub_pos_eval: E,
        sub_matrix_evals: &[E],
    ) -> Claim<E> {
        // compute the slice of the evaluation point to be used to compute the claim
        let eval_point = {
            let diff_vars = sub_matrix_evals.len();
            let start_extra_coordinates = evaluation_point.len() - diff_vars;
            &evaluation_point[start_extra_coordinates..]
        };
        let positional_matrix_eval = sub_matrix_evals
            .iter()
            .zip(eval_point)
            .fold(sub_pos_eval, |eval, (&sub_eval, &coordinate)| {
                eval * (E::ONE - coordinate) + sub_eval * coordinate
            });
        Claim::new(evaluation_point, positional_matrix_eval)
    }

    /// Build a claim for the positional matrix provided as input, starting from a claim
    /// `sub_matrix_claim` on the sub-matrix given by the first `input_num_rows` of positional matrix;
    /// it returns the claim for the positional matrix and the list of evaluations for the sub-matrices
    /// employed to build the claim, which have to be included in the proof
    fn bind_sub_claim_to_positional_matrix<'a, E: ExtensionField, T: Transcript<E>>(
        sub_matrix_claim: Claim<E>,
        output_claim: &Claim<E>,
        matrix_slice: &TensorSlice<'a, N>,
        positional_matrix: &Tensor<N>,
        input_num_rows: usize,
        transcript: &mut T,
    ) -> anyhow::Result<(Vec<E>, Claim<E>)>
    where
        N: Fieldizer<E>,
    {
        // first, we compute the number of variables that we need to fill to get to the `positional_matrix`
        // polynomial
        let num_vars = {
            let num_vars = positional_matrix.shape().num_vars_2d();
            num_vars.0 + num_vars.1
        };

        let sub_pos_vars = sub_matrix_claim.point.len();
        let diff_vars = num_vars - sub_pos_vars;

        ensure!(diff_vars >= 0);

        // now, we need to squeeze `diff_vars` coordinates from the transcript
        let extra_coordinates = Positional::<Element>::sample_random_coordinates(
            diff_vars,
            transcript,
            output_claim,
            &sub_matrix_claim,
        );

        let sub_pos_eval = sub_matrix_claim.eval;

        let evaluation_point = sub_matrix_claim
            .point
            .into_iter()
            .chain(extra_coordinates)
            .collect_vec();

        let mut slice_start = input_num_rows;
        let sub_matrices = (0..diff_vars)
            .map(|_| {
                let sub_matrix = matrix_slice.slice_over_first_dim(slice_start, slice_start * 2);
                slice_start *= 2;
                sub_matrix
            })
            .collect_vec();

        // check that all the slices of `positional_matrix` have been computed
        ensure!(slice_start == positional_matrix.shape()[0]);

        // now, evaluate the MLE of each sub-matrix
        let sub_matrix_evals = (0..diff_vars)
            .into_par_iter()
            .map(|i| {
                (
                    i,
                    sub_matrices[i]
                        .to_fields()
                        .to_mle_2d()
                        .evaluate(&evaluation_point[..sub_pos_vars + i]),
                )
            })
            .collect::<BTreeMap<_, _>>()
            .into_values()
            .collect::<Vec<_>>();

        let positional_matrix_claim = Positional::<Element>::compute_positional_matrix_claim(
            evaluation_point,
            sub_pos_eval,
            &sub_matrix_evals,
        );

        Ok((sub_matrix_evals, positional_matrix_claim))
    }
}

impl<N: Number> Evaluate<N> for Positional<N>
where
    Add<N>: Evaluate<N>,
    Tensor<N>: IntoBTensor,
    N: burn::tensor::Element,
{
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<N>],
        unpadded_input_shapes: &[Shape],
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(
            inputs.len() == 1,
            "Positional layer expects 1 input, got {}",
            inputs.len()
        );
        ensure!(
            inputs.iter().all(|x| x.shape().len() == 2),
            "positional embeddings only support 2d tensors"
        );
        ensure!(
            unpadded_input_shapes.len() == 1,
            "Expected 1 input shape for positional layer, found {}",
            unpadded_input_shapes.len(),
        );

        // default cache to be provided in case the layer was not initialized with a cache
        let new_cache = Arc::new(Mutex::new(PositionalCache::new()));

        match &self.variant {
            PositionalVariant::Absolute(absolute) => absolute.evaluate(
                inputs[0],
                &unpadded_input_shapes[0],
                self.cache.as_ref().unwrap_or(&new_cache),
            ),
            PositionalVariant::Rope(rope) => rope.evaluate(
                inputs[0],
                &unpadded_input_shapes[0],
                self.cache.as_ref().unwrap_or(&new_cache),
            ),
        }
    }
}

fn output_shapes(
    input_shapes: &[Shape],
    pos_matrix_unpadded_shape: &Shape,
    padding_mode: PaddingMode,
) -> Vec<Shape> {
    let pos_shape = match padding_mode {
        PaddingMode::NoPadding => pos_matrix_unpadded_shape.clone(),
        PaddingMode::Padding => pos_matrix_unpadded_shape.next_power_of_two(),
    };

    input_shapes.iter().for_each(|s| {
        assert!(s.is_matrix());
        assert!(s[0] <= pos_shape[0]);
        assert_eq!(s[1], pos_shape[1])
    });
    input_shapes.to_vec()
}

impl<N: Number> OpInfo for Positional<N> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        match &self.variant {
            PositionalVariant::Absolute(absolute) => {
                output_shapes(input_shapes, &absolute.unpadded_shape, padding_mode)
            }
            PositionalVariant::Rope(rope) => {
                output_shapes(input_shapes, &rope.unpadded_shape, padding_mode)
            }
        }
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        format!("Positional({:?}x{:?})", self.shape()[0], self.shape()[1])
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl ProveInfo for Positional<Element> {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeId,
        aux: ContextAux,
    ) -> anyhow::Result<(LayerCtx<E>, ContextAux)> {
        let (ctx, aux) = match &self.variant {
            PositionalVariant::Absolute(absolute) => {
                let (ctx, aux) = absolute.step_info::<E>(id, aux)?;
                (PositionalCtx::Absolute(ctx), aux)
            }
            PositionalVariant::Rope(rope) => {
                let (ctx, aux) = rope.step_info(id, aux)?;
                (PositionalCtx::Rope(ctx), aux)
            }
        };

        Ok((LayerCtx::Positional(ctx), aux))
    }
}

impl QuantizeOp for Positional<f32> {
    type QuantizedOp = Positional<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        ensure!(
            input_scaling.len() == 1,
            "Expected 1 input scaling factor for positional layer, found {}",
            input_scaling.len()
        );
        // re-initialize the cache for quantized node, if the original node has a cache enabled
        let new_cache = self
            .cache
            .map(|_| Arc::new(Mutex::new(PositionalCache::new())));

        let quantized_op = match self.variant {
            PositionalVariant::Absolute(abs) => {
                let quantized_abs = abs.quantize::<S>(data, node_id, input_scaling[0])?;
                QuantizeOutput {
                    quantized_op: Positional {
                        cache: new_cache,
                        variant: PositionalVariant::Absolute(quantized_abs.quantized_op),
                    },
                    output_scalings: quantized_abs.output_scalings,
                    requant_layer: quantized_abs.requant_layer,
                    post_quant_rule: None,
                }
            }
            PositionalVariant::Rope(rope) => {
                let quantized_rope = rope.quantize::<S>(data, node_id, input_scaling[0])?;
                QuantizeOutput {
                    quantized_op: Positional {
                        cache: new_cache,
                        variant: PositionalVariant::Rope(quantized_rope.quantized_op),
                    },
                    output_scalings: quantized_rope.output_scalings,
                    requant_layer: quantized_rope.requant_layer,
                    post_quant_rule: None,
                }
            }
        };

        Ok(quantized_op)
    }
}

impl PadOp for Positional<Element> {
    fn pad_node(self, si: &mut ShapeInfo) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let cache = self
            .cache
            .map(|_| Arc::new(Mutex::new(PositionalCache::new())));
        let padded_variant = match self.variant {
            PositionalVariant::Absolute(pos) => PositionalVariant::Absolute(pos.pad_node(si)?),
            PositionalVariant::Rope(rope) => PositionalVariant::Rope(rope.pad_node(si)?),
        };

        // no need to change `si` since the layer doesn't change the input shapes

        Ok(Self {
            cache,
            variant: padded_variant,
        })
    }
}

impl Positional<f32> {
    pub fn from_loader(loader: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        match c.variant {
            LLMVariant::Gemma3 => {
                let base_freq_id = "gemma3.rope.freq_base";
                let freq_base = loader
                    .metadata::<f32>(base_freq_id)
                    .ok_or(anyhow::anyhow!("gemma3.rope.freq_base not found"))?;
                let head_size = c.head_size;
                let max_content_length = c.max_sequence_length();
                let p = Self::new_rope_from_frequency(
                    freq_base,
                    base_freq_id.to_string().into(),
                    head_size,
                    max_content_length,
                )?;
                let PositionalVariant::Rope(mut rope) = p.variant else {
                    bail!("Expected Rope variant for gemma3");
                };
                rope.cosine_matrix = rope.cosine_matrix.map_tensor(|t| expand(t, c.num_heads));
                rope.sine_matrix = rope.sine_matrix.map_tensor(|t| expand(t, c.num_heads));
                rope.unpadded_shape
                    .set_dim(-1, rope.unpadded_shape.dim(-1) * c.num_heads);
                Ok(Positional::new_from_variant(PositionalVariant::Rope(rope)))
            }
            LLMVariant::GPT2 => {
                let position_embd = loader.get_tensor("position_embd.weight")?;
                let shape = position_embd.shape();
                ensure!(
                    shape[0] == c.context_length,
                    "position_embd must have shape [{}] vs given {:?}",
                    c.context_length,
                    position_embd.shape()
                );
                ensure!(
                    shape[1] == c.embedding_size,
                    "position_embd must have shape [{}] vs given {:?}",
                    c.embedding_size,
                    position_embd.shape()
                );
                Ok(Self::new_absolute(position_embd))
            }
        }
    }
    pub fn from_json(l: &json::FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let position_embd = l.get_tensor("position_embd.weight")?;
        ensure!(position_embd.rank() == 2, "position_embd must be 2d");
        ensure!(
            position_embd.shape()[0] == c.context_length,
            "position_embd must have shape [0] [{}] vs given {:?}",
            c.context_length,
            position_embd.shape()
        );
        ensure!(
            position_embd.shape()[1] == c.embedding_size,
            "position_embd must have shape [1] [{}] vs given {:?}",
            c.embedding_size,
            position_embd.shape()
        );
        Ok(Self::new_absolute(position_embd))
    }
}

impl<E: ExtensionField, PCS> ProvableOp<E, PCS> for Positional<Element>
where
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = PositionalCtx;

    fn prove<T: Transcript<E>>(
        &self,
        node_id: NodeId,
        _ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
        store: &mut GenStore,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        ensure!(
            last_claims.len() == 1,
            "Expected 1 output claim for positional layer, found {}",
            last_claims.len(),
        );
        ensure!(
            step_data.node_inputs.len() == 1,
            "Expected 1 input in trace for positional layer, found {}",
            step_data.node_inputs.len(),
        );

        let output_claim = last_claims[0];
        match &self.variant {
            PositionalVariant::Absolute(absolute) => {
                absolute.prove_step(node_id, output_claim, step_data, prover, store)
            }
            PositionalVariant::Rope(rope) => {
                rope.prove_step(node_id, output_claim, step_data, prover, store)
            }
        }
    }
}

impl OpInfo for PositionalCtx {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        output_shapes(input_shapes, self.unpadded_shape(), padding_mode)
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        let unpadded_shape = self.unpadded_shape();
        format!(
            "PositionalCtx:({:?}x{:?})",
            unpadded_shape[0], unpadded_shape[1]
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl PositionalCtx {
    fn unpadded_shape(&self) -> &Shape {
        match self {
            PositionalCtx::Absolute(absolute_ctx) => &absolute_ctx.unpadded_shape,
            PositionalCtx::Rope(rope_ctx) => &rope_ctx.unpadded_shape,
        }
    }
    /// Compute the claim for a full positional matrix from a claim `sub_claim` about a sub matrix
    /// and the evaluations for other sub-matrices provided as input.
    fn build_positional_matrix_claim<E: ExtensionField, T: Transcript<E>>(
        sub_claim: Claim<E>,
        output_claim: &Claim<E>,
        num_vars_full_matrix: usize,
        transcript: &mut T,
        sub_matrix_evals: &[E],
    ) -> anyhow::Result<Claim<E>> {
        // first, we compute the number of variables that we need to fill to get to the `positional_matrix`
        // polynomial
        let sub_pos_vars = sub_claim.point.len();
        let diff_vars = num_vars_full_matrix - sub_pos_vars;

        ensure!(diff_vars >= 0);

        // then, we sample the extra coordinates corresponding to these variables
        let extra_coordinates = Positional::<Element>::sample_random_coordinates(
            diff_vars,
            transcript,
            output_claim,
            &sub_claim,
        );

        let sub_pos_eval = sub_claim.eval;

        let evaluation_point = sub_claim
            .point
            .into_iter()
            .chain(extra_coordinates)
            .collect_vec();

        let positional_matrix_claim = Positional::<Element>::compute_positional_matrix_claim(
            evaluation_point,
            sub_pos_eval,
            sub_matrix_evals,
        );

        Ok(positional_matrix_claim)
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> VerifiableCtx<E, PCS>
    for PositionalCtx
{
    type Proof = PositionalProof<E>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        let num_outputs = last_claims.len();

        ensure!(
            shape_step.unpadded_output_shape.len() == num_outputs,
            "Expected {num_outputs} unpadded output shapes for Positional layer verifier, found {}",
            shape_step.unpadded_output_shape.len()
        );

        ensure!(
            shape_step.padded_output_shape.len() == num_outputs,
            "Expected {num_outputs} padded output shapes for Positional layer verifier, found {}",
            shape_step.padded_output_shape.len()
        );

        ensure!(
            shape_step.unpadded_input_shape.len() == num_outputs,
            "Expected {num_outputs} unpadded input shapes for Positional layer verifier, found {}",
            shape_step.unpadded_input_shape.len()
        );

        ensure!(
            shape_step.padded_input_shape.len() == num_outputs,
            "Expected {num_outputs} padded input shapes for Positional layer verifier, found {}",
            shape_step.padded_input_shape.len()
        );

        ensure!(
            last_claims.len() == 1,
            "Expected 1 output claim when verifying positional layer, found {}",
            last_claims.len(),
        );

        let output_claim = last_claims[0];

        match (self, proof) {
            (PositionalCtx::Absolute(absolute_ctx), PositionalProof::Absolute(proof)) => {
                absolute_ctx.verify(proof, verifier, output_claim, shape_step)
            }
            (PositionalCtx::Rope(rope_ctx), PositionalProof::Rope(proof)) => {
                rope_ctx.verify(proof, output_claim, verifier, shape_step)
            }
            _ => bail!("Incompatible ctx and proof types found for positional layer"),
        }
    }

    fn write_proof_to_transcript<T: Transcript<E>>(
        &self,
        _proof: &Self::Proof,
        _transcript: &mut T,
    ) -> anyhow::Result<()> {
        // No commitment so just return Ok(())
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Element, Tensor,
        layers::{
            provable::PadOp,
            transformer::positional::{Positional, PositionalVariant},
        },
        padding::{ShapeData, ShapeInfo},
        tensor::KeyedTensor,
    };

    #[test]
    fn test_positional_padding() {
        let seq_len = 14;
        let context_length = 17;
        let embedding_size = 45;
        let input_shape = vec![seq_len, embedding_size];
        let matrix_shape = vec![context_length, embedding_size];
        let positional_matrix = KeyedTensor::new(
            "absolute_positional_mat",
            Tensor::<Element>::random(&matrix_shape.into()),
        );
        let pos = Positional::new_absolute(positional_matrix.clone());

        let mut si = ShapeInfo::from(vec![ShapeData::new(input_shape.into())].as_slice());

        let padded_layer = pos.pad_node(&mut si).unwrap();

        let PositionalVariant::Absolute(padded_pos) = padded_layer.variant else {
            unreachable!()
        };

        let padded_shape = padded_pos.positional.shape();
        assert_eq!(padded_pos.unpadded_shape, *positional_matrix.shape());
        assert_eq!(*padded_shape, positional_matrix.shape().next_power_of_two());

        // check that padded positional matrix has the same data of original matrix
        for i in 0..padded_shape[0] {
            for j in 0..padded_shape[1] {
                if i < padded_pos.unpadded_shape[0] && j < padded_pos.unpadded_shape[1] {
                    assert_eq!(
                        padded_pos.positional.get_2d(i, j),
                        positional_matrix.get_2d(i, j)
                    );
                } else {
                    assert_eq!(padded_pos.positional.get_2d(i, j), 0);
                }
            }
        }
    }
}
