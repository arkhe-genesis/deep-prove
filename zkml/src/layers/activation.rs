use super::provable::{Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, VerifiableCtx};
use crate::{
    Claim, Element, NextPowerOfTwo, Prover, ProverContext, ScalingFactor, ScalingStrategy, Shape,
    graph::NodeId,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        Layer, LayerCtx, LayerProof,
        provable::{QuantizeOp, QuantizeOutput},
        requant::Requant,
    },
    lookup::{
        context::LookupWitnessGen,
        logup_gkr::structs::LogUpBatchProof,
        operation::{
            LookupOp,
            generic_prove::{GenericLookupProof, LookupProverResult, prove_lookup_op},
            generic_verify::{LookupVerifyResult, verify_lookup_op},
        },
        table::Table,
    },
    model::Step,
    padding::PaddingMode,
    tensor::{Tensor, TensorTypeParam, WrappedModuleFn, WrappedTensor},
};

use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::util::transpose;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{collections::HashMap, marker::PhantomData, ops::Deref};
use sumcheck::structs::IOPProof;

use transcript::Transcript;
use witness::RowMajorMatrix;

pub mod lookup_data;
use lookup_data::ActivationLookupData;

/// The short name used to identify an activation layer.
pub const ACTIVATION_LAYER: &str = "ACTI";

use anyhow::{Result, anyhow, ensure};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Activation<N> {
    /// Plain activation layer where we apply the activation function to each input independently
    Plain(ActivationLayer<N>),
    /// Variant of Activation employed in a Gated Linear Unit (GLU); in this case, the activation layer
    /// has 2 inputs: one is passed through the activation function; the output of the layer is
    /// computed as the element-wise product between the result of the activation function and the
    /// other input. More specifically, for inputs x and y, the output is activation(x) * y
    GLU(ActivationLayer<N>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ActivationLayer<N> {
    Relu(Option<ActivationLookupData>, PhantomData<N>),
    Gelu(Option<ActivationLookupData>, PhantomData<N>),
}

impl<N> ActivationLayer<N> {
    pub fn tracked_input_data_id(&self) -> String {
        match self {
            ActivationLayer::Relu(_, _) => "RELU_IN".to_string(),
            ActivationLayer::Gelu(_, _) => "GELU_IN".to_string(),
        }
    }

    pub fn tracked_output_data_id(&self) -> String {
        match self {
            ActivationLayer::Relu(_, _) => "RELU_OUT".to_string(),
            ActivationLayer::Gelu(_, _) => "GELU_OUT".to_string(),
        }
    }
}

/// Currently holds the poly info for the output polynomial of the RELU
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationCtx {
    pub op: Activation<Element>,
    pub node_id: NodeId,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct ActivationProof<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    /// proof for the accumulation of the claim from m2v + claim from lookup for the same poly
    /// e.g. the "link" between a m2v and relu layer
    pub(crate) io_accumulation: IOPProof<E>,
    /// The evaluations output by the linking sumcheck
    pub(crate) evaluations: Vec<E>,
    /// the lookup proof for the relu
    pub(crate) lookup: LogUpBatchProof<E>,
    /// The witness commitments from this function
    pub(crate) commit: PCS::Commitment,
}

impl<E, PCS> ActivationProof<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub(crate) fn write_commitment<T: Transcript<E>>(&self, transcript: &mut T) -> Result<()> {
        PCS::write_commitment(&self.commit, transcript).map_err(|e| anyhow!("{e:?}"))
    }
}

/// Type wrapping an activation layer employed in GeGlu (i.e., a GLU with Gelu as activation function)
pub struct GeGlu<N>(Activation<N>);

impl<N> Activation<N> {
    /// The port index for the "up" projection input when used in a GLU.
    pub const UP_INPUT_INDEX: usize = 1;

    /// The port index for the "gate" input when used in a GLU.
    pub const GATE_INPUT_INDEX: usize = 0;

    /// Returns a new rectified linear activation.
    pub fn new_relu() -> Self {
        Self::Plain(ActivationLayer::<N>::Relu(None, PhantomData))
    }

    /// Returns a new gaussian error activation.
    pub fn new_gelu() -> Self {
        Self::Plain(ActivationLayer::<N>::Gelu(None, PhantomData))
    }

    /// Instantiate a new Activation layer configured to be used in a GLU.
    pub fn new_for_glu(activation_type: ActivationLayer<N>) -> Self {
        Self::GLU(activation_type)
    }

    /// Instantiate a new Activation layer configured to be used as a plain activation function.
    pub fn new_plain(activation_type: ActivationLayer<N>) -> Self {
        Self::Plain(activation_type)
    }

    /// Returns a Gated GELU.
    pub fn new_geglu() -> GeGlu<N> {
        GeGlu::<N>(Self::GLU(ActivationLayer::<N>::Gelu(None, PhantomData)))
    }

    pub(crate) fn activation_type(&self) -> &ActivationLayer<N> {
        match self {
            Self::Plain(layer) | Self::GLU(layer) => layer,
        }
    }
}

impl<N> From<GeGlu<N>> for Layer<N>
where
    N: TensorTypeParam,
{
    fn from(geglu: GeGlu<N>) -> Self {
        Layer::Activation(geglu.0)
    }
}

impl<N> GeGlu<N> {
    /// Position expected in the set of layer inputs for the input value that is passed through Gelu
    pub const GELU_INPUT_INDEX: usize = 0;

    /// Position expected in the set of layers inputs for the input value that is multiplied to the output of Gelu
    pub const LINEAR_INPUT_INDEX: usize = 1;
}

impl ActivationLayer<f32> {
    fn evaluate(&self, inputs: &[&WrappedTensor<f32>]) -> Result<Vec<WrappedTensor<f32>>> {
        match self {
            ActivationLayer::Relu(..) => Ok(inputs
                .iter()
                .map(|tensor| WrappedTensor::relu((*tensor).clone()))
                .collect::<Vec<_>>()),
            ActivationLayer::Gelu(..) => Ok(inputs
                .iter()
                .map(|input| WrappedTensor::gelu((*input).clone()))
                .collect::<Vec<_>>()),
        }
    }

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
        num_outputs: usize,
    ) -> anyhow::Result<QuantizeOutput<ActivationLayer<Element>>> {
        let output_scalings = S::scaling_factors_for_node(data, node_id, num_outputs);
        let table_input_scaling = S::scaling_factor_for_intermediate_data(
            data,
            node_id,
            self.tracked_input_data_id().into(),
        );
        let table_output_scaling = S::scaling_factor_for_intermediate_data(
            data,
            node_id,
            self.tracked_output_data_id().into(),
        );
        let activation_lookup_data = ActivationLookupData::new_from_scalings(
            input_scaling[0],
            output_scalings[0],
            table_input_scaling,
            table_output_scaling,
            &self,
        )?;

        match self {
            ActivationLayer::Relu(..) => Ok(QuantizeOutput::new(
                ActivationLayer::<Element>::Relu(Some(activation_lookup_data), PhantomData),
                output_scalings,
            )),
            ActivationLayer::Gelu(..) => Ok(QuantizeOutput::new(
                ActivationLayer::<Element>::Gelu(Some(activation_lookup_data), PhantomData),
                vec![table_output_scaling],
            )),
        }
    }
}

impl ActivationLayer<Element> {
    fn get_lookup_data(&self) -> &ActivationLookupData {
        match self {
            ActivationLayer::Relu(Some(data), ..) | ActivationLayer::Gelu(Some(data), ..) => data,
            _ => panic!("Activation layer lookup data not initialized"),
        }
    }

    fn set_glu_flag(&mut self, is_glu: bool) {
        match self {
            ActivationLayer::Relu(Some(data), ..) | ActivationLayer::Gelu(Some(data), ..) => {
                data.set_glu(is_glu);
            }
            _ => unreachable!("ActivaitonLayer Element lookup data not initialized"),
        }
    }

    pub(crate) fn lookup_tables(&self) -> Vec<Table> {
        let lookup_data = self.get_lookup_data();
        let value_table = lookup_data.table;
        match value_table.is_signed() {
            true => {
                let chunking_info = lookup_data.chunking_info(&value_table).unwrap();
                let number_zero_chunks = chunking_info.number_of_zeroing_chunks();
                match number_zero_chunks {
                    0 => vec![Table::new_shift_check(), value_table],
                    1 => vec![
                        Table::new_shift_check(),
                        value_table,
                        Table::new_signed_zero_check(),
                    ],
                    _ => vec![
                        Table::new_shift_check(),
                        value_table,
                        Table::new_zero_check(),
                        Table::new_signed_zero_check(),
                    ],
                }
            }
            false => vec![
                Table::new_shift_check(),
                value_table,
                Table::new_zero_check(),
            ],
        }
    }

    fn evaluate(&self, inputs: &[&WrappedTensor<Element>]) -> Result<Vec<WrappedTensor<Element>>> {
        let lookup_data = self.get_lookup_data();
        inputs
            .iter()
            .map(|tensor| lookup_data.evaluate((*tensor).clone()))
            .collect()
    }

    fn lookup_witness<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        activation_input: &Tensor<Element>,
    ) -> Result<LookupWitnessGen<E, PCS>>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    {
        let lookup_data = self.get_lookup_data();
        let unpadded_input = if activation_input.shape() == activation_input.unpadded_shape() {
            activation_input.clone()
        } else {
            activation_input.reduce_to_shape(activation_input.unpadded_shape())?
        };
        let lookup_witness = lookup_data.get_lookup_witness(unpadded_input)?;

        let element_counts = lookup_witness.get_counts(&lookup_data.table);

        let input_evals = lookup_witness.input_mle_evals::<E>(lookup_data.table.num_columns());
        let input_width = input_evals.len();
        let output_evals = lookup_witness.output_mle_evals::<E>();
        let output_width = output_evals.len();

        // Add the witness polynomials that we need to commit to
        let transposed_input = transpose(input_evals);
        let input_rmm = RowMajorMatrix::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(transposed_input.concat(), input_width),
            witness::InstancePaddingStrategy::Default,
        );

        let transposed_output = transpose(output_evals);
        let output_rmm = RowMajorMatrix::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(transposed_output.concat(), output_width),
            witness::InstancePaddingStrategy::Default,
        );

        let commit = ctx
            .commitment_ctx
            .batch_commit(vec![input_rmm, output_rmm])?;

        let mut gen_w = LookupWitnessGen::<E, PCS>::default();
        let tables = vec![
            Table::new_shift_check(),
            lookup_data.table,
            Table::new_zero_check(),
            Table::new_signed_zero_check(),
        ];

        gen_w.insert_layer_witness_data(id, commit, tables, element_counts);

        Ok(gen_w)
    }
}

impl<N> OpInfo for Activation<N> {
    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        match self {
            Self::Plain(_) => Ok(num_inputs),
            Self::GLU(_) => Ok(1),
        }
    }

    fn describe(&self) -> String {
        match self.activation_type() {
            ActivationLayer::Relu(..) => "ReLU".to_string(),
            ActivationLayer::Gelu(..) => "GeLU".to_string(),
        }
    }

    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        _padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        match self {
            Self::Plain(_) => Ok(input_shapes.to_vec()), // same as input shapes,
            Self::GLU(_) => Ok(vec![input_shapes[0].clone()]), /* in GLU, there is only one output, which has the same shape as the first input */
        }
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl QuantizeOp for Activation<f32> {
    type QuantizedOp = Activation<Element>;

    fn quantize_op<S: crate::ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[crate::ScalingFactor],
        _unpadded_input_shapes: &[Shape],
        output_scalings: &[ScalingFactor],
        _unpadded_output_shapes: &[Shape],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        let num_outputs = self.num_outputs(input_scaling.len())?;
        match self {
            Self::Plain(layer) => layer
                .quantize_op::<S>(data, node_id, input_scaling, num_outputs)
                .map(|q_op| {
                    QuantizeOutput::new(Activation::Plain(q_op.quantized_op), q_op.output_scalings)
                }),
            Self::GLU(layer) => {
                ensure!(
                    input_scaling.len() == 2,
                    "Expected 2 input scaling factors for activation layer used in GLU, found {}",
                    input_scaling.len(),
                );

                let QuantizeOutput {
                    mut quantized_op,
                    output_scalings: activation_out_scalings,
                    ..
                } = layer.quantize_op::<S>(data, node_id, input_scaling, num_outputs)?;
                ensure!(
                    activation_out_scalings.len() == 1,
                    "Expected 1 output scaling factor for activation layer used in GLU, found {}",
                    activation_out_scalings.len(),
                );
                // Set the GLU flag in the lookup data
                quantized_op.set_glu_flag(true);

                let multiplier =
                    activation_out_scalings[0].m(&input_scaling[1], &output_scalings[0]);
                let intermediate_bit_size =
                    activation_out_scalings[0].bit_size() + input_scaling[1].bit_size() + 1;

                let requant =
                    Requant::from_multiplier(multiplier, intermediate_bit_size, output_scalings[0]);
                Ok(
                    QuantizeOutput::new(Activation::GLU(quantized_op), output_scalings.to_vec())
                        .with_requant(requant)?,
                )
            }
        }
    }
}

impl Evaluate<f32> for Activation<f32> {
    fn evaluate(&self, inputs: &[&WrappedTensor<f32>]) -> Result<LayerOut<f32>> {
        match self {
            Activation::Plain(layer) => {
                let mut activation_outputs = layer.evaluate(inputs)?;
                let activation_out = activation_outputs.pop().unwrap();
                Ok(
                    LayerOut::from_vec(vec![activation_out.clone()]).with_data_to_be_tracked(
                        HashMap::from([
                            (layer.tracked_output_data_id().into(), activation_out),
                            (layer.tracked_input_data_id().into(), inputs[0].clone()),
                        ]),
                    ),
                )
            }
            Activation::GLU(layer) => {
                ensure!(
                    inputs.len() == 2,
                    "Expected 2 inputs for activation layer used in GLU, found {} inputs instead",
                    inputs.len(),
                );
                let mut activation_outputs = layer.evaluate(&[inputs[0]])?;
                // double-check that there is only one output
                assert_eq!(activation_outputs.len(), 1);
                let activation_out = activation_outputs.pop().unwrap();
                Ok(LayerOut::from_vec(
                    // multiply `activation_out` with `inputs[1]`
                    vec![activation_out.clone().mul(inputs[1].clone())?],
                )
                .with_data_to_be_tracked(HashMap::from([
                    (layer.tracked_output_data_id().into(), activation_out),
                    (layer.tracked_input_data_id().into(), inputs[0].clone()),
                ])))
            }
        }
    }
}

impl Evaluate<Element> for Activation<Element> {
    fn evaluate(&self, inputs: &[&WrappedTensor<Element>]) -> Result<LayerOut<Element>> {
        match self {
            Activation::Plain(activation_layer) => {
                activation_layer.evaluate(inputs).map(LayerOut::from_vec)
            }
            Activation::GLU(activation_layer) => {
                ensure!(
                    inputs.len() == 2,
                    "Expected 2 inputs for activation layer used in GLU, found {} inputs instead",
                    inputs.len(),
                );
                let mut activation_outputs = activation_layer.evaluate(&[inputs[0]])?;
                // double-check that there is only one output
                assert_eq!(activation_outputs.len(), 1);
                let activation_output = activation_outputs.pop().unwrap();
                let layer_out = LayerOut::from_vec(vec![activation_output.mul(inputs[1].clone())?]);
                Ok(layer_out)
            }
        }
    }
}

impl ProveInfo for Activation<Element> {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeId,
        mut aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)> {
        // Set the model polys to be empty
        aux.model_polys = None;
        aux.max_poly_len = aux
            .last_output_shape
            .iter()
            .fold(aux.max_poly_len, |acc, shapes| {
                acc.max(shapes.next_power_of_two().product())
            });
        let act = self.clone();

        Ok((
            LayerCtx::Activation(ActivationCtx {
                op: act,
                node_id: id,
            }),
            aux,
        ))
    }
}

impl<N> PadOp for Activation<N> {}

impl<E, PCS> ProvableOp<E, PCS> for Activation<Element>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = ActivationCtx;

    fn prove<'a, 'b, 'c, 'd, T: Transcript<E>>(
        &'a self,
        id: NodeId,
        _ctx: &'b Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &Step<Element>,
        prover: &mut Prover<'c, 'd, E, T, PCS>,
    ) -> Result<Vec<Claim<E>>> {
        let inputs = &step_data.node_inputs;
        ensure!(
            !inputs.is_empty(),
            "Expected at least 1 input in inferece data for activation layer",
        );
        self.prove_step(prover, last_claims[0], step_data, id)
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        step_data: &Step<Element>,
    ) -> Result<LookupWitnessGen<E, PCS>> {
        let outputs = step_data.output_tensors()?;
        ensure!(
            outputs.len() == 1,
            "Found more than 1 output tensor in inference step of activation layer"
        );

        match self {
            Activation::Plain(activation_layer) => {
                ensure!(
                    step_data.node_inputs.len() == 1,
                    "Found more than 1 input tensor in inference step of activation layer"
                );
                activation_layer.lookup_witness(id, ctx, step_data.input_tensor_at(0)?.deref())
            }
            Activation::GLU(activation_layer) => {
                ensure!(
                    step_data.node_inputs.len() == 2,
                    "Found more than 2 input tensor in inference step of activation layer"
                );
                activation_layer.lookup_witness(id, ctx, step_data.input_tensor_at(0)?.deref())
            }
        }
    }
}

impl OpInfo for ActivationCtx {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        self.op.output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        self.op.num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        self.op.describe()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for ActivationCtx
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = ActivationProof<E, PCS>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        self.verify_activation(verifier, last_claims[0], proof, shape_step)
    }
    fn write_proof_to_transcript<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        proof.write_commitment(transcript)
    }
}

impl Activation<Element> {
    #[timed::timed_instrument(name = "Prover::prove_activation_step")]
    pub(crate) fn prove_step<'a, 'b, E, T: Transcript<E>, PCS>(
        &self,
        prover: &mut Prover<'a, 'b, E, T, PCS>,
        last_claim: &Claim<E>,
        step: &Step<Element>,
        node_id: NodeId,
    ) -> anyhow::Result<Vec<Claim<E>>>
    where
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        let lookup_op = self.activation_type().get_lookup_data();

        let LookupProverResult {
            generic_proof,
            input_claims,
            ..
        } = prove_lookup_op(
            lookup_op,
            last_claim,
            step,
            &lookup_op.table,
            None,
            prover,
            node_id,
        )?;

        let GenericLookupProof {
            logup_proof,
            sumcheck_proof,
            evaluations,
            commitment,
            ..
        } = generic_proof;

        let proof = ActivationProof {
            io_accumulation: sumcheck_proof,
            evaluations,
            lookup: logup_proof,
            commit: commitment,
        };

        // Add the proof to the prover
        prover.push_proof(node_id, LayerProof::Activation(proof));
        // Return the input claim for the next layer
        Ok(input_claims)
    }
}

impl ActivationCtx {
    pub(crate) fn verify_activation<
        E: ExtensionField,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        last_claim: &Claim<E>,
        proof: &ActivationProof<E, PCS>,
        shape_step: &ShapeStep,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        let ActivationProof {
            io_accumulation,
            evaluations,
            lookup,
            commit,
        } = proof;

        let generic_lookup_proof = GenericLookupProof::<E, PCS> {
            logup_proof: lookup.clone(),
            sumcheck_proof: io_accumulation.clone(),
            evaluations: evaluations.clone(),
            commitment: commit.clone(),
            weight_evaluation: None,
            shift_evaluations: None,
        };
        let lookup_op = self.op.activation_type().get_lookup_data();
        let LookupVerifyResult { input_claims, .. } = verify_lookup_op(
            lookup_op,
            last_claim,
            shape_step,
            &lookup_op.table,
            &generic_lookup_proof,
            verifier,
            self.node_id,
        )?;

        Ok(input_claims)
    }
}

#[cfg(test)]
mod test {
    use ark_std::rand::Rng;
    use burn::tensor::activation::gelu;
    use proptest::prelude::*;

    use crate::{
        layers::{EinSum, Layer},
        lookup::table::gelu_float,
        model::{Model, test::prove_model},
        rng_from_env_or_random,
        tensor::{IntoBTensor, KeyedTensor},
    };

    use super::*;

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

            let input_rank = rng.gen_range(1usize..=4);

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

    #[test]
    fn test_activation_proving() -> anyhow::Result<()> {
        test_activation_proving_helper(Activation::<f32>::new_relu)?;
        test_activation_proving_helper(Activation::<f32>::new_gelu)
    }

    fn test_activation_proving_helper<F>(f: F) -> anyhow::Result<()>
    where
        F: Fn() -> Activation<f32>,
    {
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

            let mut model = Model::new_from_input_shapes(
                vec![random_input.shape().clone()],
                PaddingMode::NoPadding,
            );

            let dense_id = model
                .add_consecutive_layer(Layer::EinSum(dense), None)
                .unwrap();

            let _ = model
                .add_consecutive_layer(Layer::Activation(f()), Some(dense_id))
                .unwrap();

            model.automatic_output_labelling().unwrap();
            model.describe();
            prove_model(model, &mut Default::default()).unwrap();
        }

        Ok(())
    }

    #[test]
    fn test_glu_activation_proving() -> anyhow::Result<()> {
        for _ in 0..25 {
            let input_shape = vec![7, 94].into();
            let mut model = Model::new_from_input_shapes(
                vec![input_shape; 2], // 2 inputs in case of GLU variant
                PaddingMode::NoPadding,
            );
            model.add_consecutive_layer(Activation::new_geglu().into(), None)?;
            model.automatic_output_labelling()?;
            prove_model(model, &mut Default::default()).unwrap();
        }
        Ok(())
    }

    proptest! {
        #[test]
        fn gelu_kernel_test(size in 1usize..1024) {
            let shape = Shape::new(vec![size]);
            let tensor = Tensor::<f32>::random(&shape);

            let btensor = tensor.to_btensor::<1>();
            let data = gelu(btensor).to_data().into_vec().expect("Failed to compute GELU");
            let resultb = Tensor::<f32>::new(shape.clone(), data).unwrap();

            let data = tensor.data();
            let data = data.iter().map(gelu_float).collect::<Vec<_>>();
            let result = Tensor::new(shape, data).unwrap();

            resultb.data().iter().zip(result.data().iter()).try_for_each(|(left, right)| {
                prop_assert!(
                    (left - right).abs() < 1e-3,
                    "Actual: {left}, Expected: {right}",
                );
                Ok(())
            })?;
        }
    }
}
