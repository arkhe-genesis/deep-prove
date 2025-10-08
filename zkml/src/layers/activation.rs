use crate::{
    Claim, Element, Prover, ProverContext, ScalingFactor, ScalingStrategy, Shape,
    commit::{compute_betas_eval, identity_eval},
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        Layer, LayerCtx, LayerProof,
        provable::{ProvingData, QuantizeOp, QuantizeOutput},
        requant::Requant,
    },
    lookup::{
        context::{COLUMN_SEPARATOR, LayerLookupContext, LookupWitnessGen, TableType},
        logup_gkr::{
            prover::batch_multiple_sizes_prove, structs::LogUpBatchProof,
            verifier::verify_logup_proof_multiple_sizes,
        },
    },
    model::StepData,
    number::Number,
    padding::PaddingMode,
    quantization::{self, Fieldizer},
    tensor::DryTensor,
};
use burn::tensor::activation::gelu;
use either::Either;
use ff_ext::ExtensionField;
use witness::RowMajorMatrix;

use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    Expression,
    mle::IntoMLE,
    util::{ceil_log2, transpose},
    utils::eval_by_expr_with_instance,
    virtual_poly::VPAuxInfo,
    virtual_polys::VirtualPolynomialsBuilder,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{collections::HashMap, marker::PhantomData};
use sumcheck::{
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
};
use tenstore::GenStore;
use transcript::Transcript;

use super::provable::{Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, VerifiableCtx};
use crate::{model::NodeID, quantization::BIT_LEN, tensor::Tensor};

/// The short name used to identify an activation layer.
pub const ACTIVATION_LAYER: &str = "ACTI";

use anyhow::{Result, anyhow, bail, ensure};
const GELU_SCALE_EXP: usize = 12;
const GELU_SCALE_FACTOR: usize = 1 << GELU_SCALE_EXP;

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
    Relu(Relu),
    Gelu(GELU<N>),
}
#[derive(Clone, Debug, Serialize, Deserialize)]

pub struct ActivationData {
    activation_output: Tensor<Element>,
}

/// Currently holds the poly info for the output polynomial of the RELU
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct ActivationCtx<E: ExtensionField + Serialize + DeserializeOwned> {
    pub op: Activation<Element>,
    pub lookup_context: LayerLookupContext,
    pub sumcheck_expression: Vec<Expression<E>>,
    pub node_id: NodeID,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct ActivationProof<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
{
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
    pub fn new_relu() -> Self {
        Self::Plain(ActivationLayer::Relu(Relu))
    }

    pub fn new_gelu() -> Self {
        Self::Plain(ActivationLayer::Gelu(GELU::new()))
    }

    /// Instantiate a new Activation layer configured to be used in a GLU.
    pub fn new_for_glu(activation_type: ActivationLayer<N>) -> Self {
        Self::GLU(activation_type)
    }

    pub fn new_geglu() -> GeGlu<N> {
        GeGlu(Self::GLU(ActivationLayer::Gelu(GELU::new())))
    }

    pub(crate) fn activation_type(&self) -> &ActivationLayer<N> {
        match self {
            Self::Plain(layer) | Self::GLU(layer) => layer,
        }
    }
}

impl<N> From<GeGlu<N>> for Layer<N> {
    fn from(geglu: GeGlu<N>) -> Self {
        Layer::Activation(geglu.0)
    }
}

impl<N> GeGlu<N> {
    // Position expected in the set of layer inputs for the input value that is passed through Gelu
    pub const GELU_INPUT_INDEX: usize = 0;
    // Position expected in the set of layers inputs for the input value that is multiplied to the output of Gelu
    pub const LINEAR_INPUT_INDEX: usize = 1;
}

impl ActivationLayer<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        _unpadded_input_shapes: &[Shape],
    ) -> Result<Vec<Tensor<f32>>> {
        match self {
            ActivationLayer::Relu(relu) => Ok(inputs.iter().map(|input| relu.op(input)).collect()),
            ActivationLayer::Gelu(gelu) => inputs
                .iter()
                .map(|input| {
                    let mut outputs = gelu
                        .evaluate::<E>(&[input], _unpadded_input_shapes)?
                        .outputs;
                    ensure!(outputs.len() == 1);
                    Ok(outputs.pop().unwrap())
                })
                .collect::<anyhow::Result<Vec<_>>>(),
        }
    }

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeID,
        input_scaling: &[ScalingFactor],
        num_outputs: usize,
    ) -> anyhow::Result<QuantizeOutput<ActivationLayer<Element>>> {
        let output_scalings = S::scaling_factors_for_node(data, node_id, num_outputs);
        let quantized_op = match self {
            ActivationLayer::Relu(_) => ActivationLayer::Relu(Relu),
            ActivationLayer::Gelu(g) => ActivationLayer::Gelu(g.quantize(input_scaling[0])?),
        };
        Ok(QuantizeOutput::new(quantized_op, output_scalings))
    }
}

impl ActivationLayer<Element> {
    fn evaluate(
        &self,
        inputs: &[&Tensor<Element>],
        _unpadded_input_shapes: &[Shape],
    ) -> Result<Vec<Tensor<Element>>> {
        match self {
            ActivationLayer::Relu(relu) => Ok(inputs.iter().map(|input| relu.op(input)).collect()),
            ActivationLayer::Gelu(g) => inputs
                .iter()
                .map(|input| input.try_map(|e| g.apply(e)))
                .collect::<anyhow::Result<Vec<_>>>(),
        }
    }

    fn lookup_witness<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        id: NodeID,
        ctx: &ProverContext<E, PCS>,
        activation_input: &Tensor<Element>,
        activation_output: &Tensor<Element>,
    ) -> Result<LookupWitnessGen<E, PCS>>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    {
        let input_data = activation_input.get_data();
        let output_data = activation_output.get_data();
        debug_assert_eq!(
            input_data.len(),
            output_data.len(),
            "Input and outputs must have the same length",
        );
        let size = input_data.len();

        let mut element_count = HashMap::<Element, u64>::new();
        let mut col_one = Vec::<E::BaseField>::with_capacity(size);
        let mut col_two = Vec::<E::BaseField>::with_capacity(size);
        for (a, b) in input_data.iter().zip(output_data.iter()) {
            let (a, a_field): (Element, E) = match self {
                ActivationLayer::Relu(_) => (*a, a.to_field()),
                ActivationLayer::Gelu(g) => {
                    let scaled = a * g.quant_data.as_ref().unwrap().multiplier;
                    assert!(
                        scaled >= g.quant_data.as_ref().unwrap().table_data.min
                            && scaled <= g.quant_data.as_ref().unwrap().table_data.max
                    );
                    (scaled, scaled.to_field())
                }
            };
            // Calculate the lookup element
            let el = a + COLUMN_SEPARATOR * b;
            *element_count.entry(el).or_default() += 1;

            // Calculate the column_evals
            let b_field: E = b.to_field();
            col_one.push(a_field.as_bases()[0]);
            col_two.push(b_field.as_bases()[0]);
        }
        let transposed = transpose(vec![col_one, col_two]);
        // Add the witness polynomials that we need to commit to
        let rmm = RowMajorMatrix::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(transposed.concat(), 2),
            witness::InstancePaddingStrategy::Default,
        );

        let commit = ctx.commitment_ctx.batch_commit(vec![rmm])?;

        let mut gen = LookupWitnessGen::<E, PCS>::default();
        gen.insert_logup_witness(id, commit);
        gen.insert_element_count(self.table_type(), element_count);

        Ok(gen)
    }

    fn table_type(&self) -> TableType {
        match self {
            ActivationLayer::Relu(_) => TableType::Relu,
            ActivationLayer::Gelu(g) => {
                TableType::GELU(g.quant_data.map(|q| q.table_data).unwrap())
            }
        }
    }
}

impl<N> OpInfo for Activation<N> {
    fn num_outputs(&self, num_inputs: usize) -> usize {
        match self {
            Self::Plain(_) => num_inputs,
            Self::GLU(_) => 1,
        }
    }

    fn describe(&self) -> String {
        match self.activation_type() {
            ActivationLayer::Relu(_relu) => format!("RELU: {}", 1 << Relu::num_vars()),
            ActivationLayer::Gelu(_gelu) => "GELU".to_string(),
        }
    }

    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        match self {
            Self::Plain(_) => input_shapes.to_vec(), // same as input shapes,
            Self::GLU(_) => vec![input_shapes[0].clone()], /* in GLU, there is only one output, which has the same shape as the first input */
        }
    }

    fn is_provable(&self) -> bool {
        true
    }
}

const ACTIVATION_OUT_ID: &str = "ActivationOut";

impl Evaluate<f32> for Activation<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<f32, E>> {
        match self {
            Activation::Plain(layer) => layer
                .evaluate::<E>(inputs, unpadded_input_shapes)
                .map(|outputs| LayerOut::from_vec(outputs)),
            Activation::GLU(layer) => {
                ensure!(
                    inputs.len() == 2,
                    "Expected 2 inputs for activation layer used in GLU, found {} inputs instead",
                    inputs.len(),
                );
                let mut activation_outputs =
                    layer.evaluate::<E>(&[inputs[0]], unpadded_input_shapes)?;
                // double-check that there is only one output
                assert_eq!(activation_outputs.len(), 1);
                let activation_out = activation_outputs.pop().unwrap();
                Ok(LayerOut::from_vec(
                    // multiply `activation_out` with `inputs[1]`
                    vec![activation_out.mul(inputs[1])],
                )
                .with_data_to_be_tracked(HashMap::from([(
                    ACTIVATION_OUT_ID.to_string().into(),
                    activation_out,
                )])))
            }
        }
    }
}

impl QuantizeOp for Activation<f32> {
    type QuantizedOp = Activation<Element>;

    fn quantize_op<S: crate::ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeID,
        input_scaling: &[crate::ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        let num_outputs = self.num_outputs(input_scaling.len());
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
                    quantized_op,
                    output_scalings,
                    ..
                } = layer.quantize_op::<S>(data, node_id, input_scaling, num_outputs)?;
                ensure!(
                    output_scalings.len() == 1,
                    "Expected 1 output scaling factor for activation layer used in GLU, found {}",
                    output_scalings.len(),
                );
                let activation_out_scaling = S::scaling_factor_for_intermediate_data(
                    data,
                    node_id,
                    ACTIVATION_OUT_ID.to_string().into(),
                );
                let multiplier = activation_out_scaling.m(&input_scaling[1], &output_scalings[0]);
                let intermediate_bit_size = match quantized_op {
                    ActivationLayer::Relu(_) => 2 * *quantization::BIT_LEN, /* we are multiplying 2 items with `quantization::BIT_LEN` bits, */
                    ActivationLayer::Gelu(ref g) => {
                        let quant_data_gelu = g.quant_data.unwrap();
                        quant_data_gelu
                            .table_data
                            .table()
                            .map(|(_, output)| {
                                if output.abs() != 0 {
                                    ceil_log2(output.unsigned_abs() as usize)
                                } else {
                                    0
                                }
                            })
                            .max()
                            .unwrap()
                            + *quantization::BIT_LEN
                    }
                };
                let requant = Requant::from_multiplier(multiplier, intermediate_bit_size);
                Ok(
                    QuantizeOutput::new(Activation::GLU(quantized_op), output_scalings)
                        .with_requant(requant),
                )
            }
        }
    }
}

impl Evaluate<Element> for Activation<Element> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<Element, E>> {
        match self {
            Activation::Plain(activation_layer) => activation_layer
                .evaluate(inputs, unpadded_input_shapes)
                .map(|outputs| LayerOut::from_vec(outputs)),
            Activation::GLU(activation_layer) => {
                ensure!(
                    inputs.len() == 2,
                    "Expected 2 inputs for activation layer used in GLU, found {} inputs instead",
                    inputs.len(),
                );
                let mut activation_outputs =
                    activation_layer.evaluate(&[inputs[0]], unpadded_input_shapes)?;
                // double-check that there is only one output
                assert_eq!(activation_outputs.len(), 1);
                let activation_output = activation_outputs.pop().unwrap();
                Ok(
                    LayerOut::from_vec(vec![activation_output.mul(inputs[1])]).with_proving_data(
                        ProvingData::Activation(ActivationData { activation_output }),
                    ),
                )
            }
        }
    }
}

impl ProveInfo for Activation<Element> {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeID,
        mut aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)> {
        let lookup_context = match self.activation_type() {
            ActivationLayer::Relu(_) => {
                aux.tables.insert(TableType::Relu);
                LayerLookupContext::new(vec![TableType::Relu], vec![1])
            }
            // TODO: if we want to save on memory, we can use a pointer to the vector instead
            ActivationLayer::Gelu(gelu) => {
                aux.tables.insert(TableType::GELU(
                    gelu.quant_data.map(|q| q.table_data).unwrap(),
                ));
                LayerLookupContext::new(
                    vec![TableType::GELU(
                        gelu.quant_data.map(|q| q.table_data).unwrap(),
                    )],
                    vec![1],
                )
            }
        };

        // Set the model polys to be empty
        aux.model_polys = None;
        aux.max_poly_len = aux
            .last_output_shape
            .iter()
            .fold(aux.max_poly_len, |acc, shapes| {
                acc.max(shapes.next_power_of_two().product())
            });
        let act = self.clone();
        // Build the sumcheck Expression, we presume the polynomials will be loaded in in the order: input_column, output_column, eq_polys
        let lookup_claims_expr = Expression::WitIn(2)
            * (Expression::WitIn(0)
                + Expression::WitIn(1) * Expression::Challenge(0, 1, E::ONE, E::ZERO));
        let sumcheck_expression = match self {
            Activation::GLU(_) =>
            // if `self` is used in GLU, there is an additional input, which needs to be entry-wise multiplied with the output
            // of the activation function (i.e., the output column of the lookup table); this additional input polynomial
            // is assumed to be loaded as the last polynomial in the above list
            {
                lookup_claims_expr
                    + Expression::Challenge(0, 2, E::ONE, E::ZERO)
                        * Expression::WitIn(3)
                        * Expression::WitIn(1)
                        * Expression::WitIn(4)
            }
            Activation::Plain(_) => {
                lookup_claims_expr
                    + Expression::Challenge(0, 2, E::ONE, E::ZERO)
                        * Expression::WitIn(3)
                        * Expression::WitIn(1)
            }
        };

        Ok((
            LayerCtx::Activation(ActivationCtx {
                op: act,
                lookup_context,
                node_id: id,
                sumcheck_expression: vec![sumcheck_expression],
            }),
            aux,
        ))
    }
}

impl<N> PadOp for Activation<N> {}

impl<E, PCS> ProvableOp<E, PCS> for Activation<Element>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = ActivationCtx<E>;

    fn prove<'a, 'b, 'c, 'd, T: Transcript<E>>(
        &'a self,
        id: NodeID,
        ctx: &'b Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<'c, 'd, E, T, PCS>,
        store: &mut GenStore,
    ) -> Result<Vec<Claim<E>>> {
        let inputs = &step_data.node_inputs;
        ensure!(
            !inputs.is_empty(),
            "Expected at least 1 input in inferece data for activation layer",
        );
        self.prove_step(prover, last_claims[0], ctx, inputs, id, store)
    }

    fn gen_lookup_witness(
        &self,
        id: NodeID,
        ctx: &ProverContext<E, PCS>,
        step_data: &StepData<Element, E>,
        store: &mut GenStore,
    ) -> Result<LookupWitnessGen<E, PCS>> {
        let outputs = step_data.output_tensors(store)?;
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
                activation_layer.lookup_witness(
                    id,
                    ctx,
                    &step_data.input_tensor_at(0, store)?,
                    &outputs[0],
                )
            }
            Activation::GLU(activation_layer) => {
                ensure!(
                    step_data.node_inputs.len() == 2,
                    "Found more than 2 input tensor in inference step of activation layer"
                );
                let data = step_data.node_outputs.try_activation_data().ok_or(anyhow!(
                    "Proving data not found in inference trace for activation layer"
                ))?;
                activation_layer.lookup_witness(
                    id,
                    ctx,
                    &step_data.input_tensor_at(0, store)?,
                    &data.activation_output,
                )
            }
        }
    }
}

impl<E: ExtensionField> OpInfo for ActivationCtx<E> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        self.op.output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        self.op.num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        self.op.describe()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for ActivationCtx<E>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = ActivationProof<E, PCS>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        _shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        self.verify_activation(verifier, last_claims[0], proof)
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
        step: &ActivationCtx<E>,
        inputs: &[DryTensor<E>],
        node_id: NodeID,
        store: &mut GenStore,
    ) -> anyhow::Result<Vec<Claim<E>>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        // Should only be one prover_info for this step
        let layer_commitment = prover.lookup_witness(node_id)?;
        let logup_inputs = step
            .lookup_context
            .create_logup_inputs::<PCS, E>(layer_commitment, &prover.challenge_storage)?;
        let layer_polys = PCS::get_arc_mle_witness_from_commitment(layer_commitment);
        let commit = PCS::get_pure_commitment(layer_commitment);
        // Run the lookup protocol and return the lookup proof
        let logup_proof = batch_multiple_sizes_prove(&logup_inputs, prover.transcript)?;

        let input_claim = logup_proof.output_claims()[0].clone();
        let logup_point = &input_claim.point;

        let logup_eq_poly = compute_betas_eval(logup_point).into_mle();
        let last_claim_eq = compute_betas_eval(&last_claim.point).into_mle();

        let mut either_polys = layer_polys
            .iter()
            .map(|p| Either::Left(p.as_ref()))
            .chain(
                [&logup_eq_poly, &last_claim_eq]
                    .iter()
                    .map(|&p| Either::Left(p)),
            )
            .collect::<Vec<Either<_, _>>>();

        // In case the layer is used in GLU, we need to add the MLE of second input tensor
        // to the set of polynomials involved in the sum-check. We first build the MLE
        // fro the second input tensor, if present
        let input_mle = inputs
            .get(1)
            .map(|input| anyhow::Ok(input.hydrate(store.clone())?.into_mle()))
            .unwrap_or(Ok(Default::default()))?;

        if let Activation::GLU(_) = self {
            either_polys.push(Either::Left(&input_mle));
        }

        let num_threads = optimal_sumcheck_threads(logup_point.len());
        let expr_builder = VirtualPolynomialsBuilder::<E>::new_with_mles(
            num_threads,
            logup_point.len(),
            either_polys,
        );
        let challenge = prover
            .transcript
            .sample_and_append_challenge(b"batching")
            .elements;
        let virtual_poly = expr_builder.to_virtual_polys(&step.sumcheck_expression, &[challenge]);
        let (proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);
        let mut all_evals = state.get_mle_flatten_final_evaluations();
        let point = state.collect_raw_challenges();

        // Add commitment claims to prover
        prover.add_witness_claim(
            node_id,
            vec![(point.clone(), vec![all_evals[0], all_evals[1]])],
        );

        let input_claim = match self.activation_type() {
            ActivationLayer::Gelu(g) => {
                let m: E = g.quant_data.as_ref().unwrap().multiplier.to_field();
                let mi = m.inverse();
                let eval = all_evals[0] * mi;
                Claim::new(point.clone(), eval)
            }
            _ => Claim::new(point.clone(), all_evals[0]),
        };

        // collect evaluations to be placed in the proof
        let evaluations = all_evals[..2].to_vec();
        let mut proof = ActivationProof {
            io_accumulation: proof,
            evaluations,
            lookup: logup_proof,
            commit,
        };
        Ok(match self {
            Activation::GLU(_) => {
                // we need to add the evaluation for the second input MLE evaluation, which is the last
                // polynomial in the sumcheck polynomials
                let second_input_eval = all_evals.pop().unwrap();
                proof.evaluations.push(second_input_eval);
                prover.push_proof(node_id, LayerProof::Activation(proof));
                // we need to return also the claim for the second input MLE
                let second_input_claim = Claim::new(point, second_input_eval);
                vec![input_claim, second_input_claim]
            }
            Activation::Plain(_) => {
                prover.push_proof(node_id, LayerProof::Activation(proof));
                vec![input_claim]
            }
        })
    }
}

impl<E: ExtensionField> ActivationCtx<E> {
    pub(crate) fn verify_activation<T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        last_claim: &Claim<E>,
        proof: &ActivationProof<E, PCS>,
    ) -> anyhow::Result<Vec<Claim<E>>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: Serialize + DeserializeOwned,
    {
        let ActivationProof {
            io_accumulation,
            evaluations,
            lookup,
            commit,
        } = proof;

        // 1. Verify the lookup proof
        let batch_claim = verify_logup_proof_multiple_sizes(lookup, verifier.transcript)?;
        self.lookup_context
            .verify_logup_batch_claim(&batch_claim, &verifier.challenge_storage)?;

        // 2. Verify the accumulation proof from last_claim + lookup claim into the new claim
        let challenge = verifier
            .transcript
            .sample_and_append_challenge(b"batching")
            .elements;
        let poly_evals = batch_claim.poly_evals();
        let claimed_sum = poly_evals[0] + challenge * (poly_evals[1] + challenge * last_claim.eval);
        let aux_info = VPAuxInfo {
            max_degree: match &self.op {
                Activation::GLU(_) => 3, /* in this case, max degree is 3 because we add the term related to the hadamard product */
                Activation::Plain(_) => 2,
            },
            max_num_variables: last_claim.point.len(),
            ..Default::default()
        };

        let subclaim = IOPVerifierState::<E>::verify(
            claimed_sum,
            io_accumulation,
            &aux_info,
            verifier.transcript,
        );
        let point = subclaim
            .point
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();
        let lookup_eq = identity_eval(batch_claim.point(), &point);
        let last_claim_eq = identity_eval(&last_claim.point, &point);
        let lookup_evals = evaluations[..2].to_vec();
        let mut witnesses = lookup_evals.clone();
        witnesses.push(lookup_eq);
        witnesses.push(last_claim_eq);
        // add also the evaluation for the second input MLE, if present
        if let Some(second_input_eval) = evaluations.get(2) {
            witnesses.push(*second_input_eval);
        }

        let calc_claim = self
            .sumcheck_expression
            .iter()
            .try_fold(E::ZERO, |acc, expr| {
                eval_by_expr_with_instance(&[], &witnesses, &[], &[], &[challenge], expr)
                    .right()
                    .map(|eval| acc + eval)
            })
            .ok_or(anyhow!(
                "Couldn't calculate final sumcheck evaluation in Activation"
            ))?;

        ensure!(
            calc_claim == subclaim.expected_evaluation,
            "Activation Verification failed: calculated claim: {:?} did not equal expected claim: {:?}",
            calc_claim,
            subclaim.expected_evaluation
        );

        // 3. Add the witness claim to be verified
        verifier.commit_verifier.add_witness_claim(
            self.node_id,
            commit.clone(),
            vec![(point.clone(), lookup_evals.clone())],
        );
        // 4. return the input claim for to be proven at subsequent step
        let input_claim = match self.op.activation_type() {
            ActivationLayer::Relu(_) => Claim::<E>::new(point.clone(), evaluations[0]),
            ActivationLayer::Gelu(g) => {
                let m: E = g.quant_data.as_ref().unwrap().multiplier.to_field();
                let mi = m.inverse();
                let eval = evaluations[0] * mi;
                Claim::new(point.clone(), eval)
            }
        };

        Ok(match &self.op {
            Activation::GLU(_) => {
                // we need to return also the claim to the second input
                let second_input_claim = Claim::new(point, evaluations[2]);
                vec![input_claim, second_input_claim]
            }
            Activation::Plain(_) => vec![input_claim],
        })
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub struct Relu;

impl Default for Relu {
    fn default() -> Self {
        Self::new()
    }
}

impl Relu {
    pub fn new() -> Relu {
        Self
    }
    pub fn num_vars() -> usize {
        *BIT_LEN
    }
    pub fn poly_len() -> usize {
        1 << Self::num_vars()
    }
    pub fn shape() -> Shape {
        Shape::new(vec![2, Self::poly_len()])
    }

    pub fn op<T: Number>(&self, input: &Tensor<T>) -> Tensor<T> {
        Tensor::new(
            input.shape().clone(),
            input
                .get_data()
                .par_iter()
                .map(|e| Self::apply(*e))
                .collect::<Vec<_>>(),
        )
    }

    #[inline(always)]
    pub fn apply<T: Number>(e: T) -> T {
        if e.is_negative() { T::default() } else { e }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GELU<N> {
    quant_data: Option<GELUQuantData>,
    _n: PhantomData<N>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Copy)]
pub struct GeluTableData {
    /// The minimum value of the input
    pub(crate) min: Element,
    /// The maximum value of the input
    pub(crate) max: Element,
}

impl GeluTableData {
    pub fn table_size(&self) -> usize {
        (self.max - self.min + 1).ilog2() as usize
    }
    /// Returns the input indexes of the table and the corresponding output values
    pub fn table(&self) -> impl Iterator<Item = (Element, Element)> + use<'_> {
        (self.min..self.max).map(|i| (i, self.table_output(i)))
    }
    /// NOTE: this requires the scaled input
    pub fn table_output(&self, input: Element) -> Element {
        let float_input = input as f32 / GELU_SCALE_FACTOR as f32;
        let float_output = gelu_float(&float_input);
        (float_output * *quantization::MAX as f32).round_ties_even() as Element
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Copy)]
pub struct GELUQuantData {
    /// The multiplier used to scale the input
    multiplier: Element,
    /// table data
    table_data: GeluTableData,
}

impl GELUQuantData {
    pub fn table_output(&self, input: Element) -> Element {
        self.table_data.table_output(input)
    }
}

impl<N> Default for GELU<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N> GELU<N> {
    pub fn new() -> Self {
        Self {
            quant_data: None,
            _n: PhantomData,
        }
    }
}

impl Evaluate<f32> for GELU<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        _unpadded_input_shapes: &[Shape],
    ) -> anyhow::Result<LayerOut<f32, E>> {
        let output_tensors: Vec<Tensor<f32>> = inputs
            .par_iter()
            .map(|tensor| {
                let shape = tensor.shape().clone();
                let mut tensor = (*tensor).clone();
                tensor.to_1d();
                let tensor = tensor.to_btensor::<1>();

                let result = gelu(tensor);

                let data = result.to_data().into_vec().expect("Failed to compute GELU");
                Tensor::new(shape, data)
            })
            .collect();
        Ok(LayerOut::from_vec(output_tensors))
    }
}

/// Compute the GeLU
///
/// This formula is based on [1]
///
/// [1]: https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
fn gelu_float(x: &f32) -> f32 {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();

    let x_cubed = x * x * x;
    let inner_term = c * (x + 0.044715 * x_cubed);
    0.5 * x * (1.0 + inner_term.tanh())
}

impl GELU<f32> {
    fn quantize(&self, input_scaling: ScalingFactor) -> anyhow::Result<GELU<Element>> {
        // so we want sf * SCALING = multiplier
        // then we construct the lookup table as  GELU(i / SCALING) * quantization::MAX for
        // all i in the range [-2^{7 + ceil_log2(multiplier)}, 2^{7 + ceil_log2(multiplier)}]
        // This is because the input is already requantized, and we're multipliying the input
        // by the multiplier during quantized inference such that the float input is scaled
        // to that number of bits. So with inputs of 2^7 max, multiplied by multiplier then
        // the output range is 2^{7 + ceil_log2(multiplier)}
        // During lookup, we basically scale down back to the original
        // float value, apply GELU and multiply by 128 which is right now the output maximum range.
        let multiplier =
            (GELU_SCALE_FACTOR as f32 * input_scaling.scale()).round_ties_even() as Element;
        assert!(
            multiplier > 0,
            "multiplier GELU is 0 -> change the scale factor"
        );
        let table_min = -2i32.pow(7 + ceil_log2(multiplier as usize) as u32);
        let table_max = 2i32.pow(7 + ceil_log2(multiplier as usize) as u32);
        let table_size = table_max - table_min;
        assert!((table_size as usize).is_power_of_two());
        assert!(
            table_size <= 1 << 25,
            "Table size for GELU is too bigggg: {:?}",
            table_size.ilog2()
        );
        let qd = GELUQuantData {
            multiplier,
            table_data: GeluTableData {
                min: table_min as Element,
                max: table_max as Element,
            },
        };
        Ok(GELU {
            quant_data: Some(qd),
            _n: PhantomData,
        })
    }
}

impl GELU<Element> {
    fn apply(&self, input: &Element) -> anyhow::Result<Element> {
        let Some(ref quant_data) = self.quant_data else {
            bail!("GELU not quantized");
        };
        let scaled = input * quant_data.multiplier;
        let within_range =
            quant_data.table_data.min <= scaled && scaled <= quant_data.table_data.max;
        ensure!(within_range, "Input out of range");
        Ok(self.quant_data.as_ref().unwrap().table_output(scaled))
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;
    use proptest::prelude::*;

    use crate::{
        Element,
        layers::Layer,
        model::{Model, test::prove_model},
    };

    use super::*;

    #[test]
    fn test_activation_gelu_proving() -> anyhow::Result<()> {
        let input_shape = vec![3, 100].into();
        let mut model = Model::new_from_input_shapes(vec![input_shape], PaddingMode::NoPadding);
        model.add_consecutive_layer(Layer::Activation(Activation::new_gelu()), None)?;
        model.automatic_output_labelling()?;
        prove_model(model, &mut Default::default()).unwrap();
        Ok(())
    }

    #[test]
    fn test_glu_activation_proving() -> anyhow::Result<()> {
        let input_shape = vec![7, 94].into();
        let mut model = Model::new_from_input_shapes(
            vec![input_shape; 2], // 2 inputs in case of GLU variant
            PaddingMode::NoPadding,
        );
        model.add_consecutive_layer(Activation::new_geglu().into(), None)?;
        model.automatic_output_labelling()?;
        prove_model(model, &mut Default::default()).unwrap();
        Ok(())
    }

    #[test]
    fn test_activation_gelu_quantize() -> anyhow::Result<()> {
        let gelu = GELU::<f32>::new();
        let input_scaling = ScalingFactor::from_scale(1.0, None);
        _ = gelu.quantize(input_scaling)?;
        Ok(())
    }

    #[test]
    fn test_activation_relu_apply() {
        struct TestCase {
            input: Element,
            output: Element,
        }

        impl TestCase {
            pub fn from(input: Element, output: Element) -> Self {
                Self { input, output }
            }
        }
        for case in [
            TestCase::from(-24, 0),
            TestCase::from(0, 0),
            TestCase::from(124, 124),
            TestCase::from(-127, 0),
        ] {
            assert_eq!(Relu::apply(case.input), case.output);
        }
    }

    #[test]
    fn test_activation_gelu_evaluate_f32() -> anyhow::Result<()> {
        let gelu = GELU::<f32>::new();
        let input_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let input_tensor = Tensor::new(vec![1, input_data.len()].into(), input_data.clone());

        let expected_output_data = input_data.iter().map(gelu_float).collect::<Vec<_>>();

        let layer_out = gelu.evaluate::<GoldilocksExt2>(&[&input_tensor], &[])?;
        assert_eq!(layer_out.outputs().len(), 1);
        let output_tensor = &layer_out.outputs()[0];

        assert_eq!(*output_tensor.shape(), vec![1, input_data.len()].into());
        let actual_output_data = output_tensor.get_data();

        actual_output_data
            .iter()
            .zip(expected_output_data.iter())
            .for_each(|(actual, expected)| {
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "Actual: {actual}, Expected: {expected}"
                );
            });
        Ok(())
    }

    proptest! {
        #[test]
        fn gelu_kernel_test(size in 1usize..1024) {
            let shape = Shape::new(vec![size]);
            let tensor = Tensor::<f32>::random(&shape);

            let btensor = tensor.clone().to_btensor::<1>();
            let data = gelu(btensor).to_data().into_vec().expect("Failed to compute GELU");
            let resultb = Tensor::<f32>::new(shape.clone(), data);

            let data = tensor.get_data();
            let data = data.iter().map(gelu_float).collect::<Vec<_>>();
            let result = Tensor::new(shape, data);

            resultb.get_data().iter().zip(result.get_data().iter()).try_for_each(|(left, right)| {
                prop_assert!(
                    (left - right).abs() < 1e-3,
                    "Actual: {left}, Expected: {right}",
                );
                Ok(())
            })?;
        }
    }
}
