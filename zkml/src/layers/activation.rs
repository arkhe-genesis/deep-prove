use crate::{
    Claim, Element, Prover, ProverContext, ScalingFactor, Shape,
    commit::{compute_betas_eval, identity_eval},
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{QuantizeOp, QuantizeOutput},
    },
    lookup::{
        context::{COLUMN_SEPARATOR, LayerLookupContext, LookupWitnessGen, TableType},
        logup_gkr::{
            prover::batch_multiple_sizes_prove, structs::LogUpBatchProof,
            verifier::verify_logup_proof_multiple_sizes,
        },
    },
    model::StepData,
    padding::PaddingMode,
    quantization::{self, Fieldizer},
    tensor::Number,
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

use crate::{quantization::BIT_LEN, tensor::Tensor};

use super::provable::{
    Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, VerifiableCtx,
};

use anyhow::{Result, anyhow, bail, ensure};
const GELU_SCALE_EXP: usize = 12;
const GELU_SCALE_FACTOR: usize = 1 << GELU_SCALE_EXP;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Activation<N> {
    Relu(Relu),
    Gelu(GELU<N>),
}

/// Currently holds the poly info for the output polynomial of the RELU
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct ActivationCtx<E: ExtensionField + Serialize + DeserializeOwned> {
    pub op: Activation<Element>,
    pub lookup_context: LayerLookupContext,
    pub sumcheck_expression: Vec<Expression<E>>,
    pub node_id: NodeId,
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

impl<N> OpInfo for Activation<N> {
    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        match self {
            Activation::Relu(_relu) => format!("RELU: {}", 1 << Relu::num_vars()),
            Activation::Gelu(_gelu) => "GELU".to_string(),
        }
    }

    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes.to_vec() // same as input shapes
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl Evaluate<f32> for Activation<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        _unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<f32, E>> {
        match self {
            Activation::Relu(relu) => Ok(LayerOut::from_vec(
                inputs
                    .iter()
                    .map(|input| relu.op(input))
                    .collect::<Vec<_>>(),
            )),
            Activation::Gelu(gelu) => gelu.evaluate::<E>(inputs, _unpadded_input_shapes),
        }
    }
}

impl QuantizeOp for Activation<f32> {
    type QuantizedOp = Activation<Element>;

    fn quantize_op<S: crate::ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[crate::ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        let num_outputs = self.num_outputs(input_scaling.len());
        let output_scalings = S::scaling_factors_for_node(data, node_id, num_outputs);
        ensure!(
            output_scalings.len() == 1,
            "Output scaling for convolution layer different from 1"
        );
        let q_op = match self {
            Activation::Relu(_) => Activation::Relu(Relu),
            Activation::Gelu(g) => Activation::Gelu(g.quantize(input_scaling[0])?),
        };
        Ok(QuantizeOutput::new(q_op, output_scalings))
    }
}

impl Evaluate<Element> for Activation<Element> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        _unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<Element, E>> {
        let outputs = match self {
            Activation::Relu(relu) => inputs
                .iter()
                .map(|input| relu.op(input))
                .collect::<Vec<_>>(),
            Activation::Gelu(g) => inputs
                .iter()
                .map(|input| input.try_map(|e| g.apply(e)))
                .collect::<Result<Vec<_>>>()?,
        };
        Ok(LayerOut::from_vec(outputs))
    }
}

impl ProveInfo for Activation<Element> {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeId,
        mut aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)> {
        match self {
            Activation::Relu(_) => aux.tables.insert(TableType::Relu),
            // TODO: if we want to save on memory, we can use a pointer to the vector instead
            Activation::Gelu(gelu) => aux.tables.insert(TableType::GELU(
                gelu.quant_data.map(|q| q.table_data).unwrap(),
            )),
        };

        // Set the model polys to be empty
        aux.model_polys = None;
        aux.max_poly_len = aux
            .last_output_shape
            .iter()
            .fold(aux.max_poly_len, |acc, shapes| {
                acc.max(shapes.next_power_of_two().product())
            });
        let (act, lookup_context) = match self {
            Activation::Relu(relu) => (
                Activation::Relu(*relu),
                LayerLookupContext::new(vec![TableType::Relu], vec![1]),
            ),
            Activation::Gelu(g) => (
                Activation::Gelu(g.clone()),
                LayerLookupContext::new(
                    vec![TableType::GELU(g.quant_data.map(|q| q.table_data).unwrap())],
                    vec![1],
                ),
            ),
        };
        // Build the Sumcheck Expression, we presume the polynomials will be loaded in in the order: input_column, output_column, eq_polys
        let sumcheck_expression = Expression::WitIn(0) * Expression::WitIn(2)
            + Expression::WitIn(1)
                * (Expression::Challenge(0, 1, E::ONE, E::ZERO) * Expression::WitIn(2)
                    + Expression::Challenge(0, 2, E::ONE, E::ZERO) * Expression::WitIn(3));
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
        id: NodeId,
        ctx: &'b Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        _step_data: &StepData<E, E>,
        prover: &mut Prover<'c, 'd, E, T, PCS>,
        _store: &mut GenStore,
    ) -> Result<Vec<Claim<E>>> {
        Ok(vec![self.prove_step(prover, last_claims[0], ctx, id)?])
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        step_data: &StepData<Element, E>,
        store: &mut GenStore,
    ) -> Result<LookupWitnessGen<E, PCS>> {
        let outputs = step_data.output_tensors(store)?;
        ensure!(
            step_data.node_inputs.len() == 1,
            "Found more than 1 input tensor in inference step of activation layer"
        );
        ensure!(
            outputs.len() == 1,
            "Found more than 1 output tensor in inference step of activation layer"
        );

        let input_tensors = step_data.input_tensors(store)?;
        let inputs = input_tensors[0].get_data();
        let outputs = outputs[0].get_data();
        debug_assert_eq!(
            inputs.len(),
            outputs.len(),
            "Input and outputs must have the same length",
        );
        let size = inputs.len();

        let mut element_count = HashMap::<Element, u64>::new();
        let mut col_one = Vec::<E::BaseField>::with_capacity(size);
        let mut col_two = Vec::<E::BaseField>::with_capacity(size);
        for (a, b) in inputs.iter().zip(outputs.iter()) {
            let (a, a_field): (Element, E) = match self {
                Activation::Relu(_) => (*a, a.to_field()),
                Activation::Gelu(g) => {
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
        Ok(vec![self.verify_activation(
            verifier,
            last_claims[0],
            proof,
        )?])
    }
    fn write_proof_to_transcript<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        proof.write_commitment(transcript)
    }
}

impl<N> Activation<N> {
    fn table_type(&self) -> TableType {
        match self {
            Activation::Relu(_) => TableType::Relu,
            Activation::Gelu(g) => TableType::GELU(g.quant_data.map(|q| q.table_data).unwrap()),
        }
    }
    #[timed::timed_instrument(name = "Prover::prove_activation_step")]
    pub(crate) fn prove_step<'a, 'b, E, T: Transcript<E>, PCS>(
        &self,
        prover: &mut Prover<'a, 'b, E, T, PCS>,
        last_claim: &Claim<E>,
        step: &ActivationCtx<E>,
        node_id: NodeId,
    ) -> anyhow::Result<Claim<E>>
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

        let either_polys = layer_polys
            .iter()
            .map(|p| Either::Left(p.as_ref()))
            .chain(
                [&logup_eq_poly, &last_claim_eq]
                    .iter()
                    .map(|&p| Either::Left(p)),
            )
            .collect::<Vec<Either<_, _>>>();

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
        let all_evals = state.get_mle_flatten_final_evaluations();
        let point = state.collect_raw_challenges();

        // Add commitment claims to prover
        prover.add_witness_claim(
            node_id,
            vec![(point.clone(), vec![all_evals[0], all_evals[1]])],
        );

        // Add the proof in
        prover.push_proof(
            node_id,
            LayerProof::Activation(ActivationProof {
                io_accumulation: proof,
                evaluations: all_evals[..2].to_vec(),
                lookup: logup_proof,
                commit,
            }),
        );

        let input_claim = match &self {
            Activation::Gelu(g) => {
                let m: E = g.quant_data.as_ref().unwrap().multiplier.to_field();
                let mi = m.inverse();
                let eval = all_evals[0] * mi;
                Claim::new(point.clone(), eval)
            }
            _ => Claim::new(point.clone(), all_evals[0]),
        };

        Ok(input_claim)
    }
}

impl<E: ExtensionField> ActivationCtx<E> {
    pub(crate) fn verify_activation<T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        last_claim: &Claim<E>,
        proof: &ActivationProof<E, PCS>,
    ) -> anyhow::Result<Claim<E>>
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
            max_degree: 2,
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
        let mut witnesses = evaluations.to_vec();
        witnesses.push(lookup_eq);
        witnesses.push(last_claim_eq);

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
            vec![(point.clone(), evaluations.clone())],
        );
        // 4. return the input claim for to be proven at subsequent step
        let input_claim = match &self.op {
            Activation::Relu(_) => Claim::<E>::new(point, evaluations[0]),
            Activation::Gelu(g) => {
                let m: E = g.quant_data.as_ref().unwrap().multiplier.to_field();
                let mi = m.inverse();
                let eval = evaluations[0] * mi;
                Claim::new(point, eval)
            }
        };

        Ok(input_claim)
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
        (float_output * *quantization::MAX as f32).round() as Element
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
        let multiplier = (GELU_SCALE_FACTOR as f32 * input_scaling.scale()).round() as Element;
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
        model.add_consecutive_layer(
            Layer::Activation(Activation::Gelu(GELU::<f32>::new())),
            None,
        )?;
        model.route_output(None)?;
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
