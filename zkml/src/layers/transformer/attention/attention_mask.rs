//! Implementation of various types of attention masks for transformer models.

use crate::{
    Claim, Element, Number, ScalingFactor, Shape, Tensor,
    commit::compute_betas_eval,
    iop::{
        context::{ContextAux, ShapeStep},
        prover::Prover,
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{
            Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp, QuantizeOutput,
            VerifiableCtx,
        },
        transformer::mha::eval_zeroifier_mle,
    },
    model::{NodeID, StepData},
    padding::{PaddingMode, ShapeInfo},
    quantization::Fieldizer,
    tensor::IntoBTensor,
};
use anyhow::{Result, bail, ensure};
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{Expression, util::ceil_log2};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::util::optimal_sumcheck_threads;
use tenstore::GenStore;
use transcript::Transcript;

use burn::tensor::{Tensor as BTensor, TensorData};
use either::Either;
use multilinear_extensions::{
    mle::MultilinearExtension,
    virtual_poly::{VPAuxInfo, eq_eval},
    virtual_polys::VirtualPolynomialsBuilder,
};
use p3_field::FieldAlgebra;
use sumcheck::structs::{IOPProof, IOPProverState, IOPVerifierState};

/// The short name used to identify the attention mask layer
pub const ATTENTION_MASK_LAYER: &str = "MASK";

/// Mask used in attention so that tokens can only see certain values.
/// Masks are assumed to always operate on matrices, if a tensor has higher rank than 2
/// the mask will be applied to the last two dimensions repeatedly.
/// NOTE: the attention mask supports the single token inference with caching AND
/// the regular inference with square matrices for EVALUATION only.
/// Attention mask proving logic only works for the regular inference with square matrices,
/// and that's ok since we prove "full sequence length", we dont prove intermediate
/// cached inferences for now.
#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
pub struct AttentionMask<N> {
    /// Since a casual mask is always square this is the dim size for both the rows and the columns
    pub dim_size: usize,
    span: AttentionSpan,
    /// The value for negative infinity
    negative_infinity: N,
}

/// AttentionSpan determines which entries do the mask keep.
#[derive(Clone, Debug, Serialize, Deserialize, Copy, Default)]
pub enum AttentionSpan {
    /// The mask is applied to each previous token of the one of interest
    #[default]
    Full,
    /// The mask is applied only to the last previous `n` tokens of the one of interest
    Local(usize),
}

impl<N: Number> Default for AttentionMask<N> {
    fn default() -> Self {
        AttentionMask {
            negative_infinity: N::MIN,
            dim_size: Default::default(),
            span: Default::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Mask used in attention so that tokens can only see "previous" values.
pub struct AttentionMaskProof<E: ExtensionField> {
    /// The sumcheck proof for correct application of the attention mask
    sumcheck_proof: IOPProof<E>,
    /// The input evaluations
    evaluations: Vec<E>,
}

impl<N: Number> AttentionMask<N> {
    /// Creates a new mask given the unpadded input shape and the value to use for `-inf`
    pub fn new(dim_size: usize, negative_inf: N) -> AttentionMask<N> {
        AttentionMask {
            dim_size,
            negative_infinity: negative_inf,
            span: Default::default(),
        }
    }
    pub fn with_span(self, span: AttentionSpan) -> anyhow::Result<AttentionMask<N>> {
        if let AttentionSpan::Local(n) = span {
            ensure!(
                n < self.dim_size,
                "Span cannot be greater than the dimension size"
            );
        }
        Ok(AttentionMask { span, ..self })
    }

    /// Pads the [`CasualMask`] for proving purposes
    fn pad(&mut self) -> Result<()> {
        self.dim_size = self.dim_size.next_power_of_two();

        Ok(())
    }

    /// Sets the negative infinity value
    pub fn set_negative_infinity(&mut self, negative_infinity: N) {
        self.negative_infinity = negative_infinity;
    }

    /// Apply the mask to an input, this method requires the input has rank between 2 and 4, and that the final two dims are either equal
    /// or the second to last dim is 1.
    fn apply(&self, input: &Tensor<N>, unpadded_input_shape: &Shape) -> Result<Tensor<N>>
    where
        N: burn::tensor::Element,
        Tensor<N>: IntoBTensor,
    {
        // Check the the input has suitable rank
        let num_input_dims = unpadded_input_shape.rank();
        ensure!(
            (2..=4).contains(&num_input_dims),
            "To apply Attention Mask input need to have rank at least 2 and at most 4, got: {num_input_dims}",
        );

        // Make the mask
        let shape = if num_input_dims < 4 {
            let diff = 4 - num_input_dims;
            Shape::new(vec![1; diff]).extend(input.shape())
        } else {
            input.shape().clone()
        };

        let unpadded_shape = if num_input_dims < 4 {
            let diff = 4 - num_input_dims;
            Shape::new(vec![1; diff]).extend(unpadded_input_shape)
        } else {
            unpadded_input_shape.clone()
        };

        let masked_shape = Shape::new(vec![
            unpadded_shape.dim(0),
            unpadded_shape.dim(1),
            shape.dim(2),
            shape.dim(3),
        ]);

        // input of shape [..., num_heads, q_len, seq_len]
        // if q_len == 1, we're in the caching inference case. Otherwise
        // we're in the regular square matrix case where q_len == seq_len
        let caching_case = shape.dim(-2) == 1;
        let mask = {
            let diff = shape.numel() - masked_shape.numel();
            let num_heads = masked_shape.slice(..masked_shape.rank() - 2).numel();
            let seq_len = unpadded_input_shape.dim(-1);
            let padded_q_len = input.shape().dim(-2);
            let padded_seq_len = input.shape().dim(-1);
            let data = (0..num_heads)
                .flat_map(|_| {
                    (0..padded_q_len).flat_map(move |token| {
                        // if we're in the single token case, then we take the seq_len and assume the current token is the
                        // seq_len-th token. Otherwise, we're in the regular matrix case and we use this token.
                        // -1 because "token" is 0-indexed but seq_len is 1-indexed
                        let qtoken = if caching_case { seq_len - 1 } else { token };
                        let min = match self.span {
                            AttentionSpan::Full => 0,
                            // only look at the most recent n tokens from the current one
                            AttentionSpan::Local(n) => qtoken.saturating_sub(n),
                        };
                        let max = qtoken;
                        (0..padded_seq_len).map(move |others| {
                            let within_span = (min..=max).contains(&others);
                            let keep_clear = within_span;
                            let should_mask = !keep_clear;
                            #[allow(clippy::let_and_return)]
                            should_mask
                        })
                    })
                })
                .chain(std::iter::repeat_n(false, diff))
                .collect::<Vec<bool>>();

            BTensor::<_, 4, _>::from_data(
                TensorData::new(data, shape.to_vec()),
                &Default::default(),
            )
        };

        // Convert the input into a Burn Tensor
        let b_input_data = TensorData::new(input.data().to_vec(), shape.to_vec());
        let b_input = BTensor::<_, 4, <Tensor<N> as IntoBTensor>::Kind>::from_data(
            b_input_data,
            &Default::default(),
        );
        let masked = b_input.mask_fill(mask, self.negative_infinity);
        let masked_data: Vec<N> = masked
            .to_data()
            .into_vec()
            .map_err(|e| anyhow::anyhow!("Failed to apply Casual Mask: {e:?}"))?;

        let output = Tensor::new(input.shape().clone(), masked_data);
        // println!("Mask output: {output}");
        Ok(output)
    }

    /// NOTE: the function does NOT handle the single inference with caching case, and that's ok
    /// since we never want to prove a single token inference with caching enabled.
    /// We always prove the full sequence length, without caching.
    /// However, the evaluation needs to support both cases.
    fn make_lt_poly<E: ExtensionField>(&self, seq_len: usize) -> MultilinearExtension<'_, E> {
        let evals = (0..seq_len)
            .flat_map(|token| {
                (0..seq_len).map(move |other| {
                    let min = match self.span {
                        AttentionSpan::Full => 0,
                        // i - n
                        AttentionSpan::Local(n) => token.saturating_sub(n),
                    };
                    let max = token;
                    if (min..=max).contains(&other) {
                        E::BaseField::ONE
                    } else {
                        E::BaseField::ZERO
                    }
                })
            })
            .collect::<Vec<E::BaseField>>();

        let num_vars = 2 * ceil_log2(seq_len);
        MultilinearExtension::from_evaluations_vec(num_vars, evals)
    }
}

impl AttentionMask<Element> {
    pub(crate) fn prove_internal<E, PCS, T>(
        &self,
        ctx: &AttentionMaskCtx<E>,
        last_claims: Vec<&Claim<E>>,
        mask_proving_data: MaskProvingData<E>,
        prover: &mut Prover<E, T, PCS>,
    ) -> Result<(AttentionMaskProof<E>, Vec<Claim<E>>)>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
        T: Transcript<E>,
    {
        let MaskProvingData {
            batching_challenges,
            batching_point,
            eq_evals,
            input_polys,
        } = mask_proving_data;
        let num_vars = ceil_log2(eq_evals.len());
        let eq_poly = MultilinearExtension::from_evaluations_ext_vec(num_vars, eq_evals);
        // Since the mask is square the padded seq_len is just 1 << (num_vars >> 1)
        let lt_poly = self.make_lt_poly(1 << (num_vars >> 1));
        let input_polys = input_polys
            .into_iter()
            .map(|evals| MultilinearExtension::from_evaluations_ext_vec(num_vars, evals))
            .collect::<Vec<_>>();

        let either_mles = [&eq_poly, &lt_poly]
            .into_iter()
            .chain(input_polys.iter())
            .map(Either::Left)
            .collect::<Vec<_>>();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let expr_builder =
            VirtualPolynomialsBuilder::<E>::new_with_mles(num_threads, num_vars, either_mles);

        let virtual_poly = expr_builder.to_virtual_polys(
            &ctx.sumcheck_expression[..input_polys.len()],
            &batching_challenges,
        );
        let (sumcheck_proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);
        let evaluations = state.get_mle_flatten_final_evaluations()[2..].to_vec();

        let sumcheck_point = state.collect_raw_challenges();

        // We match the number of provided claims, so if we received one claim we output one claim, if we received n claims we output n claims
        let claims = if last_claims.len() == 1 {
            let combined_eval = batching_challenges
                .iter()
                .zip(evaluations.iter())
                .fold(E::ZERO, |acc, (c, e)| acc + (*c) * (*e));
            let full_point = sumcheck_point
                .iter()
                .chain(batching_point.iter())
                .copied()
                .collect::<Vec<_>>();
            vec![Claim::<E>::new(full_point, combined_eval)]
        } else {
            evaluations
                .iter()
                .map(|eval| Claim::<E>::new(sumcheck_point.clone(), *eval))
                .collect()
        };

        Ok((
            AttentionMaskProof {
                sumcheck_proof,
                evaluations,
            },
            claims,
        ))
    }

    pub(crate) fn verify_internal<E, PCS, T>(
        &self,
        proof: &AttentionMaskProof<E>,
        last_claims: &[&Claim<E>],
        mask_verifying_data: MaskVerifyingData<E>,
        verifier: &mut Verifier<E, T, PCS>,
    ) -> Result<Vec<Claim<E>>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        PCS: PolynomialCommitmentScheme<E>,
        T: Transcript<E>,
    {
        let AttentionMaskProof {
            sumcheck_proof,
            evaluations,
        } = proof;

        let MaskVerifyingData {
            batching_challenges,
            batching_point,
            eq_point,
        } = mask_verifying_data;

        let initial_claim = match last_claims.len() {
            1 => last_claims[0].evaluation(),
            _ => last_claims
                .iter()
                .zip(batching_challenges.iter())
                .fold(E::ZERO, |acc, (claim, chal)| {
                    acc + claim.evaluation() * *chal
                }),
        };
        let num_vars = eq_point.len();
        let aux_info = VPAuxInfo {
            max_degree: 3,
            max_num_variables: num_vars,
            ..Default::default()
        };
        let subclaim = IOPVerifierState::<E>::verify(
            initial_claim,
            sumcheck_proof,
            &aux_info,
            verifier.transcript,
        );
        let sumcheck_point = subclaim
            .point
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();

        let dim_vars = num_vars >> 1;
        let eq_eval = eq_eval(&sumcheck_point, &eq_point);

        let (column_point, row_point) = sumcheck_point.split_at(dim_vars);

        let lt_eval = eval_zeroifier_mle(column_point, row_point);
        let neg_inf_field: E = self.negative_infinity.to_field();
        let calc_eval = eq_eval
            * batching_challenges.iter().zip(evaluations.iter()).fold(
                E::ZERO,
                |acc, (chal, eval)| {
                    acc + *chal * (lt_eval * *eval + neg_inf_field * (E::ONE - lt_eval))
                },
            );

        ensure!(
            calc_eval == subclaim.expected_evaluation,
            "Casual Mask verification failed, expected evaluation {:?} got {:?}",
            subclaim.expected_evaluation,
            calc_eval
        );

        // We match the number of provided claims, so if we received one claim we output one claim, if we received n claims we output n claims
        let claims = if last_claims.len() == 1 {
            let combined_eval = batching_challenges
                .iter()
                .zip(evaluations.iter())
                .fold(E::ZERO, |acc, (c, e)| acc + (*c) * (*e));
            let full_point = sumcheck_point
                .iter()
                .chain(batching_point.iter())
                .copied()
                .collect::<Vec<_>>();
            vec![Claim::<E>::new(full_point, combined_eval)]
        } else {
            evaluations
                .iter()
                .map(|eval| Claim::<E>::new(sumcheck_point.clone(), *eval))
                .collect()
        };

        Ok(claims)
    }
}

impl Evaluate<f32> for AttentionMask<f32>
where
    Tensor<f32>: IntoBTensor,
{
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<f32, E>> {
        inputs
            .iter()
            .zip(unpadded_input_shapes.iter())
            .map(|(input, unpadded_shape)| self.apply(input, unpadded_shape))
            .collect::<Result<Vec<Tensor<f32>>>>()
            .map(LayerOut::from_vec)
    }
}
impl Evaluate<Element> for AttentionMask<Element>
where
    Tensor<Element>: IntoBTensor,
{
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<Element, E>> {
        inputs
            .iter()
            .zip(unpadded_input_shapes.iter())
            .map(|(input, unpadded_shape)| {
                ensure!(
                    self.dim_size >= input.shape().dim(-1),
                    "Attention Mask dimension size does not match the input shape: dim_size: {:?}, unpadded_input_shape.dim(-1): {:?}",
                    self.dim_size,
                    input.shape().dim(-1)
                );
                self.apply(input, unpadded_shape)
            })
            .collect::<Result<Vec<Tensor<Element>>>>()
            .map(LayerOut::from_vec)
    }
}

impl PadOp for AttentionMask<Element> {
    fn pad_node(self, _si: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        let mut padded = self;
        padded.pad()?;
        Ok(padded)
    }
}

impl<N: Number> OpInfo for AttentionMask<N> {
    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes.to_vec()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        format!("CasualMask(neg_inf: {:?})", self.negative_infinity)
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl QuantizeOp for AttentionMask<f32> {
    type QuantizedOp = AttentionMask<Element>;

    fn quantize_op<S: crate::ScalingStrategy>(
        self,
        _data: &S::AuxData,
        _node_id: NodeID,
        input_scaling: &[ScalingFactor],
    ) -> Result<QuantizeOutput<Self::QuantizedOp>> {
        // Ensure we have some scaling factors
        ensure!(
            !input_scaling.is_empty(),
            "Cannot quantize Casual Mask as no input scaling factors are provided"
        );

        // If we have multiple input scaling factors we need to make sure all of their quantised domains have the same minimum
        ensure!(
            input_scaling.iter().map(|sf| sf.domain().0).all_equal(),
            "Cannot quantize Casual Mask as not all inputs have the same minimum quantised domain"
        );

        // We just have to replace negative_infinity with its quantized version
        let quantized_negative_infinity = input_scaling
            .first()
            .map(|sf| sf.domain().0)
            .expect("We have ensured there is at least one scaling factor");
        let AttentionMask { dim_size, span, .. } = self;

        let quantised_mask =
            AttentionMask::<Element>::new(dim_size, quantized_negative_infinity).with_span(span)?;

        Ok(QuantizeOutput::new(quantised_mask, input_scaling.to_vec()))
    }
}

impl ProveInfo for AttentionMask<Element> {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeID,
        mut aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)> {
        // We need the previous output shapes to compute the sumcheck expressions
        let shapes = &aux.last_output_shape;

        ensure!(
            shapes.len() == 1,
            "Casual Mask can only be applied to a single input"
        );
        let input_shape = &shapes[0];
        let negative_inf = self.negative_infinity;
        let sumcheck_expression = build_sumcheck_expression::<E>(input_shape, negative_inf);
        let layer_ctx =
            LayerCtx::AttentionMask(AttentionMaskCtx::<E>::new(*self, sumcheck_expression, id));

        // We make sure to update the ContextAux so that it knows there are no model commitments for this layer
        aux.model_polys = None;

        Ok((layer_ctx, aux))
    }
}

fn build_sumcheck_expression<E: ExtensionField>(
    input_shape: &Shape,
    negative_inf: Element,
) -> Vec<Expression<E>> {
    // If the input shape has rank greater than two we treat it as a batch of 2D images.
    // Since we handle it in this way we make the eq_poly and less than poly the first two witness indices.
    let eq_expr = Expression::<E>::WitIn(0);
    let lt_expr = Expression::<E>::WitIn(1);
    if input_shape.rank() > 2 {
        let batch_size = input_shape.slice(..input_shape.rank() - 2).numel();
        (0..batch_size as u16)
            .map(|j| {
                let wit_poly_id = j + 2;
                // Each term is the same it is just the challenge that changes
                eq_expr.clone()
                    * Expression::Challenge(j, 1, E::ONE, E::ZERO)
                    * (lt_expr.clone() * Expression::WitIn(wit_poly_id)
                        + Expression::Constant(Either::Right(negative_inf.to_field()))
                            * (Expression::Constant(Either::Right(E::ONE)) - lt_expr.clone()))
            })
            .collect()
    } else {
        vec![
            eq_expr
                * (lt_expr.clone() * Expression::WitIn(2)
                    + Expression::Constant(Either::Right(negative_inf.to_field()))
                        * (Expression::Constant(Either::Right(E::ONE)) - lt_expr)),
        ]
    }
}

/// Context for the attention mask operation, needed by the verifier to check a mask was applied correctly.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct AttentionMaskCtx<E: ExtensionField> {
    pub op: AttentionMask<Element>,
    pub sumcheck_expression: Vec<Expression<E>>,
    pub node_id: NodeID,
}

impl<E: ExtensionField> AttentionMaskCtx<E> {
    /// Create a new [`AttentionMaskCtx`]
    pub fn new(
        op: AttentionMask<Element>,
        sumcheck_expression: Vec<Expression<E>>,
        node_id: NodeID,
    ) -> Self {
        AttentionMaskCtx {
            op,
            sumcheck_expression,
            node_id,
        }
    }

    /// Getter for the dim size for the mask
    pub fn dim_size(&self) -> usize {
        self.op.dim_size
    }
}

impl<E: ExtensionField> OpInfo for AttentionMaskCtx<E> {
    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes.to_vec()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        format!(
            "CasualMask {{ neg_inf: {:?}, span: {:?} }}",
            self.op.negative_infinity, self.op.span
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<E, PCS> ProvableOp<E, PCS> for AttentionMask<Element>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = AttentionMaskCtx<E>;

    fn prove<'a, 'b, 'c, 'd, T: transcript::Transcript<E>>(
        &'a self,
        node_id: NodeID,
        ctx: &'b Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<'c, 'd, E, T, PCS>,
        store: &mut GenStore,
    ) -> Result<Vec<Claim<E>>> {
        let inputs = step_data.input_tensors(store)?;

        ensure!(
            inputs.len() == 1,
            "CasualMask can only be applied to a single input"
        );

        let mask_proving_data = MaskProvingData::from_claims_and_input(
            &last_claims,
            &inputs[0],
            &step_data.unpadded_input_shapes[0],
            &step_data.unpadded_output_shapes,
            prover.transcript,
        )?;

        let (proof, claims) = self.prove_internal(ctx, last_claims, mask_proving_data, prover)?;
        prover.push_proof(node_id, LayerProof::AttentionMask(proof));
        Ok(claims)
    }
}

/// This function is used to check that all the inputs line up before proving/verifying an [`AttentionMask`].
/// By this we mean that if `last_claims` contains more than one [`Claim`] we check they are all evaluated at the same point and also that
/// one of the following holds:
/// 1) `last_claims.len() == unpadded_output_shapes.len()`
/// 2) `last_claims.len() != 1 && unpadded_output_shapes.len() == 1`, then in this case we check that `last_claims.len() == unpadded_output_shapes[0].slice(..unpadded_output_shapes.rank() - 2).numel()`
fn mask_preamble<E: ExtensionField>(
    last_claims: &[&Claim<E>],
    unpadded_output_shapes: &[Shape],
) -> Result<()> {
    let points_equal = last_claims.iter().map(|claim| claim.point()).all_equal();
    ensure!(
        points_equal,
        "All input claims must be evaluated at the same point"
    );
    if last_claims.len() == unpadded_output_shapes.len() {
        Ok(())
    } else {
        ensure!(
            last_claims.len() != 1 && unpadded_output_shapes.len() == 1,
            "If there is only one output shape then there must be more than one input claim"
        );
        let expected_numel = unpadded_output_shapes[0]
            .slice(..unpadded_output_shapes[0].rank() - 2)
            .numel();
        ensure!(
            last_claims.len() == expected_numel,
            "If there is only one output shape then the number of input claims must match the number of slices in the output shape"
        );
        Ok(())
    }
}

#[derive(Debug, Clone)]
/// Struct storing all information to prove the application of an attention mask correctly without having to do proving work on padded parts.
pub(crate) struct MaskProvingData<E: ExtensionField> {
    /// These values are the evaluations of the eq-poly for the higher dims that aren't from padding
    batching_challenges: Vec<E>,
    /// This is the point used to make the batch challenges
    batching_point: Vec<E>,
    /// This is evaluations of the eq-poly for each of the rank-2 tensors that the mask is applied to
    eq_evals: Vec<E>,
    /// This list of evaluations are the rank-2 tensors forming the input that aren't padding parts
    input_polys: Vec<Vec<E>>,
}

impl<E: ExtensionField> MaskProvingData<E> {
    /// Create a new [`MaskProvingData`]
    pub fn new(
        batching_challenges: Vec<E>,
        batching_point: Vec<E>,
        eq_evals: Vec<E>,
        input_polys: Vec<Vec<E>>,
    ) -> Self {
        MaskProvingData {
            batching_challenges,
            batching_point,
            eq_evals,
            input_polys,
        }
    }

    pub fn from_claims_and_input<T: Transcript<E>>(
        claims: &[&Claim<E>],
        input: &Tensor<E>,
        unpadded_input_shape: &Shape,
        unpadded_output_shapes: &[Shape],
        transcript: &mut T,
    ) -> Result<Self> {
        // First we use `mask_preamble` to check that we can run proving
        mask_preamble(claims, unpadded_output_shapes)?;

        let input_shape = input.shape();

        let rank = input_shape.rank();

        match rank {
            2 => Self::from_claims_rank_two(claims, input),
            3 => Self::from_claims_rank_three(claims, input, unpadded_input_shape, transcript),
            4 => Self::from_claims_rank_four(claims, input, unpadded_input_shape, transcript),
            other => {
                bail!("Attention mask can only be applied to rank 2, 3 or 4 tensors, got {other}")
            }
        }
    }

    fn from_claims_rank_two(claims: &[&Claim<E>], input: &Tensor<E>) -> Result<Self> {
        // In this case there should be a single claim
        ensure!(
            claims.len() == 1,
            "MaskProvingData::from_claims_rank_two can only be used when there is a single claim, got {} claims",
            claims.len()
        );

        // Make the eq_poly
        let point = claims[0].point();
        let eq_evals = compute_betas_eval(point);

        Ok(MaskProvingData::new(
            vec![E::ONE], // We include E::ONE so that the output claim is calculated correctly
            vec![],
            eq_evals,
            vec![input.data().to_vec()],
        ))
    }

    fn from_claims_rank_three<T: Transcript<E>>(
        claims: &[&Claim<E>],
        input: &Tensor<E>,
        unpadded_input_shape: &Shape,
        transcript: &mut T,
    ) -> Result<Self> {
        // Now we make the batching point and the eq_poly
        let first_dim = unpadded_input_shape.dim(0);
        let (batch_evals, batch_point, eq_evals) = match claims.len() {
            1 => {
                // In this case we have to split the point
                let split = input.shape().split_point(claims[0].point())?;
                let batch_point = split[0].to_vec();
                let batch_evals = compute_betas_eval(&batch_point);
                let eq_point = split
                    .into_iter()
                    .skip(1)
                    .rev()
                    .flatten()
                    .copied()
                    .collect::<Vec<E>>();
                let eq_evals = compute_betas_eval(&eq_point);
                (batch_evals, batch_point, eq_evals)
            }
            // in this case we expect the claims to be the evaluation of each of the first_dim strides
            // of the input tensor => eq_poly has the same number of variables of the previous case
            size if size == first_dim => {
                // In this case we have to squeeze challenges from the transcript
                let batch_point = (0..ceil_log2(first_dim))
                    .map(|_| {
                        transcript
                            .sample_and_append_challenge(b"mask_batching")
                            .elements
                    })
                    .collect::<Vec<E>>();

                // The claims have already been passed through the preamble so we can just use the point from the first one
                let point = claims[0].point();
                let eq_evals = compute_betas_eval(point);
                (compute_betas_eval(&batch_point), batch_point, eq_evals)
            }
            other => {
                bail!(
                    "MaskProvingData::from_claims_rank_three can only be used when there is a single claim or unpadded_input_shape.dim(0) = {first_dim} claims, got {other}",
                )
            }
        };

        // Now we make the batching challenges and extract the non-padded parts of the input
        let strides = input.shape().strides();
        let (batching_challenges, input_polys): (Vec<E>, Vec<Vec<E>>) = batch_evals
            .iter()
            .zip(input.data().chunks(strides[0]))
            .take(first_dim)
            .map(|(chal, evals)| (*chal, evals.to_vec()))
            .unzip();

        Ok(MaskProvingData::new(
            batching_challenges,
            batch_point,
            eq_evals,
            input_polys,
        ))
    }

    fn from_claims_rank_four<T: Transcript<E>>(
        claims: &[&Claim<E>],
        input: &Tensor<E>,
        unpadded_input_shape: &Shape,
        transcript: &mut T,
    ) -> Result<Self> {
        // Now we make the batching point and the eq_poly
        let first_dim = unpadded_input_shape.dim(0);
        let second_dim = unpadded_input_shape.dim(1);
        let total_dim = first_dim * second_dim;
        let (batch_evals, batch_point, eq_evals) = match claims.len() {
            1 => {
                // In this case we have to split the point
                let split = input.shape().split_point(claims[0].point())?;
                let batch_point = split[..2]
                    .iter()
                    .rev()
                    .flat_map(|slice| *slice)
                    .copied()
                    .collect::<Vec<E>>();
                let batch_evals = compute_betas_eval(&batch_point);
                let eq_point = split
                    .into_iter()
                    .skip(2)
                    .rev()
                    .flatten()
                    .copied()
                    .collect::<Vec<E>>();
                let eq_evals = compute_betas_eval(&eq_point);
                (batch_evals, batch_point, eq_evals)
            }
            size if size == total_dim => {
                // In this case we have to squeeze challenges from the transcript
                let batch_point = (0..ceil_log2(total_dim))
                    .map(|_| {
                        transcript
                            .sample_and_append_challenge(b"mask_batching")
                            .elements
                    })
                    .collect::<Vec<E>>();

                // The claims have already been passed through the preamble so we can just use the point from the first one
                let point = claims[0].point();
                let eq_evals = compute_betas_eval(point);
                (compute_betas_eval(&batch_point), batch_point, eq_evals)
            }
            other => {
                bail!(
                    "MaskProvingData::from_claims_rank_three can only be used when there is a single claim or {total_dim} claims, got {other}",
                )
            }
        };

        // Now we make the batching challenges and extract the non-padded parts of the input
        let strides = input.shape().strides();
        let (batching_challenges, input_polys): (Vec<E>, Vec<Vec<E>>) = batch_evals
            .chunks(input.shape().dim(1))
            .zip(input.data().chunks(strides[0]))
            .take(first_dim)
            .fold(
                (vec![], vec![]),
                |(mut chal_acc, mut polys_acc), (chals, evals)| {
                    let (chals, evals): (Vec<E>, Vec<Vec<E>>) = chals
                        .iter()
                        .zip(evals.chunks(strides[1]))
                        .take(second_dim)
                        .map(|(chal, evals)| (*chal, evals.to_vec()))
                        .unzip();
                    chal_acc.extend(chals);
                    polys_acc.extend(evals);
                    (chal_acc, polys_acc)
                },
            );

        Ok(MaskProvingData::new(
            batching_challenges,
            batch_point,
            eq_evals,
            input_polys,
        ))
    }
}

#[derive(Debug, Clone)]
/// Struct storing all information to verify a [`AttentionMaskProof`].
pub(crate) struct MaskVerifyingData<E: ExtensionField> {
    /// These values are the evaluations of the eq-poly for the higher dims that aren't from padding
    batching_challenges: Vec<E>,
    /// This is the point used to make the batch challenges
    batching_point: Vec<E>,
    /// This is evaluations of the eq-poly for each of the rank-2 tensors that the mask is applied to
    eq_point: Vec<E>,
}

impl<E: ExtensionField> MaskVerifyingData<E> {
    /// Create a new [`MaskVerifyingData`]
    pub fn new(batching_challenges: Vec<E>, batching_point: Vec<E>, eq_point: Vec<E>) -> Self {
        MaskVerifyingData {
            batching_challenges,
            batching_point,
            eq_point,
        }
    }

    pub fn new_from_claims_and_shape_data<T: Transcript<E>>(
        claims: &[&Claim<E>],
        input_shape: &Shape,
        unpadded_input_shape: &Shape,
        unpadded_output_shapes: &[Shape],
        transcript: &mut T,
    ) -> Result<Self> {
        // First we use `mask_preamble` to check that we can run proving
        mask_preamble(claims, unpadded_output_shapes)?;

        let rank = input_shape.rank();

        match rank {
            2 => Self::from_claims_rank_two(claims),
            3 => {
                Self::from_claims_rank_three(claims, input_shape, unpadded_input_shape, transcript)
            }
            4 => Self::from_claims_rank_four(claims, input_shape, unpadded_input_shape, transcript),
            other => {
                bail!("Attention mask can only be applied to rank 2, 3 or 4 tensors, got {other}")
            }
        }
    }

    fn from_claims_rank_two(claims: &[&Claim<E>]) -> Result<Self> {
        // In this case there should be a single claim
        ensure!(
            claims.len() == 1,
            "MaskProvingData::from_claims_rank_two can only be used when there is a single claim"
        );

        // Make the eq_poly
        let eq_point = claims[0].point().to_vec();

        Ok(MaskVerifyingData::new(vec![], vec![], eq_point))
    }

    fn from_claims_rank_three<T: Transcript<E>>(
        claims: &[&Claim<E>],
        input_shape: &Shape,
        unpadded_input_shape: &Shape,
        transcript: &mut T,
    ) -> Result<Self> {
        // Now we make the batching point and the eq_poly
        let first_dim = unpadded_input_shape.dim(0);
        let (batch_evals, batch_point, eq_point) = match claims.len() {
            1 => {
                // In this case we have to split the point
                let split = input_shape.split_point(claims[0].point())?;
                let batch_point = split[0].to_vec();
                let batch_evals = compute_betas_eval(&batch_point);
                let eq_point = split
                    .into_iter()
                    .skip(1)
                    .rev()
                    .flatten()
                    .copied()
                    .collect::<Vec<E>>();

                (batch_evals[..first_dim].to_vec(), batch_point, eq_point)
            }
            size if size == first_dim => {
                // In this case we have to squeeze challenges from the transcript
                let batch_point = (0..ceil_log2(first_dim))
                    .map(|_| {
                        transcript
                            .sample_and_append_challenge(b"mask_batching")
                            .elements
                    })
                    .collect::<Vec<E>>();

                // The claims have already been passed through the preamble so we can just use the point from the first one
                (
                    compute_betas_eval(&batch_point)[..first_dim].to_vec(),
                    batch_point,
                    claims[0].point.to_vec(),
                )
            }
            other => {
                bail!(
                    "MaskProvingData::from_claims_rank_three can only be used when there is a single claim or unpadded_input_shape.dim(0) = {first_dim} claims, got {other}",
                )
            }
        };

        Ok(MaskVerifyingData::new(batch_evals, batch_point, eq_point))
    }

    fn from_claims_rank_four<T: Transcript<E>>(
        claims: &[&Claim<E>],
        input_shape: &Shape,
        unpadded_input_shape: &Shape,
        transcript: &mut T,
    ) -> Result<Self> {
        // Now we make the batching point and the eq_poly
        let first_dim = unpadded_input_shape.dim(0);
        let second_dim = unpadded_input_shape.dim(1);
        let total_dim = first_dim * second_dim;
        let (batch_evals, batch_point, eq_point) = match claims.len() {
            1 => {
                // In this case we have to split the point
                let split = input_shape.split_point(claims[0].point())?;
                let batch_point = split[..2]
                    .iter()
                    .rev()
                    .flat_map(|slice| *slice)
                    .copied()
                    .collect::<Vec<E>>();
                let batch_evals = compute_betas_eval(&batch_point);
                let eq_point = split
                    .into_iter()
                    .skip(2)
                    .rev()
                    .flatten()
                    .copied()
                    .collect::<Vec<E>>();

                (batch_evals, batch_point, eq_point)
            }
            size if size == total_dim => {
                // In this case we have to squeeze challenges from the transcript
                let batch_point = (0..ceil_log2(total_dim))
                    .map(|_| {
                        transcript
                            .sample_and_append_challenge(b"mask_batching")
                            .elements
                    })
                    .collect::<Vec<E>>();

                // The claims have already been passed through the preamble so we can just use the point from the first one
                (
                    compute_betas_eval(&batch_point),
                    batch_point,
                    claims[0].point().to_vec(),
                )
            }
            other => {
                bail!(
                    "MaskProvingData::from_claims_rank_three can only be used when there is a single claim or {total_dim} claims, got {other}",
                )
            }
        };

        // Now we make the batching challenges and extract the non-padded parts of the input
        let strides = input_shape.strides();
        let batching_challenges = batch_evals
            .chunks(input_shape.dim(1))
            .take(first_dim)
            .flat_map(|chals| {
                chals
                    .chunks(strides[1])
                    .take(second_dim)
                    .flat_map(|inner_chunk| inner_chunk.to_vec())
                    .collect::<Vec<E>>()
            })
            .collect::<Vec<E>>();

        Ok(MaskVerifyingData::new(
            batching_challenges,
            batch_point,
            eq_point,
        ))
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for AttentionMaskCtx<E>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = AttentionMaskProof<E>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        let input_shapes = &shape_step.padded_input_shape;
        ensure!(
            input_shapes.len() == 1,
            "CasualMask can only be applied to a single input"
        );
        ensure!(
            input_shapes[0].rank() >= 2,
            "CasualMask can only be applied to rank 2 or higher tensors"
        );
        let input_shape = &input_shapes[0];
        let unpadded_input_shape = &shape_step.unpadded_input_shape[0];
        let unpadded_output_shapes = &shape_step.unpadded_output_shape;
        let mask_verifying_data = MaskVerifyingData::new_from_claims_and_shape_data(
            last_claims,
            input_shape,
            unpadded_input_shape,
            unpadded_output_shapes,
            verifier.transcript,
        )?;
        self.op
            .verify_internal(proof, last_claims, mask_verifying_data, verifier)
    }

    fn write_proof_to_transcript<T: Transcript<E>>(
        &self,
        _proof: &Self::Proof,
        _transcript: &mut T,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::Rng;

    use crate::{
        layers::{Layer, concat_matmul::ConcatMatMul},
        model::{
            Model,
            test::{F, prove_model},
        },
        padding, rng_from_env_or_random,
    };

    use super::*;

    #[test]
    fn test_attention_mask_proving() -> Result<()> {
        let mut rng = rng_from_env_or_random();

        let dim_size: usize = rng.gen_range(3..10);
        // let dim_size: usize = 5;

        for rank in 3..4 {
            println!("Testing rank: {rank}: DIM SIZE: {dim_size}");
            let mask = AttentionMask::new(dim_size, f32::NEG_INFINITY);
            test_causal_mask_proving_helper(rank, mask)?;
        }
        Ok(())
    }
    #[test]
    fn test_attention_equivalence_cached_vs_full() -> anyhow::Result<()> {
        let seq_len: usize = 3;
        let q_len: usize = 1;
        let num_heads: usize = 2;
        let padded_seq_len = seq_len.next_power_of_two();
        let padded_q_len = q_len.next_power_of_two();
        let padded_num_heads = num_heads.next_power_of_two();
        let mut cached_mask =
            AttentionMask::new(seq_len, f32::NEG_INFINITY).with_span(AttentionSpan::Full)?;
        let mut full_mask =
            AttentionMask::new(seq_len, f32::NEG_INFINITY).with_span(AttentionSpan::Full)?;
        cached_mask.pad()?;
        full_mask.pad()?;
        let cached_input_shape = Shape::new(vec![num_heads, q_len, seq_len]);
        let full_input_shape = Shape::new(vec![num_heads, seq_len, seq_len]);
        let cached_input = Tensor::random(&cached_input_shape);
        let full_input = Tensor::random(&full_input_shape);
        let padded_cached_input = cached_input.pad_next_power_of_two();
        let padded_full_input = full_input.pad_next_power_of_two();
        let (full_it, _head_shape) = padded_full_input.slice_on_dim(0);
        let (cached_it, _) = padded_cached_input.slice_on_dim(0);

        let mut new_full_tensor: Vec<Element> =
            Vec::with_capacity(padded_full_input.shape().numel());
        for (full_head, cached_head) in full_it.zip(cached_it) {
            // for each head, copy the data of the cached tensor into the full tensor.
            // we therefore only need to copy one row of the cached tensor per head
            let to_copy = cached_head.chunks(padded_seq_len).next().unwrap();
            let mut new_full_head = full_head.to_vec();
            let min = padded_seq_len * (seq_len - 1);
            let max = min + padded_seq_len;
            let section = &mut new_full_head[min..max];
            section.copy_from_slice(to_copy);
            new_full_tensor.extend(new_full_head);
        }
        let padded_full_input = Tensor::new(padded_full_input.shape().clone(), new_full_tensor);
        println!("num_heads: {num_heads}, q_len: {q_len}, seq_len: {seq_len}");
        println!(
            "padded_num_heads: {padded_num_heads}, padded_q_len: {padded_q_len}, padded_seq_len: {padded_seq_len}"
        );
        println!("padded_full_input: {padded_full_input:?}");
        println!("padded_cached_input: {padded_cached_input:?}");

        let mask = AttentionMask::new(seq_len, Element::MIN).with_span(AttentionSpan::Full)?;
        let cached_output = mask.apply(&padded_cached_input, cached_input.shape())?;
        let full_output = mask.apply(&padded_full_input, &full_input_shape)?;
        println!("full_output: {full_output:?}");
        println!("cached_output: {cached_output:?}");
        let (full_it, _) = full_output.slice_on_dim(0);
        let (cached_it, _) = cached_output.slice_on_dim(0);
        for (full_head, cached_head) in full_it.zip(cached_it) {
            // just look at the first row
            let cached_row = cached_head.chunks(padded_seq_len).next().unwrap();
            let full_row = full_head.chunks(padded_seq_len).nth(seq_len - 1).unwrap();
            assert_eq!(cached_row, full_row);
        }
        Ok(())
    }

    #[test]
    fn test_attention_mask_inference() -> anyhow::Result<()> {
        #[derive(Debug, Clone)]
        struct TestCase {
            seq_len: usize,
            q_len: usize,
            num_heads: usize,
            span: AttentionSpan,
        }
        let test_cases = vec![
            TestCase {
                seq_len: 3,
                q_len: 3, // regular square matrix case
                num_heads: 2,
                span: AttentionSpan::Local(2),
            },
            TestCase {
                seq_len: 3,
                q_len: 1, // caching inference
                num_heads: 2,
                span: AttentionSpan::Local(2),
            },
            TestCase {
                seq_len: 3,
                q_len: 1, // caching inference
                num_heads: 2,
                span: AttentionSpan::Full,
            },
            TestCase {
                seq_len: 3,
                q_len: 3, // regular square matrix case
                num_heads: 2,
                span: AttentionSpan::Full,
            },
        ];
        for test in test_cases {
            println!("Testing test: {test:?}");
            let seq_len = test.seq_len;
            let q_len = test.q_len;
            let num_heads = test.num_heads;
            let span = test.span;
            let padded_seq_len = seq_len.next_power_of_two();
            let minus_infinity = Element::MIN;
            let mask = AttentionMask::new(seq_len, minus_infinity).with_span(span)?;
            let input_shape = Shape::new(vec![num_heads, q_len, seq_len]);
            // let padded_input_shape = input_shape.next_power_of_two();
            let mut model =
                Model::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::Padding);
            model
                .add_consecutive_layer(Layer::AttentionMask(mask), None)
                .unwrap();
            model.automatic_output_labelling().unwrap();
            let model = padding::pad_model(model)?;
            let input = Tensor::random(&input_shape);
            let padded_input = input.pad_next_power_of_two();
            let mut store = GenStore::default();
            let output = model
                .run::<F>(
                    std::slice::from_ref(&padded_input),
                    Some(vec![input_shape]),
                    &mut store,
                )
                .unwrap();
            let output_tensor = output.outputs().unwrap().pop().unwrap();
            assert_eq!(output_tensor.shape(), padded_input.shape());
            let (input_it, _) = padded_input.slice_on_dim(0);
            let (it, shape) = output_tensor.slice_on_dim(0);
            assert_eq!(
                shape,
                Shape::from(vec![q_len.next_power_of_two(), padded_seq_len])
            );
            println!("input_tensor: {padded_input:?}");
            println!("output_tensor: {output_tensor:?}");
            for (head, (output_head, input_head)) in it.zip(input_it).enumerate() {
                let caching_case = q_len == 1;
                assert_eq!(output_head.len(), input_head.len());
                if head >= num_heads {
                    // no masking for the padding heads
                    assert!(
                        output_head.iter().all(|x| *x == 0),
                        "output_head is not all 0, head: {head}, got: {output_head:?}",
                    );
                    continue;
                }
                for (token, (output_chunk, input_chunk)) in output_head
                    .chunks(padded_seq_len)
                    .zip(input_head.chunks(padded_seq_len))
                    .enumerate()
                {
                    let q_token = if caching_case { seq_len - 1 } else { token };
                    let min_clear = match span {
                        AttentionSpan::Full => 0,
                        AttentionSpan::Local(n) => q_token.saturating_sub(n),
                    };
                    let max_clear = q_token;
                    let expected = input_chunk
                        .iter()
                        .enumerate()
                        .map(|(col, elem)| {
                            if (min_clear..=max_clear).contains(&col) {
                                *elem // we clear the part of the row that is within the span
                            } else {
                                minus_infinity // we mask the rest of the row
                            }
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        output_chunk, expected,
                        "token: {token}, head: {head}, min_clear: {min_clear}, max_clear: {max_clear}, output_chunk: {output_chunk:?}, input_chunk: {input_chunk:?}"
                    );
                    // assert_eq!(output_chunk[min_clear..=max_clear],
                    //    input_chunk[min_clear..=max_clear],
                    //    "token: {token}, head: {head}, output_chunk: {output_chunk:?}, input_chunk: {input_chunk:?}");

                    // assert_eq!(
                    //    output_chunk.iter().take(upper_bound).collect::<Vec<_>>(),
                    //    input_chunk.iter().take(upper_bound).collect::<Vec<_>>(),
                    //    "token: {token}, head: {head}, output_chunk: {output_chunk:?}, input_chunk: {input_chunk:?}",
                    //);
                    // assert!(
                    //    output_chunk.iter().skip(upper_bound).all(|x| *x == minus_infinity),
                    //    "output_chunk[..] is not all {minus_infinity:?}, head: {head}, token: {token}, got: {output_chunk:?}",
                    //);
                }
            }
        }
        Ok(())
    }

    fn test_causal_mask_proving_helper(rank: usize, mask: AttentionMask<f32>) -> Result<()> {
        let mut rng = rng_from_env_or_random();

        let dim_size = mask.dim_size;

        let shape: Shape = (0..rank - 2)
            .map(|_| rng.gen_range(3..8))
            .chain([dim_size, dim_size])
            .collect::<Vec<usize>>()
            .into();
        println!("Testing mask on shape: {shape:?}");
        // we test over a model where concat matmul is the first layer, so we need 2 input shapes
        let input_shape_left = shape
            .slice(..rank - 2)
            .extend(&Shape::new(vec![dim_size, 10]));
        let input_shape_right = shape
            .slice(..rank - 2)
            .extend(&Shape::new(vec![10, dim_size]));

        let mut model = Model::new_from_input_shapes(
            vec![input_shape_left, input_shape_right],
            PaddingMode::NoPadding,
        );

        let mat_mul = ConcatMatMul::new(
            ConcatMatMul::expected_dimension_for_left_input(),
            ConcatMatMul::expected_dimension_for_right_input(),
        );
        let id = model
            .add_consecutive_layer(Layer::ConcatMatMul(mat_mul), None)
            .unwrap();

        let _ = model
            .add_consecutive_layer(Layer::AttentionMask(mask), Some(id))
            .unwrap();

        model.automatic_output_labelling().unwrap();
        model.describe();
        prove_model(model, &mut GenStore::default()).unwrap();
        Ok(())
    }
}
