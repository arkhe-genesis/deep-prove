use crate::{
    Claim, Element, Prover, ScalingFactor, ScalingStrategy, Shape, Tensor,
    commit::same_poly,
    graph::NodeId,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{
            Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp, QuantizeOutput,
            VerifiableCtx,
        },
        requant::Requant,
    },
    model::Step,
    padding::{PaddingMode, ShapeInfo, pad_qkv},
    quantization::model_scaling_factor_from_tensor_and_bias,
    tensor::{CommitmentId, KeyedTensor, TensorTypeParam, WrappedTensor},
    try_unzip, try_unzip_parallel,
    util::from_mle_list_dimensions,
};
use anyhow::{Result, bail, ensure};
use either::Either;
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    Expression,
    mle::{IntoMLE, MultilinearExtension},
    virtual_polys::VirtualPolynomialsBuilder,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::sync::{Arc, Mutex};
use sumcheck::{
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
};
use tenstore::GenStore;
use transcript::{Challenge, Transcript};

/// Short name used to identify the QKV layer
pub const QKV_LAYER: &str = "_QKV";

/// A layer that evaluates the tensor X against the matrices Q, K and V.
/// NOTE: it performs optimizations with the cache, so it actually
/// do the matrix mult only with the last entry of the input.
/// It also outputs only the "small" Q but with the help of caching, it outputs
/// the full K and V matrices as if they were computed using the whole input tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QKV<N> {
    pub(crate) q: KeyedTensor<N>,
    pub(crate) q_bias: Option<KeyedTensor<N>>,
    pub(crate) k: KeyedTensor<N>,
    pub(crate) k_bias: Option<KeyedTensor<N>>,
    pub(crate) v: KeyedTensor<N>,
    pub(crate) v_bias: Option<KeyedTensor<N>>,
    weights_unpadded_shape: Shape, // same shape for Q, K and V
    /// The cache that gets updated at each pass.
    /// interior mutability for the cache to avoid borrowing issues.
    /// Given only the QKV layer needs to update itself, it's a reasonable trade-off.
    pub cache: Arc<Mutex<CacheQKV<N>>>,
    pub(crate) num_heads: usize, // Needed to properly pad matrices for sub-sequent MHA layer
    pub(crate) head_dim: usize,  // Needed to properly pad matrices for sub-sequent MHA layer
    pub(crate) num_groups: usize, // Needed to properly pad matrices for sub-sequent MHA layer
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QKVCtx {
    node_id: NodeId,
    unpadded_shape: Shape, // same shape for Q, K and V
    num_heads: usize,
    head_dim: usize,
    q_weight_key: CommitmentId,
    k_weight_key: CommitmentId,
    v_weight_key: CommitmentId,
    q_bias_key: Option<CommitmentId>,
    k_bias_key: Option<CommitmentId>,
    v_bias_key: Option<CommitmentId>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct QKVProof<E: ExtensionField> {
    /// the actual sumcheck proof proving the QKV matrix multiplications
    pub(crate) sumcheck: IOPProof<E>,
    /// Proof for the aggregation of the claims about the input matrix to
    /// a single claim
    aggregation_proof: same_poly::Proof<E>,
    /// The evaluation of the weight MLEs over the input vector, without the bias.
    /// The verifier needs these evaluations to check the output of the sumcheck proof
    pre_bias_evals: Vec<E>,
    /// The individual evaluations of the individual polynomial for the last random part of the
    /// sumcheck. One for each polynomial involved in the "virtual poly".
    /// There is a pair of evaluations for each output matrix `Q`, `K` and `V`:
    /// the first evaluation in the pair refers to the input matrix MLE, while the second evaluation
    /// in the pair refers to the corresponding weight matrix `W_q`, `W_k`, `W_v`, respectively.
    /// The first pair contains the evaluations for `Q`, the second pair contains the evaluations
    /// for `K`, and the third pair contains the evaluations for `V`
    individual_claims: [(E, E); 3],
}

impl<E: ExtensionField> QKVProof<E> {
    /// Returns the aggregated sumcheck claims `y = f_1(r) * f_2(r) * f_3(r) ...` from the individual claims.
    pub fn individual_to_virtual_claim(&self, batching_challenges: &[Challenge<E>]) -> E {
        self.individual_claims
            .into_iter()
            .zip(batching_challenges)
            .fold(E::ZERO, |acc, (evals, chal)| {
                acc + evals.0 * evals.1 * chal.elements
            })
    }
}

fn padded_weight_shape(unpadded_shape: &Shape, num_heads: usize, head_dim: usize) -> Shape {
    Shape::new(vec![
        unpadded_shape[0].next_power_of_two(),
        head_dim.next_power_of_two() * num_heads.next_power_of_two(),
    ])
}

impl<N: TensorTypeParam> QKV<N> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q: KeyedTensor<N>,
        q_bias: Option<KeyedTensor<N>>,
        k: KeyedTensor<N>,
        k_bias: Option<KeyedTensor<N>>,
        v: KeyedTensor<N>,
        v_bias: Option<KeyedTensor<N>>,
        num_heads: usize,
        // for MHA, num_groups = num_heads
        num_groups: usize,
    ) -> anyhow::Result<Self> {
        match (q_bias.as_ref(), k_bias.as_ref(), v_bias.as_ref()) {
            (Some(q_bias), Some(k_bias), Some(v_bias)) => {
                ensure!(q_bias.rank() == 1, "Bias in QKV layer is not a 1d tensor");
                ensure!(
                    k_bias.shape() == v_bias.shape(),
                    "Incompatible shapes of k and v bias in QKV layer: k = {:?}, v = {:?}",
                    k_bias.shape(),
                    v_bias.shape(),
                );
            }
            (None, None, None) => (),
            // Not a pure restriction but if that happens, probably something went wrong in the model creation
            _ => bail!("QKV layer must have all bias tensors present or absent"),
        };
        // mat mul : [a,b] * [b, c] -> [a, c] + [c]
        let hidden_size = q.shape()[1];
        if let Some(ref q_bias) = q_bias {
            assert_eq!(
                hidden_size,
                q_bias.shape()[0],
                "q.shape() {:?} != q_bias.shape() {:?}",
                q.shape(),
                q_bias.shape()
            );
        }
        ensure!(
            hidden_size.is_multiple_of(num_heads),
            "Expected number of heads to be a divisor of hidden size, but it's not: hidden_size = {hidden_size}, num_heads = {num_heads}"
        );
        let head_dim = hidden_size / num_heads;
        let weights_unpadded_shape = q.shape().clone();
        if num_groups != num_heads {
            ensure!(
                k.shape().dim(1) == v.shape().dim(1),
                "Incompatible shapes of k and v matrices in QKV layer in GQA mode: k = {:?}, v = {:?}",
                k.shape(),
                v.shape()
            );
            ensure!(
                k.shape().dim(1).is_multiple_of(head_dim),
                "Expected number of groups to be a multiple of head dim, but it's not: head_dim = {head_dim}, num_groups = {num_groups} but k has shape {:?}",
                k.shape()
            );
        }
        Ok(Self {
            q,
            q_bias,
            k,
            k_bias,
            v,
            v_bias,
            weights_unpadded_shape,
            num_heads,
            num_groups,
            head_dim,
            cache: Arc::new(Mutex::new(CacheQKV::new())),
        })
    }

    /// Resets the cache to its default empty state. This is useful
    /// when we want to start a new sequence as QKV is the only stateful layer.
    pub(crate) fn reset_cache(&self) {
        self.cache.lock().unwrap().reset();
    }

    // Given the point of a claim referring to a 2d output tensor with `output_num_vars` variables,
    // split the point in 2 sub-points corresponding to the variables on each of the 2 dimensions (i.e., rows and columns)
    fn split_claim_point<E: ExtensionField>(
        claim_point: &[E],
        output_num_vars: (usize, usize),
    ) -> Result<(&[E], &[E])> {
        ensure!(
            claim_point.len() == output_num_vars.0 + output_num_vars.1,
            "Mismatch between size of claim point and number of variables when splitting claim point for QKV layer"
        );
        let point_for_row = &claim_point[output_num_vars.1..];
        let point_for_column = &claim_point[..output_num_vars.1];
        Ok((point_for_row, point_for_column))
    }

    // Build evaluations point for claims related to a pair (input_matrix, weight matrix) produced
    // by the sumcheck protocol in `prove`. Here, weight matrix can be one of 'self.q`, `self.k` and 'self.v`.
    // The method requires the following inputs:
    // - `claim_point`: point of the claim for the MLE of `output matrix = input_matrix*weight_matrix``
    // - `proof_point`: random point employed to prove sumcheck
    // - `output_num_vars` : number of variables of the output matrix
    fn build_points<E: ExtensionField>(
        claim_point: &[E],
        proof_point: &[E],
        output_num_vars: (usize, usize),
    ) -> Result<(Vec<E>, Vec<E>)> {
        let (point_for_row, point_for_column) =
            Self::split_claim_point(claim_point, output_num_vars)?;
        // sumcheck point is on the column variables, which are the low ones
        let input_point = [proof_point, point_for_row].concat();
        // sumcheck point is on the row variables, which are the high ones
        let weight_matrix_point = [point_for_column, proof_point].concat();
        Ok((input_point, weight_matrix_point))
    }

    // Squeeze the challenges required to batch the sumcheck equations employed to prove the layer.
    // It requires as input the output claims for the layer and the evaluations (over the same points of the output
    // claims) of the MLEs of the output tensors before bias addition, which are the claims actually used in the batched
    // sumcheck
    fn challenges_for_batched_sumcheck<E: ExtensionField, T: Transcript<E>>(
        transcript: &mut T,
        last_claims: &[&Claim<E>],
        evals_pre_bias: &[E],
    ) -> Vec<Challenge<E>> {
        // add claims about output tensors without bias to the transcript, to then squeeze the challenge necessary to batch the matrix multiplication
        // sum-check equation
        last_claims
            .iter()
            .zip(evals_pre_bias)
            .for_each(|(&claim, evals)| {
                transcript.append_field_element_exts(&claim.point);
                transcript.append_field_element_ext(evals);
            });
        // We actually need 2 random challenges, but we also return the identity element as the
        // "first challenge" to be able to batch easily with iterators
        [Challenge { elements: E::ONE }]
            .into_iter()
            .chain((0..2).map(|_| transcript.read_challenge()))
            .collect()
    }
}

const IS_PROVABLE: bool = true;

impl<N: TensorTypeParam> OpInfo for QKV<N> {
    /// Returns the shapes of the outputs (in the same order)
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        assert_eq!(input_shapes.len(), 1, "Expected one input for QKV layer");
        let input_shape = input_shapes[0].clone();
        let full_seq_len = if self.cache.lock().unwrap().is_initialized() {
            self.cache.lock().unwrap().full_seq_len() // this assumes the `evaluate` method has already been called for the
        // current input, and so the cache is already updated with the size of the output
        } else {
            input_shape[0]
        };

        match padding_mode {
            PaddingMode::NoPadding => {
                // [q_len, emb_size], [seq_len, emb_size], [seq_len, emb_size]
                vec![
                    vec![input_shape[0], self.weights_unpadded_shape[1]].into(),
                    vec![full_seq_len, self.weights_unpadded_shape[1]].into(),
                    vec![full_seq_len, self.weights_unpadded_shape[1]].into(),
                ]
            }
            PaddingMode::Padding => {
                // compute head_dim from hidden_size and num_heads
                let padded_weight = padded_weight_shape(
                    &self.weights_unpadded_shape,
                    self.num_heads,
                    self.head_dim,
                );
                vec![
                    vec![
                        input_shape[0].next_power_of_two(),
                        padded_weight[1].next_power_of_two(),
                    ]
                    .into(),
                    vec![
                        full_seq_len.next_power_of_two(),
                        padded_weight[1].next_power_of_two(),
                    ]
                    .into(),
                    vec![
                        full_seq_len.next_power_of_two(),
                        padded_weight[1].next_power_of_two(),
                    ]
                    .into(),
                ]
            }
        }
    }

    /// Compute the number of output tensors, given the number of input tensors
    /// `num_inputs`
    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs * 3
    }

    /// Textual description of the operation
    fn describe(&self) -> String {
        format!("QKV [{},{}]", self.q.shape()[0], self.q.shape()[1])
    }

    /// Specify whether the operation needs to be proven or not
    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl<N> QKV<N>
where
    N: TensorTypeParam,
{
    /// Returns x[-1,..] * Q, X * K, X * V
    fn evaluate_internal<E: ExtensionField>(
        &self,
        inputs: &[&WrappedTensor<N>],
    ) -> anyhow::Result<LayerOut<N, E>> {
        ensure!(inputs.len() == 1, "QKV expects 1 input");
        let shape = inputs[0].shape();
        let emb_size = shape.dims[1];
        let q_emb_size = self.q.shape()[0];
        ensure!(
            q_emb_size == emb_size,
            "QKV: q_emb_size {} != emb_size {} (input shape {:?} vs q shape {:?})",
            q_emb_size,
            emb_size,
            shape,
            self.q.shape()
        );

        // NOTE: we take the _whole_ input and not just the last row / token.
        // The reason is because the _first_ time we infere via this layer, this is on the user input which is X token long.
        // The subsequent times, it's just to run with the newly generated token, so there is only one here.
        let input = inputs[0].clone();
        let unpadded_shape: Shape = input.unpadded_shape().clone().into();
        let unpadded_seq_len = unpadded_shape.dim(0);
        let q = WrappedTensor::try_from(&self.q)?;
        let q_bias = self
            .q_bias
            .as_ref()
            .map(WrappedTensor::try_from)
            .transpose()?;
        let k = WrappedTensor::try_from(&self.k)?;
        let k_bias = self
            .k_bias
            .as_ref()
            .map(WrappedTensor::try_from)
            .transpose()?;
        let v = WrappedTensor::try_from(&self.v)?;
        let v_bias = self
            .v_bias
            .as_ref()
            .map(WrappedTensor::try_from)
            .transpose()?;

        // println!("QKV Info");
        // println!("QKV Q: {:?}", q.shape());
        // println!("QKV K: {:?}", k.shape());
        // println!("QKV V: {:?}", v.shape());
        // println!("QKV Input: {:?}", input.shape());
        let q = input.clone().matmul(q)?;
        let q = if let Some(qb) = q_bias {
            q.add(qb.unsqueeze_dim_2())?
        } else {
            q
        };
        let k = input.clone().matmul(k)?;
        let k = if let Some(kb) = k_bias {
            k.add(kb.unsqueeze_dim_2())?
        } else {
            k
        };
        let v = input.matmul(v)?;
        let v = if let Some(vb) = v_bias {
            v.add(vb.unsqueeze_dim_2())?
        } else {
            v
        };

        // TODO: convert cache to WrappedTensor
        #[cfg(test)]
        let (k, v) = {
            let mut cache = self.cache.lock().unwrap();
            cache.stack(k.into_native(), v.into_native(), unpadded_seq_len)?;
            let k = cache.k().into_wrapped();
            let v = cache.v().into_wrapped();
            (k, v)
        };
        #[cfg(not(test))]
        let _ = unpadded_seq_len;

        Ok(LayerOut::from_vec(vec![q, k, v]))
    }
}

impl QuantizeOp for QKV<f32> {
    type QuantizedOp = QKV<Element>;

    /// Convert an operation into its quantized version
    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
        _unpadded_input_shapes: &[Shape],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        let num_outputs = self.num_outputs(input_scaling.len());
        let output_scalings = S::scaling_factors_for_node(data, node_id, num_outputs);
        ensure!(
            output_scalings.len() == num_outputs,
            "Output scaling for QKV layer different from {num_outputs}"
        );
        self.quantize_from_scalings(input_scaling, &output_scalings)
    }
}

impl QKV<f32> {
    fn quantize_from_scalings(
        self,
        input_scaling: &[ScalingFactor],
        output_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<QKV<Element>>> {
        ensure!(input_scaling.len() == 1, "QKV: input_scaling.len() != 1");
        ensure!(output_scaling.len() == 3, "QKV: output_scaling.len() != 3");
        // for each tensor, we look at the scaling factor and the scaling factor of the associated bias
        let (matrices, (biases, requants)): (Vec<_>, (Vec<_>, Vec<_>)) = output_scaling
            .iter()
            .zip(vec![
                (self.q, self.q_bias),
                (self.k, self.k_bias),
                (self.v, self.v_bias),
            ])
            .map(|(output_scaling, (tensor, bias))| {
                let (model_scaling, bias_scaling) = model_scaling_factor_from_tensor_and_bias(
                    &input_scaling[0],
                    &tensor,
                    &bias.as_ref().map(|b| b.tensor()),
                );
                let input_scaling = &input_scaling[0];
                let quantized_matrix = tensor.quantize(&model_scaling);
                let quantized_bias = bias.map(|bias| bias.quantize(&bias_scaling));
                let intermediate_bitsize = quantized_matrix.matmul_output_bitsize(None, None);
                let requant = Requant::from_scaling_factors(
                    *input_scaling,
                    model_scaling,
                    *output_scaling,
                    intermediate_bitsize,
                );
                (quantized_matrix, (quantized_bias, requant))
            })
            .unzip();
        let mut matit = matrices.into_iter();
        let (q, k, v) = (
            matit.next().unwrap(),
            matit.next().unwrap(),
            matit.next().unwrap(),
        );
        let mut biasit = biases.into_iter();
        let (q_bias, k_bias, v_bias) = (
            biasit.next().unwrap(),
            biasit.next().unwrap(),
            biasit.next().unwrap(),
        );
        let quantized_op = QKV::new(
            q,
            q_bias,
            k,
            k_bias,
            v,
            v_bias,
            self.num_heads,
            self.num_groups,
        )?;
        Ok(QuantizeOutput::new(quantized_op, output_scaling.to_vec()).with_requants(requants))
    }
}

impl Evaluate<f32> for QKV<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&WrappedTensor<f32>],
    ) -> anyhow::Result<LayerOut<f32, E>> {
        // f32 inference is only used for quantization parameter computation so we don't want
        // to make it stateful since it would have to make the quantization strategy be aware that
        // we're using caching and it's LLM. It's currently not in scope.
        self.cache.lock().unwrap().reset();
        assert_eq!(
            self.cache.lock().unwrap().full_seq_len(),
            0,
            "Cache should be empty during float evaluation"
        );

        self.evaluate_internal(inputs)
    }
}

impl Evaluate<Element> for QKV<Element> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&WrappedTensor<Element>],
    ) -> anyhow::Result<LayerOut<Element, E>> {
        // as we only want to do the the matmul for the new token, not for the previously generated ones
        // This check is only true if the cache is not empty, i.e. if we've already used the cache before.
        // in case it's the first time we use it, then we accept to get any first user input to put in the cache.
        // NOTE: we dont enforce this check during float inference since this is used only to compute quantization factors
        // and this would force the quantization strategy to be aware of the specificities of the LLM logic.
        if self.cache.lock().unwrap().is_initialized() {
            ensure!(inputs[0].shape().dims[0] == 1, "QKV: seq_len != 1");
        }

        self.evaluate_internal(inputs)
    }
}

impl ProveInfo for QKV<Element> {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeId,
        mut aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)> {
        ensure!(
            aux.last_output_shape.len() == 1,
            "expected one input shape for context of QKV layer"
        );
        ensure!(
            aux.last_output_shape[0][1] == self.q.shape()[0],
            "Number of columns in input matrix ({}) is different from number of rows in Q weight matrix ({})",
            aux.last_output_shape[0][1],
            self.q.shape()[0],
        );
        // reset the cache of QKV as it might be filled with data from a previous inference
        self.reset_cache();
        ensure!(
            self.cache.lock().unwrap().full_seq_len() == 0,
            "Cache should be empty"
        );
        aux.last_output_shape = self.output_shapes(&aux.last_output_shape, PaddingMode::Padding);
        let mut array = vec![
            (self.q.commitment_id(), &self.q),
            (self.k.commitment_id(), &self.k),
            (self.v.commitment_id(), &self.v),
        ];
        if let Some(ref q_bias) = self.q_bias {
            array.push((q_bias.commitment_id(), q_bias));
        }
        if let Some(ref k_bias) = self.k_bias {
            array.push((k_bias.commitment_id(), k_bias));
        }
        if let Some(ref v_bias) = self.v_bias {
            array.push((v_bias.commitment_id(), v_bias));
        }
        aux.model_polys = Some(
            array
                .into_iter()
                .map(|(poly_id, matrix)| {
                    let evals = matrix.pad_next_power_of_two().into_data();
                    (poly_id, evals)
                })
                .collect(),
        );

        let ctx = QKVCtx {
            node_id: id,
            unpadded_shape: self.weights_unpadded_shape.clone(),
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            q_weight_key: self.q.commitment_id(),
            k_weight_key: self.k.commitment_id(),
            v_weight_key: self.v.commitment_id(),
            q_bias_key: self.q_bias.as_ref().map(|q| q.commitment_id()),
            k_bias_key: self.k_bias.as_ref().map(|k| k.commitment_id()),
            v_bias_key: self.v_bias.as_ref().map(|v| v.commitment_id()),
        };

        Ok((LayerCtx::QKV(ctx), aux))
    }
}

impl PadOp for QKV<Element> {
    fn pad_node(self, si: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        pad_qkv(self, si)
    }
}

impl<E: ExtensionField, PCS> ProvableOp<E, PCS> for QKV<Element>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = QKVCtx;

    fn prove<T: Transcript<E>>(
        &self,
        node_id: NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &Step<E, Element, E>,
        prover: &mut Prover<E, T, PCS>,
        store: &mut GenStore,
    ) -> Result<Vec<Claim<E>>> {
        let input_tensors = step_data.input_tensors(store)?;
        let output_tensors = step_data.output_tensors(store)?;

        let expected_num_outputs = self.num_outputs(1);
        ensure!(
            last_claims.len() == expected_num_outputs,
            "Expected {expected_num_outputs} output claims for QKV layer, found {}",
            last_claims.len()
        );
        ensure!(
            step_data.node_inputs.len() == 1,
            "Expected 1 input tenstor in inference data for QKV layer, found {}",
            step_data.node_inputs.len()
        );
        let input = &input_tensors[0];
        ensure!(
            input.shape().is_matrix(),
            "Input tensor for QKV layer is not a matrix"
        );
        let ncols = input.ncols_2d();
        let nrows = self.q.nrows_2d();
        ensure!(
            ncols == nrows,
            "Number of columns in input matrix ({ncols}) different from number of rows in Q weight matrix of QKV layer ({nrows})"
        );
        let expected_output_shape: Shape = vec![input.nrows_2d(), self.q.ncols_2d()].into();
        ensure!(
            output_tensors.len() == expected_num_outputs,
            "Expected {expected_num_outputs} output tensors in inference data for QKV layer, found {}",
            output_tensors.len()
        );
        output_tensors.iter().try_for_each(|out| {
                ensure!(*out.shape() == expected_output_shape,
                    "Expected shape {expected_output_shape:?} for output of QKV layer, found shape {:?}", out.shape(),
                );
                Ok(())
            }
        )?;
        let output_num_vars_2d = output_tensors[0].shape().num_vars_2d(); // we can use the first one since we checked all outputs
        // have the same shape
        let output_num_vars = output_num_vars_2d.0 + output_num_vars_2d.1; // overall number of variables for the MLEs of outputs
        last_claims.iter().try_for_each(|claim| {
            ensure!(claim.point.len() == output_num_vars,
                "Unexpected length of output claim for QKV layer: expected {output_num_vars}, found {}", claim.point.len(),
            );
            Ok(())
        })?;

        // compute claims about the bias polynomials
        let (bias_claims, evals_pre_bias): (Vec<_>, Vec<_>) = try_unzip_parallel(
            last_claims
                .par_iter()
                .zip([&self.q_bias, &self.k_bias, &self.v_bias].par_iter())
                .map(|(&claim, bias_vector)| {
                    let (_, point_for_column) =
                        Self::split_claim_point(&claim.point, output_num_vars_2d)?;
                    if let Some(bias_vector) = bias_vector {
                        ensure!(
                            point_for_column.len() == bias_vector.get_data().len().ilog2() as usize
                        );

                        let eval = bias_vector
                            .to_field::<E>()
                            .into_mle()
                            .evaluate(point_for_column);
                        let bias_claim = Claim::new(point_for_column.to_vec(), eval);
                        // subtract the bias evals from output claims to get claims about the tensors before bias addition
                        let eval_pre_bias = claim.eval - eval;
                        Ok((Some(bias_claim), eval_pre_bias))
                    } else {
                        let eval_pre_bias = claim.eval;
                        Ok((None, eval_pre_bias))
                    }
                }),
        )?;

        let challenges =
            Self::challenges_for_batched_sumcheck(prover.transcript, &last_claims, &evals_pre_bias);

        let input_mle = input.to_mle_2d();

        // Number of variables involved in the sum-check corresponds to the number of columns of the input matrix
        let num_vars = input.shape().num_vars_2d().1;
        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<E>::new(num_threads, num_vars);

        let terms = last_claims
            .iter()
            .zip([&self.q, &self.k, &self.v])
            .map(|(&claim, weight_matrix)| {
                let mut weight_mle = weight_matrix.to_2d_mle();
                let (point_for_row, point_for_column) =
                    Self::split_claim_point(&claim.point, output_num_vars_2d)?;
                let fixed_input_mle = input_mle.fix_high_variables(point_for_row);
                weight_mle.fix_variables_in_place(point_for_column);

                Ok((fixed_input_mle, weight_mle))
            })
            .collect::<Result<Vec<(MultilinearExtension<E>, MultilinearExtension<E>)>, anyhow::Error>>()?;

        let expr = terms.iter().zip(challenges.iter()).fold(
            Expression::Constant(Either::Right(E::ZERO)),
            |acc, ((fi, w), c)| {
                let fi_expr = expr_builder.lift(Either::Left(fi));
                let w_expr = expr_builder.lift(Either::Left(w));
                let challenge = Expression::Constant(Either::Right(c.elements));
                // vp.add_mle_list(vec![fixed_input_mle.into(), weight_mle.into()], coefficient);
                acc + fi_expr * w_expr * challenge
            },
        );

        let virtual_poly = expr_builder.to_virtual_polys(&[expr], &[]);
        let (proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);

        // get claims for all the MLEs involved in sum-check
        let sumcheck_evals = state
            .get_mle_flatten_final_evaluations()
            .chunks(2) // each chunk refers to a pair of (input, weight matrix) MLEs in the sumcheck
            .map(|evals| (evals[0], evals[1]))
            .collect_vec();

        // Build claims corresponding to each evaluation, splitting between claims related to the input matrix
        // and claims related to the weight matrices
        let (input_claims, weight_claims): (Vec<_>, Vec<_>) = try_unzip(
            last_claims
                .iter()
                .zip(&sumcheck_evals)
                .map(|(&claim, evals)| {
                    let (point_for_input, point_for_weight) = Self::build_points(
                        &claim.point,
                        &state.collect_raw_challenges(),
                        output_num_vars_2d,
                    )?;
                    anyhow::Ok((
                        Claim::new(point_for_input, evals.0),
                        Claim::new(point_for_weight, evals.1),
                    ))
                }),
        )?;

        // debug: check input claims
        debug_assert!(input_claims.iter().all(|claim| {
            let eval = input_mle.evaluate(&claim.point);
            claim.eval == eval
        }));

        // Build set of claims to be proven via polynomial commitment opening proof
        let common_claims = weight_claims
            .into_iter()
            .map(Some)
            .chain(bias_claims)
            .zip([
                Some(&ctx.q_weight_key),
                Some(&ctx.k_weight_key),
                Some(&ctx.v_weight_key),
                ctx.q_bias_key.as_ref(),
                ctx.k_bias_key.as_ref(),
                ctx.v_bias_key.as_ref(),
            ])
            // filter the bias claims that are not present
            .filter(|(claim, id)| claim.is_some() && id.is_some())
            .map(|(claim, id)| (id.unwrap().clone(), claim.unwrap()))
            .collect();

        prover.add_common_claims(node_id, common_claims);

        // Aggregate input claims into a single one, which will be returned as output
        let mut same_poly_prover = same_poly::Prover::new(input_mle);

        input_claims
            .into_iter()
            .try_for_each(|claim| same_poly_prover.add_claim(claim))?;

        let (aggregation_proof, aggregated_claim) = same_poly_prover.prove(prover.transcript)?;

        let proof = QKVProof {
            sumcheck: proof,
            aggregation_proof,
            individual_claims: sumcheck_evals.try_into().unwrap(),
            pre_bias_evals: evals_pre_bias,
        };

        prover.push_proof(node_id, LayerProof::QKV(proof));

        Ok(vec![aggregated_claim])
    }
}

impl OpInfo for QKVCtx {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        let weight_shape = match padding_mode {
            PaddingMode::NoPadding => &self.unpadded_shape,
            PaddingMode::Padding => {
                &padded_weight_shape(&self.unpadded_shape, self.num_heads, self.head_dim)
            }
        };

        assert_eq!(
            input_shapes.len(),
            1,
            "Expected only 1 input shape for QKV layer"
        );

        assert_eq!(
            input_shapes[0][1], weight_shape[0],
            "Shape mismatch for QKV ctx: number of columns in input shape different from number of rows of weight matrices {} != {}",
            input_shapes[0][1], weight_shape[0],
        );

        vec![vec![input_shapes[0][0], weight_shape[1]].into(); self.num_outputs(1)]
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs * 3
    }

    fn describe(&self) -> String {
        let padded_matrix_shape = self.unpadded_shape.next_power_of_two();
        format!(
            "QKV [{},{}]",
            padded_matrix_shape[0], padded_matrix_shape[1]
        )
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> VerifiableCtx<E, PCS> for QKVCtx
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    type Proof = QKVProof<E>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        ensure!(
            shape_step.padded_input_shape.len() == 1,
            "Expected 1 input shape for QKV verifier, found {}",
            shape_step.padded_input_shape.len(),
        );
        let padded_input_shape = &shape_step.padded_input_shape[0];
        let expected_num_outputs = self.num_outputs(1);
        ensure!(
            shape_step.padded_output_shape.len() == expected_num_outputs,
            "Expected {expected_num_outputs} shapes for QKV verifier, found {}",
            shape_step.padded_output_shape.len()
        );
        ensure!(
            last_claims.len() == expected_num_outputs,
            "Expected {expected_num_outputs} output claims for QKV verifier, found {}",
            last_claims.len()
        );

        let output_shape = &shape_step.padded_output_shape[0]; // we can just take the first one since all the output shapes
        // are expected to be the same
        let output_num_vars = (
            output_shape[0].ilog2() as usize,
            output_shape[1].ilog2() as usize,
        );

        let bias_presents = [
            self.q_bias_key.is_some(),
            self.k_bias_key.is_some(),
            self.v_bias_key.is_some(),
        ];
        // compute claims for the bias vector, subtracting the `pre_bias_evals` found in the proof from the output claims
        let bias_claims = last_claims
            .iter()
            .zip(&proof.pre_bias_evals)
            .zip(&bias_presents)
            .map(|((&claim, eval), bias_present)| {
                if *bias_present {
                    let bias_eval = claim.eval - *eval;
                    let (_, point_for_column) =
                        QKV::<Element>::split_claim_point(&claim.point, output_num_vars)?;
                    Ok(Some(Claim::new(point_for_column.to_vec(), bias_eval)))
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let challenges = QKV::<Element>::challenges_for_batched_sumcheck(
            verifier.transcript,
            last_claims,
            &proof.pre_bias_evals,
        );

        // use challenge to batch evaluations used in sumcheck
        let batched_evals = proof
            .pre_bias_evals
            .iter()
            .zip(&challenges)
            .fold(E::ZERO, |acc, (eval, chal)| acc + *eval * chal.elements);

        // verify batched sumcheck
        // number of variables in the sum-check is equal to the number of variables corresponding to
        // the columns of input matrix
        let num_vars = padded_input_shape.num_vars()[1];

        let vp_aux = from_mle_list_dimensions(&vec![vec![num_vars, num_vars]; 3]);
        let subclaim = IOPVerifierState::<E>::verify(
            batched_evals,
            &proof.sumcheck,
            &vp_aux,
            verifier.transcript,
        );

        // Build claims corresponding to each evaluation of the MLEs involved in the batched sumcheck,
        // splitting between claims related to the input matrix and claims related to the weight matrices
        let (input_claims, weight_claims): (Vec<_>, Vec<_>) = try_unzip(
            last_claims
                .iter()
                .zip(
                    proof.individual_claims.iter(), // each chunk refers to a pair of (input, weight matrix) MLEs in the sumcheck
                )
                .map(|(&claim, evals)| {
                    let (point_for_input, point_for_weight) = QKV::<Element>::build_points(
                        &claim.point,
                        &subclaim.point.iter().map(|p| p.elements).collect::<Vec<_>>(),
                        output_num_vars,
                    )?;
                    anyhow::Ok((
                        Claim::new(point_for_input, evals.0),
                        Claim::new(point_for_weight, evals.1),
                    ))
                }),
        )?;

        // Build set of claims to be proven via polynomial commitment opening proof
        let common_claims = weight_claims
            .into_iter()
            .map(Some)
            .chain(bias_claims)
            .zip([
                Some(&self.q_weight_key),
                Some(&self.k_weight_key),
                Some(&self.v_weight_key),
                self.q_bias_key.as_ref(),
                self.k_bias_key.as_ref(),
                self.v_bias_key.as_ref(),
            ])
            // there may not be any bias claims
            .filter(|(claim, id)| claim.is_some() && id.is_some())
            .map(|(claim, id)| (id.unwrap().clone(), claim.unwrap()))
            .collect();

        verifier.add_common_claims(self.node_id, common_claims);

        // SUMCHECK verification part
        // Instead of computing the polynomial at the random point requested like this
        // let computed_point = vp.evaluate(
        //     subclaim
        //         .point
        //         .iter()
        //         .map(|c| c.elements)
        //         .collect_vec()
        //         .as_ref(),
        //
        // We compute the evaluation directly from the individual final evaluations of each polynomial
        // involved in the sumcheck the prover's giving,e.g. y(res) = SUM f_i(res)
        ensure!(
            proof.individual_to_virtual_claim(&challenges) == subclaim.expected_evaluation,
            "sumcheck claim failed",
        );

        let sum_check_num_vars = padded_input_shape.iter().product::<usize>().ilog2() as usize;

        let ctx = same_poly::Context::new(sum_check_num_vars);

        let mut same_poly_verifier = same_poly::Verifier::new(&ctx);

        input_claims
            .into_iter()
            .try_for_each(|claim| same_poly_verifier.add_claim(claim))?;

        let aggregated_claim =
            same_poly_verifier.verify(&proof.aggregation_proof, verifier.transcript)?;

        Ok(vec![aggregated_claim])
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheQKV<N> {
    cache_k: Tensor<N>,
    cache_v: Tensor<N>,
    seq_len: usize,
    initialized: bool,
    pub(crate) padding_mode: PaddingMode,
}

impl<N: TensorTypeParam> Default for CacheQKV<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: TensorTypeParam> CacheQKV<N> {
    pub fn new() -> Self {
        Self {
            cache_k: Tensor::new(vec![0].into(), vec![]),
            cache_v: Tensor::new(vec![0].into(), vec![]),
            seq_len: 0,
            initialized: false,
            padding_mode: PaddingMode::NoPadding,
        }
    }
    pub fn reset(&mut self) {
        *self = Self::default();
    }
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
    pub fn full_seq_len(&self) -> usize {
        self.seq_len
    }
    pub fn stack(&mut self, k: Tensor<N>, v: Tensor<N>, seq_len: usize) -> anyhow::Result<()> {
        assert_eq!(
            k.shape(),
            v.shape(),
            "k and v have different shapes {:?} != {:?}",
            k.shape(),
            v.shape()
        );
        if self.initialized {
            assert!(k.shape().is_vector(), "k is not a vector {:?}", k.shape());
            assert_eq!(
                self.cache_k.shape()[1],
                k.shape()[1],
                "cache_k and k have different last dimension {:?} != {:?}",
                self.cache_k.shape(),
                k.shape()
            );
            assert_eq!(
                seq_len, 1,
                "subsequent forward pass after the initial one must have seq_len = 1"
            );

            self.cache_k.concat_from_unpadded(self.seq_len, k, 1)?;
            self.cache_v.concat_from_unpadded(self.seq_len, v, 1)?;
            let expected_first_dim = if let PaddingMode::Padding = self.padding_mode {
                self.cache_k = self.cache_k.pad_next_power_of_two();
                self.cache_v = self.cache_v.pad_next_power_of_two();
                (self.seq_len + 1).next_power_of_two()
            } else {
                self.seq_len + 1
            };
            // we can only check the padded version since the unpadded version is "hidden" inside the shape of the tensor
            assert_eq!(
                self.cache_k.shape()[0],
                expected_first_dim,
                "cache_k shape is not correct {:?} != {:?}",
                self.cache_k.shape()[0],
                expected_first_dim,
            );
            assert_eq!(
                self.cache_v.shape()[0],
                expected_first_dim,
                "cache_v shape is not correct {:?} != {:?}",
                self.cache_v.shape()[0],
                expected_first_dim,
            );
            self.seq_len += 1;
        } else {
            self.cache_k = k;
            self.cache_v = v;
            self.seq_len = seq_len;
            self.initialized = true;
        }
        Ok(())
    }
    pub fn k_shape(&self) -> &Shape {
        self.cache_k.shape()
    }
    pub fn v_shape(&self) -> &Shape {
        self.cache_v.shape()
    }
    pub fn k(&self) -> Tensor<N> {
        self.cache_k.clone()
    }
    pub fn v(&self) -> Tensor<N> {
        self.cache_v.clone()
    }
}

#[cfg(test)]
mod tests {
    use ff_ext::GoldilocksExt2;
    use proptest::prelude::*;
    use std::{fmt::Debug, ops::Range, slice};

    use crate::{
        Shape,
        layers::{Layer, provable::evaluate_layer},
        model::{Model, test::prove_model},
        padding::ShapeData,
        tensor::BShape,
    };

    use super::*;

    impl<N: TensorTypeParam> QKV<N> {
        pub(crate) fn random(
            num_heads: usize,
            emb_size: usize,
            hidden_size: usize,
            bias: bool,
            layer_name: Option<CommitmentId>,
        ) -> Result<Self> {
            let layer_name = layer_name.unwrap_or("QKV".to_string().into());
            let q = KeyedTensor::new(
                format!("{layer_name}_weight_q"),
                Tensor::<N>::random(&vec![emb_size, hidden_size].into()),
            );
            let q_bias = if bias {
                Some(KeyedTensor::new(
                    format!("{layer_name}_bias_q"),
                    Tensor::<N>::random(&vec![hidden_size].into()),
                ))
            } else {
                None
            };
            let k = KeyedTensor::new(
                format!("{layer_name}_weight_k"),
                Tensor::<N>::random(&vec![emb_size, hidden_size].into()),
            );
            let k_bias = if bias {
                Some(KeyedTensor::new(
                    format!("{layer_name}_bias_k"),
                    Tensor::<N>::random(&vec![hidden_size].into()),
                ))
            } else {
                None
            };
            let v = KeyedTensor::new(
                format!("{layer_name}_weight_v"),
                Tensor::<N>::random(&vec![emb_size, hidden_size].into()),
            );
            let v_bias = if bias {
                Some(KeyedTensor::new(
                    format!("{layer_name}_bias_v"),
                    Tensor::<N>::random(&vec![hidden_size].into()),
                ))
            } else {
                None
            };
            Self::new(q, q_bias, k, k_bias, v, v_bias, num_heads, num_heads)
        }
    }

    #[test]
    fn test_qkv_cache() {
        // first token
        let seq_len = 1;
        let emb_size = 2;
        let hidden_size = 4;
        let num_heads = 2;
        let qkv = QKV::random(num_heads, emb_size, hidden_size, true, None).unwrap();
        let input = Tensor::<Element>::random(&vec![1, emb_size].into());
        let output = qkv
            .evaluate::<GoldilocksExt2>(&[&input.as_wrapped()])
            .unwrap()
            .outputs;
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].shape(), BShape::from(vec![1, hidden_size]));
        assert_eq!(output[1].shape(), BShape::from(vec![seq_len, hidden_size]));
        let mut out_k = input.matmul(&qkv.k).add_dim2(qkv.k_bias.as_ref().unwrap());
        assert_eq!(output[1].get_data(), out_k.get_data());
        let mut out_v = input.matmul(&qkv.v).add_dim2(qkv.v_bias.as_ref().unwrap());
        assert_eq!(output[2].shape(), BShape::from(vec![seq_len, hidden_size]));
        assert_eq!(output[2].get_data(), out_v.get_data());
        // second token
        let seq_len = 2;
        let new_token_emb = Tensor::<Element>::random(&vec![1, emb_size].into());
        // we dont concat here because we should only send the last token each time
        // input.concat(new_token_emb.clone());
        let output = qkv
            .evaluate::<GoldilocksExt2>(&[&new_token_emb.as_wrapped()])
            .unwrap()
            .outputs;
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].shape(), BShape::from(vec![1, hidden_size]));
        assert_eq!(output[1].shape(), BShape::from(vec![seq_len, hidden_size]));
        assert_eq!(output[2].shape(), BShape::from(vec![seq_len, hidden_size]));
        let out_q = new_token_emb
            .matmul(&qkv.q)
            .add_dim2(qkv.q_bias.as_ref().unwrap());
        assert_eq!(output[0].get_data(), out_q.get_data());
        out_k.concat(
            new_token_emb
                .matmul(&qkv.k)
                .add_dim2(qkv.k_bias.as_ref().unwrap()),
        );
        assert_eq!(output[1].get_data(), out_k.get_data());
        out_v.concat(
            new_token_emb
                .matmul(&qkv.v)
                .add_dim2(qkv.v_bias.as_ref().unwrap()),
        );
        assert_eq!(output[2].get_data(), out_v.get_data());

        qkv.cache.lock().unwrap().reset();
        assert_eq!(qkv.cache.lock().unwrap().seq_len, 0);
    }

    //#[test]
    // fn test_qkv_no_cache() {
    //    // first token
    //    let seq_len = 3;
    //    let emb_size = 2;
    //    let hidden_size = 3;
    //    let num_heads = 1;
    //    let q = Tensor::<f32>::random(&vec![emb_size, hidden_size].into());
    //    let q_bias = Tensor::<f32>::random(&vec![hidden_size].into());
    //    let k = Tensor::<f32>::random(&vec![emb_size, hidden_size].into());
    //    let k_bias = Tensor::<f32>::random(&vec![hidden_size].into());
    //    let v = Tensor::<f32>::random(&vec![emb_size, hidden_size].into());
    //    let v_bias = Tensor::<f32>::random(&vec![hidden_size].into());
    //    let qkv = QKV::new(
    //        q.clone(),
    //        q_bias.clone(),
    //        k.clone(),
    //        k_bias.clone(),
    //        v.clone(),
    //        v_bias.clone(),
    //        num_heads,
    //    )
    //    .unwrap();
    //    let mut input = Tensor::<f32>::random(&vec![seq_len, emb_size].into());
    //    let output = qkv
    //        .evaluate::<GoldilocksExt2>(&[&input], vec![])
    //        .unwrap()
    //        .outputs;
    //    assert_eq!(output.len(), 3);
    //    assert_eq!(output[0].shape(), vec![seq_len, hidden_size].into());
    //    assert_eq!(output[1].shape(), vec![seq_len, hidden_size].into());
    //    let mut out_k = input.matmul(&k).add_dim2(&k_bias);
    //    assert_eq!(output[1].get_data(), out_k.get_data());
    //    let mut out_v = input.matmul(&v).add_dim2(&v_bias);
    //    assert_eq!(output[2].shape(), vec![seq_len, hidden_size].into());
    //    assert_eq!(output[2].get_data(), out_v.get_data());
    //    // second token
    //    let seq_len = seq_len + 1;
    //    let new_token_emb = Tensor::<f32>::random(&vec![1, emb_size].into());
    //    input.concat(new_token_emb.clone());
    //    let output = qkv
    //        .evaluate::<GoldilocksExt2>(&[&input], vec![])
    //        .unwrap()
    //        .outputs;
    //    assert_eq!(output.len(), 3);
    //    assert_eq!(output[0].shape(), vec![seq_len, hidden_size].into());
    //    assert_eq!(output[1].shape(), vec![seq_len, hidden_size].into());
    //    assert_eq!(output[2].shape(), vec![seq_len, hidden_size].into());
    //    let out_q = input.matmul(&q).add_dim2(&q_bias);
    //    assert_eq!(output[0].get_data(), out_q.get_data());
    //    out_k.concat(new_token_emb.matmul(&k).add_dim2(&k_bias));
    //    assert_eq!(output[1].get_data(), out_k.get_data());
    //    out_v.concat(new_token_emb.matmul(&v).add_dim2(&v_bias));
    //    assert_eq!(output[2].get_data(), out_v.get_data());
    //}

    #[test]
    fn test_qkv_padding() {
        let num_inputs = 57;
        let embedding_size = 77;
        let hidden_size = 35;
        let num_heads = 7;
        let unpadded_input_shape = Shape::new(vec![num_inputs, embedding_size]);
        let weight_shape = Shape::new(vec![embedding_size, hidden_size]);
        let bias_shape = Shape::new(vec![hidden_size]);

        let layer =
            QKV::<Element>::random(num_heads, embedding_size, hidden_size, true, None).unwrap();
        let mut si = vec![ShapeData::new(unpadded_input_shape.clone())]
            .as_slice()
            .into();
        let padded_layer = layer.clone().pad_node(&mut si).unwrap();
        assert_eq!(padded_layer.cache.lock().unwrap().full_seq_len(), 0);

        let padded_weight_shape = weight_shape.next_power_of_two();
        let padded_bias_shape = bias_shape.next_power_of_two();

        let unpadded_output_shapes = layer.output_shapes(
            slice::from_ref(&unpadded_input_shape),
            PaddingMode::NoPadding,
        );
        assert_eq!(unpadded_output_shapes, si.unpadded_input_shapes(),);
        // check unpadded output shapes for padded layer
        let unpadded_output_shapes = padded_layer.output_shapes(
            slice::from_ref(&unpadded_input_shape),
            PaddingMode::NoPadding,
        );
        assert_eq!(unpadded_output_shapes, si.unpadded_input_shapes(),);
        // check padded output shapes
        let padded_input_shape = unpadded_input_shape.next_power_of_two();
        let padded_output_shapes =
            padded_layer.output_shapes(slice::from_ref(&padded_input_shape), PaddingMode::Padding);
        assert_eq!(padded_output_shapes, si.padded_input_shapes(),);
        assert!(matches!(
            padded_layer.cache.lock().unwrap().padding_mode,
            PaddingMode::Padding
        ));
        assert_eq!(padded_layer.cache.lock().unwrap().full_seq_len(), 0);

        assert_eq!(*padded_layer.q.shape(), padded_weight_shape);
        assert_eq!(*padded_layer.k.shape(), padded_weight_shape);
        assert_eq!(*padded_layer.v.shape(), padded_weight_shape);
        assert_eq!(
            *padded_layer.q_bias.as_ref().unwrap().shape(),
            padded_bias_shape
        );
        assert_eq!(
            *padded_layer.k_bias.as_ref().unwrap().shape(),
            padded_bias_shape
        );
        assert_eq!(
            *padded_layer.v_bias.as_ref().unwrap().shape(),
            padded_bias_shape
        );

        // check data in padded layer is the same of original layer
        let head_dim = layer.head_dim;
        assert_eq!(head_dim, hidden_size / num_heads);
        let padded_head_dim = head_dim.next_power_of_two();
        [&layer.q, &layer.k, &layer.v]
            .into_iter()
            .zip([&padded_layer.q, &padded_layer.k, &padded_layer.v])
            .for_each(|(weight, padded_weight)| {
                let padded_weight_shape = padded_weight.shape();
                for i in 0..padded_weight_shape[0] {
                    for j in 0..padded_weight_shape[1] {
                        if i < embedding_size
                            && j % padded_head_dim < head_dim
                            && j / padded_head_dim < num_heads
                        {
                            let original_matrix_index =
                                j / padded_head_dim * head_dim + j % padded_head_dim;
                            assert_eq!(
                                weight.get_2d(i, original_matrix_index),
                                padded_weight.get_2d(i, j)
                            );
                        } else {
                            assert_eq!(0, padded_weight.get_2d(i, j));
                        }
                    }
                }
            });

        // test also evaluation over padded layer
        let mut input = Tensor::<Element>::random(&unpadded_input_shape);
        let output =
            evaluate_layer::<GoldilocksExt2, _, _>(&layer, &[&input.as_wrapped()]).unwrap();

        println!("unpadded input shape: {unpadded_input_shape:?}");
        println!("padded input shape: {padded_input_shape:?}");
        println!("hidden size: {hidden_size}");
        println!("num heads: {num_heads}");
        println!("head dim: {}", hidden_size / num_heads);
        println!("padded head dim: {padded_head_dim}");
        println!("embedding size: {embedding_size}");
        println!("num inputs: {num_inputs}");
        println!(
            "output shapes: {:?}",
            output
                .outputs()
                .iter()
                .map(|o| o.shape())
                .collect::<Vec<_>>()
        );
        println!("unpadded output shapes: {unpadded_output_shapes:?}");
        assert!(
            output
                .outputs()
                .into_iter()
                .zip(&unpadded_output_shapes)
                .all(|(out, expected_shape)| Shape::from(out.shape()) == *expected_shape)
        );

        input.pad_to_shape(padded_input_shape);
        assert_eq!(padded_layer.cache.lock().unwrap().full_seq_len(), 0);
        let padded_output =
            evaluate_layer::<GoldilocksExt2, _, _>(&padded_layer, &[&input.as_wrapped()]).unwrap();

        assert!(
            padded_output
                .outputs()
                .into_iter()
                .zip(&padded_output_shapes)
                .all(|(out, expected_shape)| Shape::from(out.shape()) == *expected_shape)
        );

        // check that padded_output has same values of output in non-padded entries
        output
            .outputs()
            .into_iter()
            .zip(padded_output.outputs())
            .zip([
                &padded_layer.q_bias,
                &padded_layer.k_bias,
                &padded_layer.v_bias,
            ]) // we need to include the bias
            // vectors for the padded rows
            .for_each(|((output, padded_out), padded_bias)| {
                let output = output.to_native();
                let padded_out = padded_out.to_native();
                let padded_out_shape = padded_out.shape();
                for i in 0..padded_out_shape[0] {
                    for j in 0..padded_out_shape[1] {
                        if i < num_inputs {
                            if j % padded_head_dim < head_dim && j / padded_head_dim < num_heads {
                                let original_matrix_index =
                                    j / padded_head_dim * head_dim + j % padded_head_dim;
                                assert_eq!(
                                    output.get_2d(i, original_matrix_index),
                                    padded_out.get_2d(i, j)
                                );
                            } else {
                                assert_eq!(0, padded_out.get_2d(i, j));
                            }
                        } else {
                            assert_eq!(
                                padded_bias.as_ref().unwrap().get_data()[j],
                                padded_out.get_2d(i, j)
                            );
                        }
                    }
                }
            });
    }

    #[test]
    fn test_qkv_already_padded() {
        // use power of 2 dimensions
        let num_inputs = 64;
        let embedding_size = 128;
        let hidden_size = 32;
        let num_heads = 8;
        let unpadded_input_shape = Shape::new(vec![num_inputs, embedding_size]);
        let weight_shape = Shape::new(vec![embedding_size, hidden_size]);
        let bias_shape = Shape::new(vec![hidden_size]);

        let layer =
            QKV::<Element>::random(num_heads, embedding_size, hidden_size, true, None).unwrap();
        let mut si = vec![ShapeData::new(unpadded_input_shape.clone())]
            .as_slice()
            .into();
        let padded_layer = layer.clone().pad_node(&mut si).unwrap();

        let unpadded_output_shapes = layer.output_shapes(
            slice::from_ref(&unpadded_input_shape),
            PaddingMode::NoPadding,
        );
        assert_eq!(unpadded_output_shapes, si.unpadded_input_shapes(),);
        // check unpadded output shapes for padded layer
        let unpadded_output_shapes = padded_layer.output_shapes(
            slice::from_ref(&unpadded_input_shape),
            PaddingMode::NoPadding,
        );
        assert_eq!(unpadded_output_shapes, si.unpadded_input_shapes(),);
        // check padded output shapes
        let padded_output_shapes = padded_layer
            .output_shapes(slice::from_ref(&unpadded_input_shape), PaddingMode::Padding);
        assert_eq!(padded_output_shapes, si.padded_input_shapes(),);

        assert_eq!(*padded_layer.q.shape(), weight_shape);
        assert_eq!(*padded_layer.k.shape(), weight_shape);
        assert_eq!(*padded_layer.v.shape(), weight_shape);
        assert_eq!(*padded_layer.q_bias.as_ref().unwrap().shape(), bias_shape);
        assert_eq!(*padded_layer.k_bias.as_ref().unwrap().shape(), bias_shape);
        assert_eq!(*padded_layer.v_bias.as_ref().unwrap().shape(), bias_shape);

        // check data in padded layer is the same of original layer
        [&layer.q, &layer.k, &layer.v]
            .into_iter()
            .zip([&padded_layer.q, &padded_layer.k, &padded_layer.v])
            .for_each(|(weight, padded_weight)| {
                assert_eq!(weight.get_data(), padded_weight.get_data())
            });
    }

    #[test]
    fn test_proven_qkv_layer() {
        for with_bias in [true, false] {
            let num_inputs = 49;
            let embedding_size = 78;
            let hidden_size = 120;
            let num_heads = 10;

            let input_shape = vec![num_inputs, embedding_size].into();
            let mut model =
                Model::<f32>::new_from_input_shapes(vec![input_shape], PaddingMode::NoPadding);

            let _qkv_node_id = model
                .add_consecutive_layer(
                    Layer::QKV(
                        QKV::random(num_heads, embedding_size, hidden_size, with_bias, None)
                            .unwrap(),
                    ),
                    None,
                )
                .unwrap();

            model.automatic_output_labelling().unwrap();
            model.describe();
            prove_model(model, &mut GenStore::default()).unwrap();
        }
    }

    proptest! {
        #[test]
        fn test_qkv_with_f32(input in any_input::<f32>(1..32, 1..256, 1..8, 1..256)) {
            let Input { q, q_bias, k, k_bias, v, v_bias, num_heads, input } = input;

            let expected_q = input.matmul(&q).add_dim2(&q_bias);
            let expected_k = input.matmul(&k).add_dim2(&k_bias);
            let expected_v = input.matmul(&v).add_dim2(&v_bias);

            let layer = QKV::<f32>::new(q, Some(q_bias), k, Some(k_bias), v, Some(v_bias), num_heads,num_heads).unwrap();
            let computed = layer.evaluate::<GoldilocksExt2>(&[&input.as_wrapped()] ).expect("qkv evaluation must be successful");

            prop_assert_eq!(expected_q.shape(), &Shape::from(computed.outputs[0].shape()));
            prop_assert_eq!(expected_k.shape(), &Shape::from(computed.outputs[1].shape()));
            prop_assert_eq!(expected_v.shape(), &Shape::from(computed.outputs[2].shape()));
            let assert_data = |a: &Tensor<f32>, b: &Tensor<f32>| {
                for (a, b) in a.data().iter().zip(b.data()) {
                    let abs = (a - b).abs();
                    // The differences are in tensor matmul
                    const THRESHOLD: f32 =  1e-3;
                    prop_assert!(abs < THRESHOLD, "Absolute diff {} not within threshold {}", abs, THRESHOLD);
                }
                Ok(())
            };
            assert_data(&expected_q, &computed.outputs[0].to_native())?;
            assert_data(&expected_k, &computed.outputs[1].to_native())?;
            assert_data(&expected_v, &computed.outputs[2].to_native())?;
        }

        #[test]
        fn test_qkv_with_element(input in any_input::<Element>(1..64, 1..64, 1..8, 1..64)) {
            let Input { q, q_bias, k, k_bias, v, v_bias, num_heads, input } = input;

            let expected_q = input.matmul(&q).add_dim2(&q_bias);
            let expected_k = input.matmul(&k).add_dim2(&k_bias);
            let expected_v = input.matmul(&v).add_dim2(&v_bias);

            let layer = QKV::<Element>::new(q, Some(q_bias), k, Some(k_bias), v, Some(v_bias), num_heads,num_heads).unwrap();
            let computed = layer.evaluate::<GoldilocksExt2>(&[&input.as_wrapped()], ).expect("qkv evaluation must be successful");

            prop_assert_eq!(expected_q.shape(), &Shape::from(computed.outputs[0].shape()));
            prop_assert_eq!(expected_k.shape(), &Shape::from(computed.outputs[1].shape()));
            prop_assert_eq!(expected_v.shape(), &Shape::from(computed.outputs[2].shape()));
            prop_assert_eq!(&expected_q, &computed.outputs[0].to_native());
            prop_assert_eq!(&expected_k, &computed.outputs[1].to_native());
            prop_assert_eq!(&expected_v, &computed.outputs[2].to_native());
        }
    }

    struct Input<T> {
        q: KeyedTensor<T>,
        q_bias: KeyedTensor<T>,
        k: KeyedTensor<T>,
        k_bias: KeyedTensor<T>,
        v: KeyedTensor<T>,
        v_bias: KeyedTensor<T>,
        num_heads: usize,
        input: Tensor<T>,
    }

    impl<T> Debug for Input<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Input").finish_non_exhaustive()
        }
    }

    fn any_input<T: TensorTypeParam>(
        num_heads: Range<usize>,
        dim_x: Range<usize>,
        dim_y: Range<usize>,
        dim_input: Range<usize>,
    ) -> impl Strategy<Value = Input<T>> {
        (num_heads, dim_x, dim_y, dim_input).prop_flat_map(
            |(num_heads, dim_x, dim_y, dim_input)| {
                let dim_y = dim_y * num_heads;
                let shape_2d = Shape::new(vec![dim_x, dim_y]);
                let shape_1d = Shape::new(vec![dim_y]);
                let q = Tensor::<T>::any(shape_2d.clone());
                let q_bias = Tensor::<T>::any(shape_1d.clone());
                let k = Tensor::<T>::any(shape_2d.clone());
                let k_bias = Tensor::<T>::any(shape_1d.clone());
                let v = Tensor::<T>::any(shape_2d);
                let v_bias = Tensor::<T>::any(shape_1d);
                let input = Tensor::<T>::any(Shape::new(vec![dim_input, dim_x]));
                (q, q_bias, k, k_bias, v, v_bias, Just(num_heads), input).prop_map(
                    |(q, q_bias, k, k_bias, v, v_bias, num_heads, input)| Input {
                        q: KeyedTensor::new("q_weight", q),
                        q_bias: KeyedTensor::new("q_bias", q_bias),
                        k: KeyedTensor::new("k_weight", k),
                        k_bias: KeyedTensor::new("k_bias", k_bias),
                        v: KeyedTensor::new("v_weight", v),
                        v_bias: KeyedTensor::new("v_bias", v_bias),
                        num_heads,
                        input,
                    },
                )
            },
        )
    }
}
