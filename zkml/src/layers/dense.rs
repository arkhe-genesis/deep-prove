use std::{cmp::Ordering, collections::HashMap};

use crate::{
    Claim, Prover, ScalingStrategy, Shape,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{LayerCtx, LayerProof, requant::Requant},
    model::StepData,
    number::Number,
    padding::{PaddingMode, ShapeInfo, pad_dense},
    quantization::{self, ScalingFactor, model_scaling_factor_from_tensor_and_bias},
    tensor::{KeyedTensor, TensorKey},
    util::from_mle_list_dimensions,
};
use anyhow::{Result, ensure};
use burn::tensor::module::linear;
use either::Either;
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    util::ceil_log2,
    virtual_polys::VirtualPolynomialsBuilder,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::{
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
};
use tenstore::GenStore;
use tracing::warn;
use transcript::Transcript;

use crate::{Element, tensor::Tensor};

use super::provable::{
    Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp, QuantizeOutput,
    VerifiableCtx,
};
use crate::model::NodeID;
/// The short name used to identify a dense layer
pub const DENSE_LAYER: &str = "DENS";

/// Description of the layer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dense<T> {
    pub matrix: KeyedTensor<T>,
    pub bias: Option<KeyedTensor<T>>,
    // set to matrix shape if the matrix is not padded
    pub unpadded_matrix_shape: Shape,
}

/// Information stored in the context (setup phase) for this layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DenseCtx {
    pub node_id: NodeID,
    pub unpadded_matrix_shape: Shape,
    pub padded_matrix_shape: Shape,
    matrix_key: TensorKey,
    bias_key: Option<TensorKey>,
}

/// Proof of the layer.
#[derive(Default, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct DenseProof<E: ExtensionField> {
    /// the actual sumcheck proof proving the mat2vec protocol
    pub(crate) sumcheck: IOPProof<E>,
    /// The evaluation of the bias at the previous claims in the proving flow.
    /// The verifier subtracts this from the previous claim to end up with one claim only
    /// about the matrix, without the bias.
    /// If there is no bias, then there is no bias eval
    bias_eval: Option<E>,
    /// The individual evaluations of the individual polynomial for the last random part of the
    /// sumcheck. One for each polynomial involved in the "virtual poly". Since we only support quadratic right now it's
    /// a flat list.
    individual_claims: Vec<E>,
}

fn output_shape(input_shape: &Shape, matrix_shape: &Shape) -> Shape {
    assert_eq!(
        input_shape.product(),
        matrix_shape[1],
        "matrix_shape must be 2D: input_shape {input_shape:?} vs matrix {matrix_shape:?}"
    );
    Shape::new(vec![matrix_shape[0]])
}

impl<T: Number> Dense<T> {
    pub fn new(matrix: KeyedTensor<T>, bias: KeyedTensor<T>) -> Self {
        Self::new_with(matrix, Some(bias))
    }
    pub fn new_with(matrix: KeyedTensor<T>, bias: Option<KeyedTensor<T>>) -> Self {
        if let Some(ref bbias) = bias {
            assert_eq!(matrix.nrows_2d(), bbias.shape()[0]);
        }
        let unpadded_matrix_shape = matrix.shape().clone();
        Self {
            matrix,
            bias,
            unpadded_matrix_shape,
        }
    }
    pub fn ncols(&self) -> usize {
        self.matrix.ncols_2d()
    }
    pub fn nrows(&self) -> usize {
        self.matrix.nrows_2d()
    }

    pub fn pad_next_power_of_two(self) -> Self {
        let matrix = self.matrix.map_tensor(|t| t.pad_next_power_of_two());
        let bias = self
            .bias
            .map(|b| b.map_tensor(|t| t.pad_1d(matrix.nrows_2d())));
        Self {
            matrix,
            bias,
            unpadded_matrix_shape: self.unpadded_matrix_shape,
        }
    }

    pub fn output_shape(&self, input_shape: &Shape, padding_mode: PaddingMode) -> Shape {
        let matrix_shape = match padding_mode {
            PaddingMode::NoPadding => self.unpadded_matrix_shape.clone(),
            PaddingMode::Padding => self.unpadded_matrix_shape.next_power_of_two(),
        };
        output_shape(input_shape, &matrix_shape)
    }

    fn num_outputs(num_inputs: usize) -> usize {
        assert_eq!(num_inputs, 1);
        1
    }
}

const IS_PROVABLE: bool = true;

impl<N: Number> OpInfo for Dense<N> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes
            .iter()
            .map(|shape| self.output_shape(shape, padding_mode))
            .collect()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        Self::num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        format!(
            "Dense: ({}x{}) + bias ({})",
            self.nrows(),
            self.ncols(),
            !self
                .bias
                .as_ref()
                .map(|a| a
                    .get_data()
                    .iter()
                    .all(|x| x.compare(&N::default()) == Ordering::Equal))
                .unwrap_or(true)
        )
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl Evaluate<Element> for Dense<Element> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        _unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<Element, E>> {
        ensure!(
            inputs.len() == 1,
            "Found more than 1 input when evaluating dense layer"
        );
        ensure!(
            inputs[0].shape().product() == self.matrix.shape().dim(1),
            "incompatible dense evaluation shapes: input {:?} vs matrix {:?}",
            inputs[0].shape(),
            self.matrix.shape()
        );

        let matrix = self.matrix.tensor().to_btensor::<2>();
        let input = inputs[0].to_flatten().to_btensor::<1>();
        let bias = self.bias.as_ref().map(|b| b.to_flatten().to_btensor::<1>());

        // NOTE: Can not use the [burn::tensor::module::linear] because it
        // is defined only for floats
        let input = input.unsqueeze_dim(1);
        let matmul = matrix.matmul(input);
        let matmul = matmul.squeeze(1);
        let res = match bias {
            Some(b) => matmul.add(b),
            None => matmul,
        };

        let data = res.to_data().into_vec().expect("Failed to compute Dense");
        let shape = Shape::new(vec![data.len()]);
        let out = Tensor::<Element>::new(shape, data);
        Ok(LayerOut::from_vec(vec![out]))
    }
}

impl Evaluate<f32> for Dense<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        _unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<f32, E>> {
        ensure!(
            inputs.len() == 1,
            "Found more than 1 input when evaluating dense layer"
        );
        let input = inputs[0];

        let matrix = self.matrix.tensor().to_btensor::<2>();
        let input = input.to_flatten().to_btensor::<1>();
        let bias = self.bias.as_ref().map(|b| b.to_flatten().to_btensor::<1>());
        let res = linear(input, matrix.transpose(), bias);

        let data = res.to_data().into_vec().expect("Failed to compute Dense");
        let shape = Shape::new(vec![data.len()]);
        let out = Tensor::<f32>::new(shape, data);

        Ok(LayerOut::from_vec(vec![out]))
    }
}

impl ProveInfo for Dense<Element> {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeID,
        mut aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)> {
        // construct dimension of the polynomial given to the sumcheck
        aux.last_output_shape
            .iter_mut()
            .for_each(|shape| *shape = Shape::new(vec![self.nrows()]));

        let dense_info = LayerCtx::Dense(DenseCtx {
            node_id: id,
            unpadded_matrix_shape: self.unpadded_matrix_shape.clone(),
            padded_matrix_shape: self.matrix.shape().clone(),
            matrix_key: self.matrix.key(),
            bias_key: self.bias.as_ref().map(|b| b.key()),
        });

        let weights_evals = self.matrix.pad_next_power_of_two().into_data();
        let bias_evals = self
            .bias
            .as_ref()
            .map(|b| b.pad_next_power_of_two().into_data());

        aux.model_polys = {
            let mut model_polys = HashMap::new();
            model_polys.insert(self.matrix.key(), weights_evals);
            if let Some(bias) = &self.bias {
                model_polys.insert(
                    bias.key(),
                    bias_evals.expect("No bias evals found in Dense Layer"),
                );
            }
            Some(model_polys)
        };
        Ok((dense_info, aux))
    }
}

impl PadOp for Dense<Element> {
    fn pad_node(self, si: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        pad_dense(self, si)
    }
}

impl Dense<f32> {
    // Quantize a dense layer using scaling factor of input and output
    fn quantize_from_scalings(
        self,
        input_scaling: &[ScalingFactor],
        output_scaling: ScalingFactor,
    ) -> anyhow::Result<QuantizeOutput<Dense<Element>>> {
        let (model_scaling, bias_scaling) = model_scaling_factor_from_tensor_and_bias(
            &input_scaling[0],
            &self.matrix,
            &self.bias.as_ref().map(|b| b.tensor()),
        );
        ensure!(
            input_scaling.len() == 1,
            "Number of input scaling factor for dense layer different from 1"
        );
        let input_scaling = &input_scaling[0];
        let quantized_dense = self.quantize(&model_scaling, &bias_scaling);
        let intermediate_bit_size = quantized_dense.output_bitsize();
        let requant = Requant::from_scaling_factors(
            *input_scaling,
            model_scaling,
            output_scaling,
            intermediate_bit_size,
        );

        Ok(QuantizeOutput::new(quantized_dense, vec![output_scaling]).with_requant(requant))
    }
}

impl QuantizeOp for Dense<f32> {
    type QuantizedOp = Dense<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeID,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        let num_outputs = self.num_outputs(input_scaling.len());
        let mut output_scalings = S::scaling_factors_for_node(data, node_id, num_outputs);
        ensure!(
            output_scalings.len() == 1,
            "Output scaling for convolution layer different from 1"
        );
        let output_scaling = output_scalings.pop().unwrap();
        self.quantize_from_scalings(input_scaling, output_scaling)
    }
}

impl<E, PCS> ProvableOp<E, PCS> for Dense<Element>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = DenseCtx;

    fn prove<T: Transcript<E>>(
        &self,
        id: NodeID,
        ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
        store: &mut GenStore,
    ) -> Result<Vec<Claim<E>>> {
        let input_tensor = step_data.input_tensor_at(0, store)?;
        let output_tensor = step_data.output_tensor_at(0, store)?;

        Ok(vec![self.prove_step(
            prover,
            last_claims[0],
            &input_tensor,
            &output_tensor,
            ctx,
            id,
        )?])
    }
}

impl OpInfo for DenseCtx {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes
            .iter()
            .map(|shape| self.output_shape(shape, padding_mode))
            .collect()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        Dense::<Element>::num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        format!(
            "Dense: ({},{})",
            self.padded_matrix_shape[0], self.padded_matrix_shape[1],
        )
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for DenseCtx
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = DenseProof<E>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        _shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        Ok(vec![self.verify_dense(verifier, last_claims[0], proof)?])
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

impl Dense<f32> {
    /// Quantize the parameters of the dense layer. It uses a custom scaling factor `bias_s` for
    /// the bias, if provided, otherwise the same scaling factor of the weights (i.e., `s`) is used
    pub fn quantize(self, s: &ScalingFactor, bias_s: &ScalingFactor) -> Dense<Element> {
        let matrix = self.matrix.quantize(s);
        let bias = self.bias.map(|b| b.quantize(bias_s));
        Dense::<Element> {
            matrix,
            bias,
            unpadded_matrix_shape: self.unpadded_matrix_shape,
        }
    }

    pub fn new_from_weights(weights: KeyedTensor<f32>, bias: Option<KeyedTensor<f32>>) -> Self {
        let unpadded_matrix_shape = weights.shape().clone();
        Self {
            matrix: weights,
            bias,
            unpadded_matrix_shape,
        }
    }

    /// TODO: compute two different scaling factors for weights and bias
    pub fn max_abs_weight(&self) -> f32 {
        let max_weight = self.matrix.max_abs_output();
        let max_bias = self.bias.as_ref().map(|b| b.max_abs_output());
        let distance = max_bias
            .map(|b| (max_weight - b).abs() / max_weight)
            .unwrap_or(0.0);
        if distance > 0.1 {
            warn!(
                "max_abs_weight DENSE: distance between max_weight and max_bias is too large: {:.2}%",
                distance * 100.0
            );
        }
        self.matrix.max_abs_output().max(
            self.bias
                .as_ref()
                .map(|b| b.max_abs_output())
                .unwrap_or(f32::MIN),
        )
    }
}

impl Dense<Element> {
    /// Returns the (min,max) output range of the dense layer for a given input range.
    pub fn output_range(&self, _min_input: Element, _max_input: Element) -> (Element, Element) {
        // formula is 2^{2 * BIT_LEN + log(c) + 1} where c is the number of columns and +1 because of the bias
        let ncols = self.matrix.ncols_2d() as u32;
        // - 1 because numbers are signed so only half of the range is used when doing multiplication
        let power = 2 * (*quantization::BIT_LEN as u32 - 1) + ncols.ilog2() + 1;
        let min = -(2u64.pow(power) as Element);
        let max = 2u64.pow(power) as Element;
        (min, max)
    }

    /// Returns the maximum size in bits of the output
    pub fn output_bitsize(&self) -> usize {
        // formula is 2^{2 * BIT_LEN + log(c) + 1} where c is the number of columns and +1 because of the bias
        let ncols = self.matrix.ncols_2d();
        // - 1 because numbers are signed so only half of the range is used when doing multiplication
        2 * (*quantization::BIT_LEN - 1) + ceil_log2(ncols) + 1
    }

    #[timed::timed_instrument(name = "Prover::prove_dense")]
    pub fn prove_step<'b, E, T, PCS>(
        &self,
        prover: &mut Prover<E, T, PCS>,
        last_claim: &Claim<E>,
        input: &Tensor<E>,
        output: &Tensor<E>,
        _info: &DenseCtx,
        id: NodeID,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        let matrix = &self.matrix;
        let (nrows, ncols) = (self.nrows(), self.ncols());
        assert_eq!(
            nrows,
            output.get_data().len(),
            "dense proving: nrows {} vs output {}",
            nrows,
            output.get_data().len()
        );
        assert_eq!(
            nrows.ilog2() as usize,
            last_claim.point.len(),
            "something's wrong with the randomness"
        );
        assert_eq!(
            ncols,
            input.get_data().len(),
            "something's wrong with the input"
        );
        // Evaluates the bias at the random point so verifier can subtract the evaluation
        // from the sumcheck claim that is only about the matrix2vec product.
        // If there is no bias, then then there is no bias claim as well !
        let bias_eval = if let Some(bias) = &self.bias {
            assert_eq!(
                bias.get_data().len().ilog2() as usize,
                last_claim.point.len(),
                "something's wrong with the randomness"
            );
            Some(bias.to_field::<E>().into_mle().evaluate(&last_claim.point))
        } else {
            None
        };
        // construct the MLE combining the input and the matrix
        let mut mat_mle: MultilinearExtension<'_, E> = matrix.to_2d_mle();
        // fix the variables from the random input
        // NOTE: here we must fix the HIGH variables because the MLE is addressing in little
        // endian so (rows,cols) is actually given in (cols, rows)
        // mat_mle.fix_variables_in_place_parallel(partial_point);
        mat_mle.fix_high_variables_in_place(&last_claim.point);
        let input_mle: MultilinearExtension<'_, E> = input.get_data().to_vec().into_mle();
        let num_vars = input_mle.num_vars();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<E>::new(num_threads, num_vars);
        let mat_expr = expr_builder.lift(Either::Left(&mat_mle));
        let input_expr = expr_builder.lift(Either::Left(&input_mle));
        let dense_expr = mat_expr * input_expr;
        let virtual_poly = expr_builder.to_virtual_polys(&[dense_expr], &[]);
        assert_eq!(mat_mle.num_vars(), input_mle.num_vars());

        let (proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);

        // PCS part: here we need to create an opening proof for the final evaluation of the matrix poly
        // Note we need the _full_ input to the matrix since the matrix MLE has (row,column) vars space
        let point = [
            state.collect_raw_challenges().as_slice(),
            last_claim.point.as_slice(),
        ]
        .concat();
        let eval = state.get_mle_flatten_final_evaluations()[0];
        // add the bias claim over the last claim input, since that is what is needed to "remove" the bias
        // to only verify the matrix2vec product via the sumcheck proof.
        let bias_claim = bias_eval
            .as_ref()
            .map(|b| Claim::new(last_claim.point.clone(), *b));
        let weights_claim = Claim::new(point, eval);

        // Add common commitment claims to be proven
        let common_claims = {
            let mut claims = HashMap::new();
            claims.insert(self.matrix.key(), weights_claim);
            if let Some(bias) = &self.bias {
                claims.insert(
                    bias.key(),
                    bias_claim.expect("No bias claim found when proving Dense Layer"),
                );
            }
            claims
        };
        prover.add_common_claims(id, common_claims);

        // the claim that this proving step outputs is the claim about not the matrix but the vector poly.
        // at next step, that claim will be proven over this vector poly (either by the next dense layer proving, or RELU etc).
        let claim = Claim {
            point: state.collect_raw_challenges(),
            eval: state.get_mle_flatten_final_evaluations()[1],
        };
        prover.push_proof(
            id,
            LayerProof::Dense(DenseProof {
                sumcheck: proof,
                bias_eval,
                individual_claims: state.get_mle_flatten_final_evaluations(),
            }),
        );
        Ok(claim)
    }
}

impl DenseCtx {
    pub fn output_shape(&self, input_shape: &Shape, mode: PaddingMode) -> Shape {
        let mat_shape = match mode {
            PaddingMode::NoPadding => &self.unpadded_matrix_shape,
            PaddingMode::Padding => &self.padded_matrix_shape,
        };
        output_shape(input_shape, mat_shape)
    }
    pub(crate) fn verify_dense<
        E: ExtensionField,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        last_claim: &Claim<E>,
        proof: &DenseProof<E>,
    ) -> anyhow::Result<Claim<E>> {
        ensure!(
            self.bias_key.is_some() == proof.bias_eval.is_some(),
            "bias eval is missing while expected"
        );
        // Subtract the bias evaluation from the previous claim to remove the bias
        // if there is none that just means the last claim is the output of the dense matrix already
        let eval_no_bias = if let Some(ref be) = proof.bias_eval {
            last_claim.eval - *be
        } else {
            last_claim.eval
        };
        // TODO: currently that API can panic - should remove panic for error
        let matrix_num_vars = self.padded_matrix_shape.num_vars()[1];
        let matrix_poly_aux = from_mle_list_dimensions(&[vec![matrix_num_vars, matrix_num_vars]]);
        let subclaim = IOPVerifierState::<E>::verify(
            eval_no_bias,
            &proof.sumcheck,
            &matrix_poly_aux,
            verifier.transcript,
        );

        // MATRIX OPENING PART
        // pcs_eval means this evaluation should come from a PCS opening proof
        // TODO: no collecting should be done here
        let pcs_eval_input = subclaim
            .point
            .iter()
            .map(|p| p.elements)
            .collect_vec()
            .iter()
            .chain(last_claim.point.iter())
            .cloned()
            .collect_vec();
        // 0 because Matrix comes first in Matrix x Vector
        // Note we don't care about verifying that for the vector since it's verified at the next
        // step.
        let pcs_eval_output = proof.individual_claims[0];

        let weights_claim = Claim::new(pcs_eval_input, pcs_eval_output);

        // add the common commitment claims to be verified
        let common_claims = {
            let mut claims = HashMap::new();
            claims.insert(self.matrix_key.clone(), weights_claim);
            if let Some(ref be) = proof.bias_eval {
                let bias_claim = Claim::new(last_claim.point.clone(), *be);
                claims.insert(self.bias_key.clone().unwrap(), bias_claim);
            }
            claims
        };
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
            proof.individual_to_virtual_claim() == subclaim.expected_evaluation,
            "sumcheck claim failed",
        );

        // the output claim for this step that is going to be verified at next step
        Ok(Claim {
            // the new randomness to fix at next layer is the randomness from the sumcheck !
            point: subclaim.point.iter().map(|p| p.elements).collect_vec(),
            // the claimed sum for the next sumcheck is MLE of the current vector evaluated at the
            // random point. 1 because vector is secondary.
            eval: proof.individual_claims[1],
        })
    }
}

impl<E: ExtensionField> DenseProof<E> {
    /// Returns the individual claims f_1(r) f_2(r)  f_3(r) ... at the end of a sumcheck multiplied
    /// together
    pub fn individual_to_virtual_claim(&self) -> E {
        self.individual_claims
            .iter()
            .fold(E::ONE, |acc, e| acc * *e)
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;
    use proptest::prelude::*;
    use std::{fmt::Debug, ops::Range};

    use crate::{
        layers::{Layer, provable::evaluate_layer},
        model::{Model, test::prove_model},
    };

    use super::*;

    impl<T: Number> Dense<T> {
        /// Require a `layer_name` in case there is the need to use different tensor
        /// keys form the default ones
        pub fn random(shape: Shape, layer_name: Option<TensorKey>) -> Self {
            assert_eq!(shape.len(), 2);
            let (nrows, ncols) = (shape[0], shape[1]);
            let layer_name = layer_name.unwrap_or("dense".to_string().into());
            let matrix = KeyedTensor::new(
                format!("{layer_name}_weight"),
                Tensor::<T>::random(&vec![nrows, ncols].into()),
            );

            // let bias = Tensor::random(vec![nrows]);
            let bias = KeyedTensor::new(
                format!("{layer_name}_bias"),
                Tensor::<T>::random(&vec![nrows].into()),
            );
            Self::new(matrix, bias)
        }
    }

    #[test]
    fn test_dense_pad_next_power_of_two() {
        // Create a Dense layer with non-power-of-two dimensions
        let matrix = KeyedTensor::new(
            "dense_weight",
            Tensor::<Element>::matrix_from_coeffs(vec![
                vec![1, 2, 3],
                vec![4, 5, 6],
                vec![7, 8, 9],
            ])
            .unwrap(),
        );

        let bias = KeyedTensor::new(
            "dense_bias",
            Tensor::<Element>::new(vec![3].into(), vec![10, 11, 12]),
        );

        let dense = Dense::new(matrix, bias);

        // Pad to next power of two
        let padded = dense.pad_next_power_of_two();

        // Check padded dimensions are powers of two
        let padded_dims = padded.matrix.shape();
        assert_eq!(padded_dims[0], 4); // Next power of 2 after 3
        assert_eq!(padded_dims[1], 4); // Next power of 2 after 3

        // Check bias is padded
        let bias_dims = padded.bias.as_ref().unwrap().shape();
        assert_eq!(bias_dims[0], 4); // Next power of 2 after 3

        // Check original values are preserved
        let padded_matrix = padded.matrix;
        assert_eq!(padded_matrix.get_data()[0], 1);
        assert_eq!(padded_matrix.get_data()[1], 2);
        assert_eq!(padded_matrix.get_data()[2], 3);
        assert_eq!(padded_matrix.get_data()[4], 4);
        assert_eq!(padded_matrix.get_data()[8], 7);

        // Check added values are zeros
        assert_eq!(padded_matrix.get_data()[3], 0);
        assert_eq!(padded_matrix.get_data()[7], 0);
        assert_eq!(padded_matrix.get_data()[15], 0);

        // Check bias values
        let padded_bias = padded.bias.as_ref().unwrap();
        assert_eq!(padded_bias.get_data()[0], 10);
        assert_eq!(padded_bias.get_data()[1], 11);
        assert_eq!(padded_bias.get_data()[2], 12);
        assert_eq!(padded_bias.get_data()[3], 0); // Padding
    }

    #[test]
    fn test_dense_pad_already_power_of_two() {
        // Create a Dense layer with power-of-two dimensions
        let matrix = KeyedTensor::new(
            "dense_weight",
            Tensor::<Element>::matrix_from_coeffs(vec![
                vec![1, 2, 3, 4],
                vec![5, 6, 7, 8],
                vec![9, 10, 11, 12],
                vec![13, 14, 15, 16],
            ])
            .unwrap(),
        );

        let bias = KeyedTensor::new(
            "dense_bias",
            Tensor::<Element>::new(vec![4].into(), vec![20, 21, 22, 23]),
        );

        let dense = Dense::new(matrix, bias);

        // Pad to next power of two
        let padded = dense.clone().pad_next_power_of_two();

        // Check dimensions remain the same
        let padded_dims = padded.matrix.shape();
        assert_eq!(padded_dims[0], 4);
        assert_eq!(padded_dims[1], 4);

        // Check bias dimensions remain the same
        let bias_dims = padded.bias.as_ref().unwrap().shape();
        assert_eq!(bias_dims[0], 4);

        // Check values are preserved
        for i in 0..16 {
            assert_eq!(padded.matrix.get_data()[i], dense.matrix.get_data()[i]);
        }

        for i in 0..4 {
            assert_eq!(
                padded.bias.as_ref().unwrap().get_data()[i],
                dense.bias.as_ref().unwrap().get_data()[i]
            );
        }
    }

    #[test]
    fn test_dense_pad_mixed_dimensions() {
        // Create a Dense layer with one power-of-two dimension and one non-power-of-two
        let matrix = KeyedTensor::new(
            "dense_weight",
            Tensor::<Element>::matrix_from_coeffs(vec![
                vec![1, 2, 3, 4],
                vec![5, 6, 7, 8],
                vec![9, 10, 11, 12],
            ])
            .unwrap(),
        );

        let bias = KeyedTensor::new(
            "dense_bias",
            Tensor::<Element>::new(vec![3].into(), vec![20, 21, 22]),
        );

        let dense = Dense::new(matrix, bias);

        // Pad to next power of two
        let padded = dense.pad_next_power_of_two();

        // Check dimensions are padded correctly
        let padded_dims = padded.matrix.shape();
        assert_eq!(padded_dims[0], 4); // Next power of 2 after 3
        assert_eq!(padded_dims[1], 4); // Already a power of 2

        // Check bias is padded
        let bias_dims = padded.bias.as_ref().unwrap().shape();
        assert_eq!(bias_dims[0], 4); // Next power of 2 after 3

        // Check original values are preserved and padding is zeros
        let padded_matrix = padded.matrix;
        assert_eq!(padded_matrix.get_data()[0], 1);
        assert_eq!(padded_matrix.get_data()[4], 5);
        assert_eq!(padded_matrix.get_data()[8], 9);
        assert_eq!(padded_matrix.get_data()[12], 0); // Padding

        // Check bias values
        let padded_bias = padded.bias.as_ref().unwrap();
        assert_eq!(padded_bias.get_data()[0], 20);
        assert_eq!(padded_bias.get_data()[1], 21);
        assert_eq!(padded_bias.get_data()[2], 22);
        assert_eq!(padded_bias.get_data()[3], 0); // Padding
    }

    #[test]
    fn test_quantization_with_padded_dense() {
        // Create input data that needs quantization
        let input_data = [0.5f32, -0.3f32, 0.1f32];

        // Quantize the input
        let quantized_input: Vec<Element> = input_data
            .iter()
            .map(|x| ScalingFactor::default().quantize(x))
            .collect();

        // Create a Dense layer
        let matrix = KeyedTensor::new(
            "dense_weight",
            Tensor::<Element>::matrix_from_coeffs(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap(),
        );

        let bias = KeyedTensor::new(
            "dense_bias",
            Tensor::<Element>::new(vec![2].into(), vec![10, 11]),
        );

        let dense = Dense::new(matrix, bias);

        // Pad the dense layer
        let padded = dense.clone().pad_next_power_of_two();

        // Create input tensor
        let input_tensor = Tensor::<Element>::new(vec![3].into(), quantized_input);

        // Apply the dense operation on both original and padded
        let output = evaluate_layer::<GoldilocksExt2, _, _>(&dense, &[&input_tensor], None)
            .unwrap()
            .outputs()[0]
            .clone();
        let padded_output =
            evaluate_layer::<GoldilocksExt2, _, _>(&padded, &[&input_tensor.pad_1d(4)], None)
                .unwrap()
                .outputs()[0]
                .clone();

        // Check that the result is correct (for the non-padded parts)
        for i in 0..2 {
            assert_eq!(output.get_data()[i], padded_output.get_data()[i]);
        }
    }

    #[test]
    fn test_dense_proving_with_bias() {
        let [a, b] = [10, 20];
        let first_input_shape = vec![a];

        let mut model =
            Model::new_from_input_shapes(vec![first_input_shape.into()], PaddingMode::NoPadding);

        let dense = Dense::<f32>::random(vec![b, a].into(), None);

        let _ = model
            .add_consecutive_layer(Layer::Dense(dense), None)
            .unwrap();
        model.automatic_output_labelling().unwrap();
        model.describe();
        prove_model(model, &mut GenStore::default()).unwrap();
    }

    proptest! {
        #[test]
        fn test_dense_with_element(input in any_input::<Element>(1..256)) {
            let Input {matrix, bias, input} = input;

            let expected = matrix.matvec(&input).add(&bias);

            let dense = Dense::<Element>::new(matrix.clone(), bias.clone());
            let computed = dense.evaluate::<GoldilocksExt2>(&[&input], &[]).expect("Dense evaluation must be successful");

            prop_assert_eq!(&expected, &computed.outputs[0]);
        }

        #[test]
        fn test_dense_with_f32(input in any_input::<f32>(1usize..256)) {
            let Input {matrix, bias, input} = input;

            let expected = matrix.matvec(&input).add(&bias);

            let dense = Dense::<f32>::new(matrix.clone(), bias.clone());
            let computed = dense.evaluate::<GoldilocksExt2>(&[&input], &[]).expect("Dense evaluation must be successful");

            for (left, right) in expected.get_data().iter().zip(computed.outputs[0].get_data().iter()) {
                prop_assert!(
                    (left - right).abs() < 1e-3,
                    "Actual: {left}, Expected: {right}",
                );
            }
        }
    }

    struct Input<T> {
        matrix: KeyedTensor<T>,
        bias: KeyedTensor<T>,
        input: Tensor<T>,
    }

    impl<T> Debug for Input<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Input").finish_non_exhaustive()
        }
    }

    fn any_input<T: Number>(dim: Range<usize>) -> impl Strategy<Value = Input<T>> {
        dim.prop_flat_map(|dim| {
            let matrix = Tensor::<T>::any(Shape::new(vec![dim, dim]));
            let bias = Tensor::<T>::any(Shape::new(vec![dim]));
            let input = Tensor::<T>::any(Shape::new(vec![dim]));
            (matrix, bias, input).prop_map(|(matrix, bias, input)| Input {
                matrix: KeyedTensor::new("dense_weight", matrix),
                bias: KeyedTensor::new("dense_bias", bias),
                input,
            })
        })
    }
}
