use crate::{
    Claim, Element, Prover, ScalingFactor, ScalingStrategy, Shape, Tensor,
    commit::{compute_betas_eval, identity_eval},
    graph::NodeId,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerProof,
        add::Add,
        provable::{Evaluate, LayerOut, PadOp, QuantizeOutput},
        requant::Requant,
        transformer::positional::{Positional, PositionalCache, PositionalCtx, PositionalProof},
    },
    model::Step,
    quantization::{self, Fieldizer, TensorFielder},
    tensor::{
        CommitmentId, KeyedTensor, TensorSlice, TensorTypeParam, WrappedTensor,
        is_close_with_tolerance,
    },
    util::from_mle_list_dimensions,
};
use anyhow::{Ok, Result, ensure};
use either::Either;
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{mle::IntoMLE, virtual_polys::VirtualPolynomialsBuilder};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    ops::Deref,
    sync::{Arc, Mutex},
};
use sumcheck::{
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
};
use tenstore::GenStore;
use tracing::warn;
use transcript::Transcript;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct RopeProof<E: ExtensionField> {
    // Evaluations of the sub-matrices required to compute the claim
    // about the cosine matrix.
    sub_cosine_evals: Vec<E>,
    // Evaluations of the sub-matrices required to compute the claim
    // about the sine matrix.
    sub_sine_evals: Vec<E>,
    // Proof to link the output with the input
    sumcheck_proof: IOPProof<E>,
    // Evaluations of the polynomials involved in `sumcheck_proof`
    sumcheck_evals: Vec<E>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RopeCtx {
    node_id: NodeId,
    pub(super) unpadded_shape: Shape,
    num_vars_positional_matrix: usize,
    cosine_key: CommitmentId,
    sine_key: CommitmentId,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rope<N> {
    pub(super) cosine_matrix: KeyedTensor<N>,
    pub(super) sine_matrix: KeyedTensor<N>,
    pub(super) unpadded_shape: Shape,
}

impl<N: TensorTypeParam> Rope<N> {
    pub(crate) fn build_from_angles(
        angles: Vec<f32>,
        base_id: CommitmentId,
        max_content_length: usize,
    ) -> Result<Self> {
        // build the rotational vectors
        let matrix_shape = Shape::new(vec![max_content_length, angles.len() * 2]);
        let mut cosine_data = Vec::with_capacity(matrix_shape.numel());
        let mut sine_data = Vec::with_capacity(matrix_shape.numel());
        for i in 0..max_content_length {
            angles.iter().try_for_each(|angle| {
                let cosine = N::from_f32((angle * i as f32).cos())?;
                let sine = N::from_f32((angle * i as f32).sin())?;
                cosine_data.append(&mut vec![cosine; 2]);
                sine_data.append(&mut vec![sine; 2]);
                Ok(())
            })?;
        }
        let cosine_matrix = KeyedTensor::new(
            format!("{}_cosine", base_id),
            Tensor::new(matrix_shape.clone(), cosine_data),
        );
        let sine_matrix = KeyedTensor::new(
            format!("{}_sine", base_id),
            Tensor::new(matrix_shape.clone(), sine_data),
        );
        Ok(Self {
            cosine_matrix,
            sine_matrix,
            unpadded_shape: matrix_shape,
        })
    }

    pub(crate) fn build_from_frequency(
        base_frequency: f32,
        base_frequency_id: CommitmentId,
        head_size: usize,
        max_content_length: usize,
    ) -> Result<Self> {
        let angles = (0..head_size / 2)
            .map(|i| base_frequency.powf((-2.0 * i as f32) / head_size as f32))
            .collect_vec();
        Self::build_from_angles(angles, base_frequency_id, max_content_length)
    }

    pub(crate) fn new(cosine_matrix: KeyedTensor<N>, sine_matrix: KeyedTensor<N>) -> Result<Self> {
        ensure!(
            cosine_matrix.shape() == sine_matrix.shape(),
            "Shapes of provided cosine and sine matrices are different: cosine_shape {:?} vs sine shape {:?}",
            cosine_matrix.shape(),
            sine_matrix.shape(),
        );
        let matrix_shape = cosine_matrix.shape().clone();
        Ok(Self {
            cosine_matrix,
            sine_matrix,
            unpadded_shape: matrix_shape,
        })
    }

    /// Since cosine and since matrix are committed with one variable less, we need to drop a point coordinate in the
    /// claims for such matrices; since the elements being erased from the committed matrices are the ones in even
    /// columns, we remove the coordinate point corresponding to the least significant bit of the column index,
    /// which is the first coordinate in the point
    fn claim_for_committed_matrix<E: ExtensionField>(claim: &mut Claim<E>) {
        claim.point.remove(0);
    }

    pub(super) fn prove_step<
        E: ExtensionField,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    >(
        &self,
        node_id: NodeId,
        output_claim: &Claim<E>,
        step_data: &Step<E, Element, E>,
        prover: &mut Prover<E, T, PCS>,
        store: &mut GenStore,
    ) -> anyhow::Result<Vec<Claim<E>>>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
        N: Fieldizer<E>,
    {
        let input = &step_data.node_inputs[0];
        let input = input.hydrate(store.clone())?;
        let input_shape = input.shape().clone();
        let cosine_matrix_slice = TensorSlice::from(self.cosine_matrix.deref());
        let sine_matrix_slice = TensorSlice::from(self.sine_matrix.deref());
        let sub_cos_matrix = cosine_matrix_slice
            .slice_over_first_dim(0, input.shape()[0])
            .to_fields();
        let sub_sin_matrix = sine_matrix_slice
            .slice_over_first_dim(0, input.shape()[0])
            .to_fields();

        // build sum-check to compute the output as `input*sub_cos_matrix + permuted_input*sub_sin_matrix`
        // The permuted input is the input tensor permuted as follows:
        // given an input tensor `x=[x_1,x_2,x_3,x_4,...,x_n-1,x_n]`,
        // the permuted input tensor is `x' = [-x_2,x_1,-x_4,x_3,...,-x_n,x_n-1]`
        // The input MLE is structured, for variables y_1, y_2, ..., y_m, as
        // x_1*eq(0,y_2..)(1-y_1) + x_2*eq(1,y_2..)y_1 + x_3*eq(2, y_2..)(1-y_1) + x_4*eq(3, y_2..)y_1  + ...
        // The permuted input MLE is structured, for variables y_1, y_2, ..., y_m, as
        // -x_2*eq(0,y_2..)(1-y_1) + x_1*eq(1,y_2..)y_1 - x_4*eq(2, y_2..)(1-y_1) + x_3*eq(3, y_2..)y_1 + ...
        // Therefore, given a claim `input_claim` for the input MLE, it holds that:
        // `input_claim.eval = \sum_{i < n} eq_poly(i) * input_mle(i)`, where `eq_poly(i) = \beta(input_claim.point, i)`
        // Similarly, for a claim `permuted_input_claim` for the permuted input MLE, it holds that:
        // `permuted_input_claim.eval = \sum_{i < n} permuted_eq_poly(i) * input_mle(i)`,
        // where `permuted_eq_poly(i) = \beta(permuted_input_claim.point, i + 1) if i is even, else -\beta(permuted_input_claim.point, i - 1)`.
        // Indeed, if `permuted_input_claim.point = [r_1, r_2, ..., r_m]`, then:
        // - `permuted_eq_poly(0) = eq(1,r_2..)r_1`, which is exactly the term related to `x_1` in the permuted input MLE
        // - `permuted_eq_poly(1) = -eq(0,r_2..)(1-r_1)`, which is exactly the term related to `x_2` in the permuted input MLE
        // and so on for all other n-2 elements of input polynomial.
        // In conclusion, we can prove the correctness of the output computation by running a sumcheck over
        // the relationship `output = eq_poly * input * sub_cos_matrix + permuted_eq_poly * input * sub_sin_matrix`

        // compute MLEs of the tensors involved in the sum-check
        let input_mle = input.into_mle();
        let sub_cos_mle = sub_cos_matrix.into_mle();
        let sub_sin_mle = sub_sin_matrix.into_mle();
        let num_vars = input_mle.num_vars();
        ensure!(
            output_claim.point.len() == num_vars,
            "Number of variables of input tensor ({num_vars}) different from number of variables of output claim point ({})",
            output_claim.point.len(),
        );
        // compute evals of EQ poly for the output claim
        let beta_vec = compute_betas_eval(&output_claim.point);
        // compute `permuted_eq_poly`: by definition, given the evals of `eq_poly` (i.e., `beta_vec`),
        // it is constructed as `permuted_eq_vec[i] = beta_vec[i+1] if i is even, -beta_vec[i-1] if i is odd`
        let permuted_eq_mle = beta_vec
            .chunks(2)
            .flat_map(|chunk| vec![chunk[1], E::ZERO - chunk[0]])
            .collect_vec()
            .into_mle();
        let eq_mle = beta_vec.into_mle();

        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<E>::new(num_threads, num_vars);
        let input_expr = expr_builder.lift(Either::Left(&input_mle));
        let sub_cos_expr = expr_builder.lift(Either::Left(&sub_cos_mle));
        let sub_sin_expr = expr_builder.lift(Either::Left(&sub_sin_mle));
        let eq_expr = expr_builder.lift(Either::Left(&eq_mle));
        let permuted_eq_expr = expr_builder.lift(Either::Left(&permuted_eq_mle));

        let virtual_poly = expr_builder.to_virtual_polys(
            &[input_expr * (eq_expr * sub_cos_expr + permuted_eq_expr * sub_sin_expr)],
            &[],
        );
        let (sumcheck_proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);

        let sumcheck_point = state.collect_raw_challenges();
        let sumcheck_evals = state.get_mle_flatten_final_evaluations()[..3].to_vec();
        let input_eval = sumcheck_evals[0];
        let input_claim = Claim::new(sumcheck_point.clone(), input_eval);

        let sub_cos_claim = Claim::new(sumcheck_point.clone(), sumcheck_evals[1]);
        let sub_sin_claim = Claim::new(sumcheck_point.clone(), sumcheck_evals[2]);

        let (sub_cos_evals, mut cosine_claim) =
            Positional::<N>::bind_sub_claim_to_positional_matrix(
                sub_cos_claim,
                output_claim,
                &cosine_matrix_slice,
                &self.cosine_matrix,
                input_shape[0],
                prover.transcript,
            )?;
        Self::claim_for_committed_matrix(&mut cosine_claim);

        let (sub_sin_evals, mut sine_claim) = Positional::<N>::bind_sub_claim_to_positional_matrix(
            sub_sin_claim,
            output_claim,
            &sine_matrix_slice,
            &self.sine_matrix,
            input_shape[0],
            prover.transcript,
        )?;

        Self::claim_for_committed_matrix(&mut sine_claim);

        let proof = RopeProof {
            sub_cosine_evals: sub_cos_evals,
            sub_sine_evals: sub_sin_evals,
            sumcheck_proof,
            sumcheck_evals,
        };

        let commons_claims = [
            (self.cosine_matrix.commitment_id(), cosine_claim),
            (self.sine_matrix.commitment_id(), sine_claim),
        ]
        .into_iter()
        .collect();

        prover.add_common_claims(node_id, commons_claims);

        prover.push_proof(
            node_id,
            LayerProof::Positional(PositionalProof::Rope(proof)),
        );

        Ok(vec![input_claim])
    }
}

impl<N> Rope<N> {
    pub(super) fn evaluate<E: ExtensionField>(
        &self,
        input: &WrappedTensor<N>,
        positional_cache: &Arc<Mutex<PositionalCache>>,
    ) -> Result<LayerOut<N, E>>
    where
        N: TensorTypeParam,
        Add<N>: Evaluate<N>,
    {
        // Old vs new implementation
        // Let input be: x = [x_0, x_1, x_2, x_3, ..., x_{2k}, x_{2k+1}, ..., x_{d-2}, x_{d-1}] with even d.
        // Let cosine and sine generated by build_from_angles() be:
        //   c = [c_0, c_0, c_1, c_1, ..., c_k, c_k, ..., c_{d/2-1}, c_{d/2-1}]
        //   s = [s_0, s_0, s_1, s_1, ..., s_k, s_k, ..., s_{d/2-1}, s_{d/2-1}]
        //
        // OLD:
        //   n(x) = [-x_1, x_0, -x_3, x_2, ..., -x_{2k+1}, x_{2k}, ..., -x_{d-1}, x_{d-2}].
        // Element-wise output:
        //   y = x * c + n(x) * s
        // Expanding a single pair with indices (2k, 2k+1):
        //   y_{2k}   =  x_{2k} * c_k + (-x_{2k+1}) * s_k =  x_{2k} c_k - x_{2k+1} s_k
        //   y_{2k+1} =  x_{2k+1} * c_k + ( x_{2k})   * s_k =  x_{2k+1} c_k + x_{2k} s_k
        //
        // NEW:
        // Instead of constructing n(x), reshape x, c, s to [rows, d/2, 2] so that each last-dimension pair
        // corresponds to one rotational block. From the reshaped tensors we define:
        //   x1 = x[:,:,0]  (even positions  2k)
        //   x2 = x[:,:,1]  (odd  positions  2k+1)
        //   c1 = c[:,:,0]; c2 = c[:,:,1]
        //   s1 = s[:,:,0]; s2 = s[:,:,1]
        // Note: c1 = c2 and s1 = s2 because of the way the cosine and sine matrices are built
        // Compute the two rotation components per pair:
        //   out_even = x1 * c1 - x2 * s1        (gives y_{2k})
        //   out_odd  = x2 * c2 + x1 * s2        (gives y_{2k+1})
        // Because of duplication (c1==c2, s1==s2) these match the expanded formulas above.
        // Finally concatenate out_even and out_odd along pair axis to reconstruct y.
        let past_length = positional_cache.lock().unwrap().seq_len;
        let cosine_matrix_bt = WrappedTensor::try_from(&self.cosine_matrix)?;
        let sine_matrix_bt = WrappedTensor::try_from(&self.sine_matrix)?;
        let cosine_slice_bt = cosine_matrix_bt.slice([
            past_length..(past_length + input.shape().dims[0]),
            0..input.shape().dims[1],
        ]);
        let sine_slice_bt = sine_matrix_bt.slice([
            past_length..(past_length + input.shape().dims[0]),
            0..input.shape().dims[1],
        ]);

        positional_cache
            .lock()
            .unwrap()
            .set_seq_len(past_length + input.unpadded_shape().dims[0])?;

        ensure!(
            cosine_slice_bt.shape() == input.shape(),
            "Incompatible shapes in Rope evaluation between rotational matrices ({:?}) and input ({:?})",
            cosine_slice_bt.shape(),
            input.shape(),
        );

        let rows = input.shape().dims[0];
        let half = input.shape().dims[1] / 2;

        let inputs = input.clone().reshape([rows, half, 2].into())?;
        let cosines = cosine_slice_bt.reshape([rows, half, 2].into())?;
        let sines = sine_slice_bt.reshape([rows, half, 2].into())?;

        // x1 = [:,:,0], x2 = [:,:,1], and same for c1,c2,s1,s2
        let x1 = inputs
            .clone()
            .slice([0..rows, 0..half, 0..1])
            .reshape([rows, half].into())?;
        let x2 = inputs
            .slice([0..rows, 0..half, 1..2])
            .reshape([rows, half].into())?;
        let c1 = cosines
            .clone()
            .slice([0..rows, 0..half, 0..1])
            .reshape([rows, half].into())?;
        let c2 = cosines
            .slice([0..rows, 0..half, 1..2])
            .reshape([rows, half].into())?;
        let s1 = sines
            .clone()
            .slice([0..rows, 0..half, 0..1])
            .reshape([rows, half].into())?;
        let s2 = sines
            .slice([0..rows, 0..half, 1..2])
            .reshape([rows, half].into())?;

        // out_even = x1*c1 + (-x2)*s1   (rotation real part)
        let neg_x2 = x2.clone().neg();
        let out_even = x1.clone().mul(c1)?.add(neg_x2.mul(s1)?)?;
        // out_odd  = x2*c2 + x1*s2     (rotation imag part)
        let out_odd = x2.mul(c2)?.add(x1.mul(s2)?)?;

        let out_even_e = out_even.reshape([rows, half, 1].into())?;
        let out_odd_e = out_odd.reshape([rows, half, 1].into())?;
        let output =
            WrappedTensor::cat(vec![out_even_e, out_odd_e], 2)?.reshape([rows, half * 2].into())?;
        Ok(LayerOut::from_tensor(output))
    }
}

impl Rope<f32> {
    pub(super) fn quantize<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: ScalingFactor,
    ) -> anyhow::Result<QuantizeOutput<Rope<Element>>> {
        // compute scaling factor for cosine and sine matrices
        let max_cos = self.cosine_matrix.max_value();
        let max_sin = self.sine_matrix.max_value();
        // check that the maximum values are close
        if !is_close_with_tolerance(&[max_cos], &[max_sin], 1e-3, 0.0) {
            warn!(
                "Maximum values are too distant for cosine/sine matrices in positional rope with id {node_id}"
            );
        }
        let min_cos = self.cosine_matrix.min_value();
        let min_sin = self.sine_matrix.min_value();
        // check that the minimum values are close
        if !is_close_with_tolerance(&[min_cos], &[min_sin], 1e-3, 0.0) {
            warn!(
                "Minimum values are too distant for cosine/sine matrices in positional rope with id {node_id}"
            );
        }
        // compute the scaling factor for both matrices
        let matrix_scale =
            ScalingFactor::from_span(min_cos.min(min_sin), max_cos.max(max_sin), None);

        let output_scalings = S::scaling_factors_for_node(data, node_id, 1);
        ensure!(
            output_scalings.len() == 1,
            "Expected 1 output scaling factor for Positional layer {node_id}, found {}",
            output_scalings.len(),
        );
        let output_scaling = &output_scalings[0];
        // in the layer, given a row of input tensor `x``, a row of cosine matrix `c` and a row of sine matrix `s``,
        // we compute the corresponding row in the output tensor `out` as:
        // out = x*c + pi(x)*s, where `pi(x)` is a permutation of input vector x.
        // We are using the same scaling factor `matrix_scale` for `c` and `s`, and `pi(x)` has the same
        // scaling factor `input_scaling` of `x`, as it is simply a permutation of `x`.
        // So, both the element-wise multiplications in the above equation are between an item with
        // scaling factor `input_scaling` and an item with scaling factor `matrix_scale`.
        // Therefore, we can use the multiplier `input_scaling*matrix_scale/output_scaling` to requantize
        let multiplier = input_scaling.m(&matrix_scale, output_scaling);
        let output_bit_size = 2 * *quantization::BIT_LEN + 1; // +1 because we are adding 2 products of items with `quantization::BIT_LEN` bits
        let requant = Requant::from_multiplier(multiplier, output_bit_size);

        let quantized_rope = Rope {
            cosine_matrix: self.cosine_matrix.quantize(&matrix_scale),
            sine_matrix: self.sine_matrix.quantize(&matrix_scale),
            unpadded_shape: self.unpadded_shape,
        };

        Ok(QuantizeOutput::new(quantized_rope, output_scalings).with_requant(requant))
    }
}

impl PadOp for Rope<Element> {
    fn pad_node(mut self, _si: &mut crate::padding::ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        self.cosine_matrix = self.cosine_matrix.map_tensor(|t| t.pad_next_power_of_two());
        self.sine_matrix = self.sine_matrix.map_tensor(|t| t.pad_next_power_of_two());

        Ok(self)
    }
}

impl Rope<Element> {
    pub(super) fn step_info(
        &self,
        id: NodeId,
        mut aux: ContextAux,
    ) -> anyhow::Result<(RopeCtx, ContextAux)> {
        // this closure retains only values in odd columns, relying on the fact that `self.cosine_matrix`
        // and `self.sine_matrix` have pairwise identical elements in each column. In this way, the size
        // of the polynomial being committed is halved
        let matrix_to_evals = |matrix: &Tensor<Element>| {
            matrix
                .get_data()
                .chunks(2)
                .map(|chunk| chunk[0])
                .collect_vec()
        };

        aux.model_polys = Some(
            [
                (
                    self.cosine_matrix.commitment_id(),
                    matrix_to_evals(&self.cosine_matrix),
                ),
                (
                    self.sine_matrix.commitment_id(),
                    matrix_to_evals(&self.sine_matrix),
                ),
            ]
            .into_iter()
            .collect(),
        );
        let num_vars = self.cosine_matrix.shape().num_vars().into_iter().sum();
        let ctx = RopeCtx {
            unpadded_shape: self.unpadded_shape.clone(),
            node_id: id,
            num_vars_positional_matrix: num_vars,
            cosine_key: self.cosine_matrix.commitment_id(),
            sine_key: self.sine_matrix.commitment_id(),
        };

        Ok((ctx, aux))
    }
}

impl RopeCtx {
    pub(super) fn verify<
        E: ExtensionField,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        &self,
        proof: &RopeProof<E>,
        last_claim: &Claim<E>,
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        let input_num_vars = shape_step.padded_input_shape[0]
            .num_vars()
            .into_iter()
            .sum();
        let aux_info = from_mle_list_dimensions(&[
            vec![input_num_vars, input_num_vars, input_num_vars],
            vec![input_num_vars, input_num_vars, input_num_vars],
        ]);
        let subclaim = IOPVerifierState::verify(
            last_claim.eval,
            &proof.sumcheck_proof,
            &aux_info,
            verifier.transcript,
        );
        let sumcheck_point = subclaim
            .point
            .iter()
            .map(|p| p.elements)
            .collect::<Vec<_>>();

        let beta_eval = identity_eval(&last_claim.point, &sumcheck_point);
        let permuted_eq_eval = evaluate_permutation_eq_poly(&last_claim.point, &sumcheck_point);

        // verify sum-check evaluation
        ensure!(
            proof.sumcheck_evals.len() == 3,
            "expected 3 evaluations for sumcheck in Positional::Rope proof, found {}",
            proof.sumcheck_evals.len(),
        );
        let input_eval = proof.sumcheck_evals[0];
        let cosine_eval = proof.sumcheck_evals[1];
        let sine_eval = proof.sumcheck_evals[2];
        ensure!(
            subclaim.expected_evaluation
                == input_eval * (beta_eval * cosine_eval + permuted_eq_eval * sine_eval),
            "Sumcheck verification failed for Positional::Rope verifier"
        );

        let input_claim = Claim::new(sumcheck_point.clone(), input_eval);
        let sub_cos_claim = Claim::new(sumcheck_point.clone(), cosine_eval);
        let sub_sine_claim = Claim::new(sumcheck_point, sine_eval);

        // // verify permutation proof
        // let num_col_vars = shape_step.padded_input_shape[0].num_vars_2d().1;
        // let aux_info = from_mle_list_dimensions(&[vec![num_col_vars, num_col_vars]]);
        // let subclaim = IOPVerifierState::verify(
        // permuted_input_claim.eval,
        // &proof.permute_proof,
        // &aux_info,
        // verifier.transcript,
        // );
        //
        // let sumcheck_point = subclaim
        // .point
        // .iter()
        // .map(|p| p.elements)
        // .collect::<Vec<_>>();
        // let (point_for_input, point_for_permute) = Rope::<Element>::full_points(
        // &permuted_input_claim,
        // &sumcheck_point,
        // shape_step.padded_input_shape[0].clone(),
        // );
        // let additional_input_claim = Claim::new(point_for_input, proof.input_eval);
        // c compute permutation matrix evaluation and verify sum-check evaluation
        // let permutation_matrix_eval =
        // evaluate_permutation_matrix_poly(&point_for_permute, num_col_vars)?;
        //
        // ensure!(
        // subclaim.expected_evaluation == proof.input_eval * permutation_matrix_eval,
        // "Permutation sumcheck verification failed for Positional::Rope layer"
        // );
        //
        // let same_poly_ctx = same_poly::Context::new(input_num_vars);
        // let mut same_poly_verifier = same_poly::Verifier::new(&same_poly_ctx);
        //
        // same_poly_verifier.add_claim(input_claim)?;
        // same_poly_verifier.add_claim(additional_input_claim)?;
        //
        // let final_input_claim =
        // same_poly_verifier.verify(&proof.aggregation_proof, verifier.transcript)?;

        let mut cosine_matrix_claim = PositionalCtx::build_positional_matrix_claim(
            sub_cos_claim,
            last_claim,
            self.num_vars_positional_matrix,
            verifier.transcript,
            &proof.sub_cosine_evals,
        )?;

        Rope::<Element>::claim_for_committed_matrix(&mut cosine_matrix_claim);

        let mut sine_matrix_claim = PositionalCtx::build_positional_matrix_claim(
            sub_sine_claim,
            last_claim,
            self.num_vars_positional_matrix,
            verifier.transcript,
            &proof.sub_sine_evals,
        )?;

        Rope::<Element>::claim_for_committed_matrix(&mut sine_matrix_claim);

        verifier.add_common_claims(
            self.node_id,
            [
                (self.cosine_key.clone(), cosine_matrix_claim),
                (self.sine_key.clone(), sine_matrix_claim),
            ]
            .into_iter()
            .collect(),
        );

        Ok(vec![input_claim])
    }
}

// compute evaluation of the `permute_eq_poly` employed in proving.
// For a claim point `r_c`, the `permute_eq_poly` is constructed over
// `n`` variables as:
// - `permute_eq_poly[i] = \beta(r_c, i+1) if i is even`
// - `permute_eq_poly[i] = -\beta(r_c, i-1) if i is odd`
// This method evalautes `permuted_eq_poly(input)` given an input point
// with `n` coordinates and the claim point `r_c` employed to construct
// `permuted_eq_poly`
fn evaluate_permutation_eq_poly<E: ExtensionField>(claim_point: &[E], input: &[E]) -> E {
    // Given the coordinates `r_1, ..., r_n` of `claim_point` and the `n`
    // input variables `y_1,...,y_n`, we have:
    // - `permute_eq_poly[i]` = eq(i,r_2..)*r_1*(1-y_1) when i is even
    // - `permute_eq_poly[i]` = -eq(i,r_2..)*(1-r_1)*y_1 when i is odd
    // Since all these terms are summed in the MLE, we obtain:
    // eq(i,r_2..)[r_1*(1-y_1) - y_1*(1-r_1)] = eq(i,r_2..)[r_1 - y_1]
    // so, we split the evaluation in 2 terms:
    // - the first term `eq(i,r_2..)`, which involves all the variables
    //  besides the first one, and it is the same as computing the EQ
    //  polynomial for these variables
    // - the term related to the first variable, which is what differentiates
    //  the `permuted_eq_poly` from a generic EQ polynomial
    let eq_eval = identity_eval(&claim_point[1..], &input[1..]);
    eq_eval * (claim_point[0] - input[0])
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use ark_std::rand::Rng;
    use itertools::Itertools;

    use rstest::rstest;

    use tenstore::GenStore;

    use crate::{
        Element, ScalingFactor, Tensor,
        layers::{
            Layer,
            provable::PadOp,
            transformer::positional::{Positional, PositionalCache, rope::Rope},
        },
        model::{Model, test::prove_model},
        padding::{PaddingMode, ShapeData, ShapeInfo},
        quantization::AbsoluteMax,
        rng_from_env_or_random,
        tensor::{TensorTypeParam, is_close_with_tolerance},
    };

    use ff_ext::GoldilocksExt2;
    use proptest::prelude::*;
    use std::sync::{Arc, Mutex};

    fn random_angles(num_angles: usize) -> Vec<f32> {
        let mut rng = rng_from_env_or_random();
        (0..num_angles)
            .map(|_| rng.gen_range(0.0..2.0 * PI))
            .collect_vec()
    }

    #[test]
    fn test_cosine_and_sine_bounds() {
        let mut rng = rng_from_env_or_random();
        for _ in 0..20 {
            let num_angles = rng.gen_range(16..768);
            let angles = random_angles(num_angles);
            let max_context_length = rng.gen_range(2..1024);
            println!("Testing for {num_angles} angles and context length {max_context_length}");
            let rope = Rope::<f32>::build_from_angles(
                angles,
                "rope_angles".to_string().into(),
                max_context_length,
            )
            .unwrap();
            println!(
                "Max cos: {}, Max sin: {}",
                rope.cosine_matrix.max_value(),
                rope.sine_matrix.max_value(),
            );
            println!(
                "Min cos: {}, Min sin: {}",
                rope.cosine_matrix.min_value(),
                rope.sine_matrix.min_value(),
            );
        }
    }

    #[rstest]
    #[case::less_input_than_context_length(14, 18, 31)]
    #[case::same_input_as_context_length(31, 18, 31)]
    fn test_proven_rope_positional_layer(
        #[case] seq_len: usize,
        #[case] embedding_size: usize,
        #[case] context_length: usize,
    ) {
        let input_shape = vec![seq_len, embedding_size];

        let mut model =
            Model::new_from_input_shapes(vec![input_shape.into()], PaddingMode::NoPadding);

        // build angles for rotational matrix
        assert!(embedding_size.is_multiple_of(2));
        let angles = random_angles(embedding_size / 2);

        let _ = model
            .add_consecutive_layer(
                Layer::Positional(
                    Positional::new_rope(angles, "rope_angles".to_string().into(), context_length)
                        .unwrap(),
                ),
                None,
            )
            .unwrap();

        model.automatic_output_labelling().unwrap();

        let _ = prove_model(model, &mut GenStore::default()).unwrap();
    }

    #[derive(Clone)]
    struct Input<T> {
        seq_len: usize,
        embedding_size: usize,
        context_length: usize,
        input: Tensor<T>,
        angles: Vec<f32>,
    }

    impl<T: core::fmt::Debug> core::fmt::Debug for Input<T> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("Input")
                .field("seq_len", &self.seq_len)
                .field("embedding_size", &self.embedding_size)
                .field("context_length", &self.context_length)
                .finish_non_exhaustive()
        }
    }

    fn rope_input<T: TensorTypeParam>() -> impl Strategy<Value = Input<T>> {
        (1..32usize, 1..32usize).prop_flat_map(|(seq_len, half_embed)| {
            let embedding_size = half_embed * 2;
            (
                seq_len..=64usize,
                Tensor::<T>::any(vec![seq_len, embedding_size].into()),
                proptest::collection::vec(0.0..2.0 * PI, embedding_size / 2),
            )
                .prop_map(move |(context_length, input, angles)| Input {
                    seq_len,
                    embedding_size,
                    context_length,
                    input,
                    angles,
                })
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn test_rope_f32(inp in rope_input::<f32>()) {
            let Input { seq_len, embedding_size, context_length, input, angles } = inp.clone();
            prop_assume!(embedding_size % 2 == 0 && embedding_size >= 2);
            let layer = Rope::<f32>::build_from_angles(angles.clone(), "rope_angles".to_string().into(), context_length).expect("build rope");
            let cache = Arc::new(Mutex::new(PositionalCache::new()));

            let out = layer
                .evaluate::<GoldilocksExt2>(&input.as_wrapped(), &cache)
                .expect("rope evaluate").outputs.into_iter().next().unwrap();

            let mut expected = Vec::with_capacity(seq_len * embedding_size);
            let in_data = input.data();
            for row in 0..seq_len {
                for k in 0..(embedding_size/2) {
                    let angle = angles[k] * row as f32;
                    let cosv = angle.cos();
                    let sinv = angle.sin();
                    let x1 = in_data[row * embedding_size + 2*k];
                    let x2 = in_data[row * embedding_size + 2*k + 1];
                    expected.push(x1 * cosv - x2 * sinv);
                    expected.push(x2 * cosv + x1 * sinv);
                }
            }
            let expected_t = Tensor::new(vec![seq_len, embedding_size].into(), expected);
            let out = is_close_with_tolerance(&out.get_data(), expected_t.data(), 1e-5, 1e-4);
            prop_assert!(out);
        }

        #[test]
        fn test_rope_quantized(inp in rope_input::<f32>()) {
            let Input { seq_len, embedding_size, context_length, input, angles } = inp.clone();
            prop_assume!(embedding_size % 2 == 0 && embedding_size >= 2);

            let layer = Rope::<f32>::build_from_angles(angles.clone(), "rope_angles".to_string().into(), context_length).expect("build rope");
            let input_sf = ScalingFactor::from_tensor(&input, None);
            let q = layer.quantize::<AbsoluteMax>(&(), 0.into(), input_sf).expect("quantize rope");
            let layer_q = q.quantized_op;
            let input_q = input.to_quantized(&input_sf);
            let cache = Arc::new(Mutex::new(PositionalCache::new()));

            let out_q = layer_q
                .evaluate::<GoldilocksExt2>(&input_q.as_wrapped(), &cache)
                .expect("rope evaluate quant").outputs.into_iter().next().unwrap();

            prop_assert_eq!(out_q.shape(), vec![seq_len, embedding_size].into());
        }

        #[test]
        fn test_rope_padding_prop(inp in rope_input::<Element>()) {
            let Input { seq_len, embedding_size, context_length, input: _, angles } = inp.clone();
            prop_assume!(embedding_size % 2 == 0 && embedding_size >= 2);

            let layer = Rope::<Element>::build_from_angles(angles.clone(), "rope_angles".to_string().into(), context_length).expect("build rope element");
            let mut si = ShapeInfo::from(vec![ShapeData::new(vec![seq_len, embedding_size].into())].as_slice());
            let padded = PadOp::pad_node(layer, &mut si).expect("pad rope");
            let padded_shape = padded.cosine_matrix.shape();
            prop_assert_eq!(&padded.unpadded_shape, &vec![context_length, embedding_size].into());
            prop_assert_eq!(padded_shape, &padded.unpadded_shape.next_power_of_two());

            for i in 0..padded.unpadded_shape[0] {
                for j in 0..padded.unpadded_shape[1] {
                    if j % 2 == 0 {
                        prop_assert_eq!(padded.cosine_matrix.get_2d(i,j), padded.cosine_matrix.get_2d(i,j+1));
                        prop_assert_eq!(padded.sine_matrix.get_2d(i,j), padded.sine_matrix.get_2d(i,j+1));
                    }
                }
            }
        }

        #[test]
        fn test_rope_proving_prop(inp in rope_input::<f32>()) {
            let Input { seq_len, embedding_size, context_length, input: _, angles } = inp.clone();
            prop_assume!(embedding_size % 2 == 0 && embedding_size >= 2);
            prop_assume!(seq_len >= 2);

            let input_shape = vec![seq_len, embedding_size];
            let mut model = Model::new_from_input_shapes(vec![input_shape.into()], PaddingMode::NoPadding);

            model.add_consecutive_layer(Layer::Positional(Positional::new_rope(angles, "rope_angles".to_string().into(), context_length).expect("rope")), None).expect("rope layer");
            model.automatic_output_labelling().expect("route output");
            let _ = prove_model(model, &mut GenStore::default()).expect("prove model");
        }
    }
}
