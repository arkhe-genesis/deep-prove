use std::{
    ops::Sub,
    sync::{Arc, Mutex},
};

use anyhow::{Ok, Result, ensure};
use either::Either;
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{mle::IntoMLE, virtual_polys::VirtualPolynomialsBuilder};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::{
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
};
use tenstore::GenStore;
use tracing::warn;
use transcript::Transcript;

use crate::{
    Claim, Element, Prover, ScalingFactor, ScalingStrategy, Shape, Tensor,
    commit::{compute_betas_eval, identity_eval},
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerProof,
        add::Add,
        provable::{Evaluate, LayerOut, NodeId, PadOp, QuantizeOutput},
        requant::Requant,
        transformer::positional::{Positional, PositionalCache, PositionalCtx, PositionalProof},
    },
    model::StepData,
    number::Number,
    quantization::{self, Fieldizer, TensorFielder},
    tensor::{TensorSlice, is_close_with_tolerance},
    util::from_mle_list_dimensions,
};

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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rope<N> {
    pub(super) cosine_matrix: Tensor<N>,
    pub(super) sine_matrix: Tensor<N>,
    pub(super) unpadded_shape: Shape,
}

const COSINE_POLY_ID: &str = "CosineMatrix";
const SINE_POLY_ID: &str = "SineMatrix";

/// This method computes the permuted input tensor needed for Rope computation.
/// The input is permuted as follows: given an input tensor `x=[x_1,x_2,x_3,x_4,...,x_n-1,x_n]`,
/// the permuted input tensor is `x' = [-x_2,x_1,-x_4,x_3,...,-x_n,x_n-1]`
fn permuted_input<T: Sub<Output = T> + Copy + Default>(input: &Tensor<T>) -> Tensor<T> {
    let zero = T::default() - T::default();
    Tensor::new(
        input.shape().clone(),
        input
            .data()
            .chunks(2)
            .flat_map(|chunk| vec![zero - chunk[1], chunk[0]])
            .collect_vec(),
    )
}

impl<N: Number> Rope<N> {
    pub(crate) fn build_from_angles(angles: Vec<f32>, max_content_length: usize) -> Result<Self> {
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
        let cosine_matrix = Tensor::new(matrix_shape.clone(), cosine_data);
        let sine_matrix = Tensor::new(matrix_shape.clone(), sine_data);
        Ok(Self {
            cosine_matrix,
            sine_matrix,
            unpadded_shape: matrix_shape,
        })
    }

    pub(crate) fn build_from_frequency(
        base_frequency: f32,
        head_size: usize,
        max_content_length: usize,
    ) -> Result<Self> {
        // println!(
        //    "ROPE: from _frequency: base_frequency: {}, head_size: {}, max_content_length: {}",
        //    base_frequency, head_size, max_content_length
        //);
        let angles = (0..head_size / 2)
            .map(|i| base_frequency.powf((-2.0 * i as f32) / head_size as f32))
            .collect_vec();
        Self::build_from_angles(angles, max_content_length)
    }

    pub(crate) fn new(cosine_matrix: Tensor<N>, sine_matrix: Tensor<N>) -> Result<Self> {
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

    pub(super) fn evaluate<E: ExtensionField>(
        &self,
        input: &Tensor<N>,
        unpadded_input_shape: &Shape,
        positional_cache: &Arc<Mutex<PositionalCache>>,
    ) -> Result<LayerOut<N, E>>
    where
        Add<N>: Evaluate<N>,
    {
        let past_length = positional_cache.lock().unwrap().seq_len;
        let cosine_slice = self
            .cosine_matrix
            .slice_2d(past_length, past_length + input.shape()[0]);
        let sine_slice = self
            .sine_matrix
            .slice_2d(past_length, past_length + input.shape()[0]);
        // The positional cache needs to store the number of tokens processed so far.
        // The number of tokens processed in this round of `evaluate` corresponds to
        // `unpadded_input_shape[0]` rather than `input.shape()[0]`, as `input` could be
        // padded to the next power of 2. The actual number of tokens processed is required
        // since this defines the starting row of the slices of cosine and sine matrices
        // in the next iteration
        positional_cache
            .lock()
            .unwrap()
            .set_seq_len(past_length + unpadded_input_shape[0])?;
        ensure!(
            cosine_slice.shape() == input.shape(),
            "Incompatible shapes in Rope evaluation between rotational matrices ({:?}) and input ({:?})",
            cosine_slice.shape(),
            input.shape(),
        );
        // the output is computed as `input*cosine_slice + permuted_input*sine_slice`, where * is the
        // entry-wise tensor product
        let cosine_input = input.mul(&cosine_slice);
        let permuted_input = permuted_input(input);
        let sine_input = permuted_input.mul(&sine_slice);
        let output = cosine_input.add(&sine_input);
        Ok(LayerOut::from_vec(vec![output]))
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
        step_data: &StepData<E, E>,
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
        let cosine_matrix_slice = TensorSlice::from(&self.cosine_matrix);
        let sine_matrix_slice = TensorSlice::from(&self.sine_matrix);
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
            (COSINE_POLY_ID.to_string(), cosine_claim),
            (SINE_POLY_ID.to_string(), sine_claim),
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
            cosine_matrix: self.cosine_matrix.to_quantized(&matrix_scale),
            sine_matrix: self.sine_matrix.to_quantized(&matrix_scale),
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
        self.cosine_matrix = self.cosine_matrix.pad_next_power_of_two();
        self.sine_matrix = self.sine_matrix.pad_next_power_of_two();

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
                    COSINE_POLY_ID.to_string(),
                    matrix_to_evals(&self.cosine_matrix),
                ),
                (SINE_POLY_ID.to_string(), matrix_to_evals(&self.sine_matrix)),
            ]
            .into_iter()
            .collect(),
        );
        let num_vars = self.cosine_matrix.shape().num_vars().into_iter().sum();
        let ctx = RopeCtx {
            unpadded_shape: self.unpadded_shape.clone(),
            node_id: id,
            num_vars_positional_matrix: num_vars,
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
                (COSINE_POLY_ID.to_string(), cosine_matrix_claim),
                (SINE_POLY_ID.to_string(), sine_matrix_claim),
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

    use crate::{layers::transformer::positional::rope::Rope, rng_from_env_or_random};

    use rstest::rstest;

    use tenstore::GenStore;

    use crate::{
        layers::{Layer, transformer::positional::Positional},
        model::{Model, test::prove_model},
        padding::PaddingMode,
    };

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
            let rope = Rope::<f32>::build_from_angles(angles, max_context_length).unwrap();
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
                Layer::Positional(Positional::new_rope(angles, context_length).unwrap()),
                None,
            )
            .unwrap();

        model.route_output(None).unwrap();

        let _ = prove_model(model, &mut GenStore::default()).unwrap();
    }
}
