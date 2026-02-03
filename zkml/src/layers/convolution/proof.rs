use super::{
    Convolution, IS_PROVABLE, conv2d_shape, new_clearing_tensor, padded_conv2d_shape, to_bits,
};
use crate::{
    Claim, Element, Shape, VectorTranscript,
    commit::identity_eval,
    fft::get_root_of_unity,
    graph::NodeId,
    iop::{context::ShapeStep, verifier::Verifier},
    layers::{
        hadamard,
        provable::{OpInfo, VerifiableCtx},
    },
    padding::PaddingMode,
    tensor::CommitmentId,
    util::from_mle_list_dimensions,
};
use anyhow::{Context, Result, ensure};
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{mle::IntoMLE, virtual_poly::VPAuxInfo};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::HashMap;
use sumcheck::structs::{IOPProof, IOPVerifierState};
use transcript::Transcript;

fn pow_two_omegas<E: ExtensionField>(n: usize, is_fft: bool) -> Vec<E> {
    let mut pows = vec![E::ZERO; n - 1];
    let mut rou: E = get_root_of_unity(n);
    if is_fft {
        rou = rou.inverse();
    }
    pows[0] = rou;
    for i in 1..(n - 1) {
        pows[i] = pows[i - 1] * pows[i - 1];
    }
    pows
}

fn phi_eval<E: ExtensionField>(
    r: &[E],
    rand1: E,
    rand2: E,
    exponents: &[E],
    first_iter: bool,
) -> E {
    let mut eval = E::ONE;
    for i in 0..r.len() {
        eval *= E::ONE - r[i] + r[i] * exponents[exponents.len() - r.len() + i];
    }

    if first_iter {
        eval = (E::ONE - rand2) * (E::ONE - rand1 + rand1 * eval);
    } else {
        eval = E::ONE - rand1 + (E::ONE - E::from_canonical_u64(2) * rand2) * rand1 * eval;
    }

    eval
}

/// Info about the convolution layer derived during the setup phase
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConvCtx {
    pub node_id: NodeId,
    pub kw: usize,
    pub kx: usize,
    pub real_nw: usize,
    pub nw: usize,
    pub filter_size: usize,
    pub unpadded_filter_shape: Shape,
    pub padded_filter_shape: Shape,
    pub filter_key: CommitmentId,
    pub bias_key: CommitmentId,
}

impl OpInfo for ConvCtx {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        Ok(input_shapes
            .iter()
            .map(|shape| self.output_shape(shape, padding_mode))
            .collect())
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        Convolution::<Element>::num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        format!(
            "Conv Ctx: ({},{},{},{})",
            self.kw, self.kx, self.nw, self.nw,
        )
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl ConvCtx {
    pub fn output_shape(&self, input_shape: &Shape, padding_mode: PaddingMode) -> Shape {
        match padding_mode {
            PaddingMode::NoPadding => conv2d_shape(input_shape, &self.unpadded_filter_shape),
            PaddingMode::Padding => padded_conv2d_shape(input_shape, &self.padded_filter_shape),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn verify_fft_delegation<
        E: ExtensionField,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        mut claim: E,
        proof: &ConvProof<E>,
        delegation_proof: &[IOPProof<E>],
        delegation_claims: &[Vec<E>],
        delegation_poly_aux: &[VPAuxInfo<E>],
        delegation_points: &[Vec<E>],
        mut prev_r: Vec<E>,
    ) -> Result<()> {
        let iter = delegation_proof.len();
        // Verify delegation protocol of W iFFT matrix
        let exponents = pow_two_omegas(iter + 1, false);
        for i in 0..iter {
            IOPVerifierState::<E>::verify(
                claim,
                &delegation_proof[i],
                &delegation_poly_aux[i],
                verifier.transcript,
            );

            ensure!(
                identity_eval(delegation_points[i].as_slice(), prev_r.clone().as_slice())
                    == delegation_claims[i][0],
                "Error in identity evaluation fft delegation iter : {i}"
            );

            ensure!(
                phi_eval(
                    &delegation_points[i],
                    proof.hadamard_point[i],
                    prev_r[prev_r.len() - 1],
                    &exponents,
                    i == 0
                ) == delegation_claims[i][1],
                "Error in phi computation fft delegation iter : {i}"
            );

            claim = delegation_claims[i][2];
            prev_r = delegation_points[i].clone();
        }
        ensure!(
            claim
                == (E::ONE - E::from_canonical_u64(2) * proof.hadamard_point[iter]) * prev_r[0]
                    + E::ONE
                    - prev_r[0],
            "Error in final FFT delegation step"
        );

        Ok(())
    }

    pub(crate) fn verify_convolution<
        E: ExtensionField,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        last_claim: &Claim<E>,
        proof: &ConvProof<E>,
        shape_step: &ShapeStep,
    ) -> anyhow::Result<Claim<E>> {
        ensure!(
            shape_step.unpadded_input_shape.len() == 1,
            "More than 1 unpadded input shape found for convolution layer",
        );
        ensure!(
            shape_step.padded_input_shape.len() == 1,
            "More than 1 padded input shape found for convolution layer",
        );
        // The first thing to do is to recreate the hadamard clearing tensor
        // Since this is only coming from public information, the verifier
        // creates the vector and evaluates it.
        // NOTE: for succinctness of verification, we could also have
        // the prover commits to the tensor product and we could skip this step.
        // OR find a closed formula
        //
        // To recreat it, we need the unpadded output shape and the real output shape.
        let unpadded_output_shape = conv2d_shape(
            &shape_step.unpadded_input_shape[0],
            &self.unpadded_filter_shape,
        );
        let real_output_shape =
            padded_conv2d_shape(&shape_step.padded_input_shape[0], &self.padded_filter_shape);
        let clearing_tensor = new_clearing_tensor(&unpadded_output_shape, &real_output_shape)?;
        // now we need to verify the hadamard proof for the sumcheck part.
        let hctx = hadamard::HadamardCtx::from_len(real_output_shape.product());
        let expected_v2_eval = clearing_tensor
            .to_field_mle()
            .evaluate(proof.clearing_proof.random_point());
        // also set the claim to be the non-cleared output of conv. The rest of the logic is about proving the bias + fft claims.
        let last_claim = hadamard::verify(
            &hctx,
            verifier.transcript,
            &proof.clearing_proof,
            last_claim,
            expected_v2_eval,
        )
        .context("failure for hadamard proof")?;

        let conv_claim = last_claim.eval - proof.bias_claim;

        let mut delegation_fft_aux = Vec::new();
        for i in (0..(self.filter_size.ilog2() as usize)).rev() {
            delegation_fft_aux.push(from_mle_list_dimensions(&[vec![i + 1, i + 1, i + 1]]));
        }
        ensure!(
            delegation_fft_aux.len() == proof.ifft_delegation_proof.len(),
            "Inconsistency in iFFT delegation proofs/aux size, {} != {}",
            delegation_fft_aux.len(),
            proof.ifft_delegation_proof.len(),
        );

        let fft_aux = from_mle_list_dimensions(&[vec![
            (self.filter_size.ilog2() as usize) + 1,
            (self.filter_size.ilog2() as usize) + 1,
        ]]);

        let hadamard_aux = from_mle_list_dimensions(&[vec![
            ((self.kx * self.filter_size).ilog2() as usize) + 1,
            ((self.kx * self.filter_size).ilog2() as usize) + 1,
            ((self.kx * self.filter_size).ilog2() as usize) + 1,
        ]]);

        IOPVerifierState::<E>::verify(conv_claim, &proof.ifft_proof, &fft_aux, verifier.transcript);

        let iter = proof.ifft_delegation_proof.len();
        let mut claim = proof.ifft_claims[1];
        let exponents = pow_two_omegas(iter + 1, true);
        let mut prev_r = proof.ifft_point.clone();
        for (i, ifft_proof) in proof.ifft_delegation_proof.iter().enumerate() {
            IOPVerifierState::<E>::verify(
                claim,
                ifft_proof,
                &delegation_fft_aux[i],
                verifier.transcript,
            );
            ensure!(
                identity_eval(
                    proof.ifft_delegation_points[i].as_slice(),
                    prev_r.clone().as_slice()
                ) == proof.ifft_delegation_claims[i][0],
                "Error in identity evaluation ifft delegation iter : {i}"
            );
            ensure!(
                phi_eval(
                    proof.ifft_delegation_points[i].as_slice(),
                    E::ONE - last_claim.point[i],
                    prev_r[prev_r.len() - 1],
                    &exponents,
                    false
                ) == proof.ifft_delegation_claims[i][1],
                "Error in phi computation ifft delegation iter : {i}"
            );

            prev_r = proof.ifft_delegation_points[i].clone();
            claim = proof.ifft_delegation_claims[i][2];
        }
        let scale = E::from_canonical_u64(1 << (iter + 1)).inverse();

        ensure!(
            claim == scale * (E::ONE) * prev_r[0] + scale * (E::ONE - prev_r[0]),
            "Error in final iFFT delegation step"
        );

        IOPVerifierState::<E>::verify(
            proof.ifft_claims[0],
            &proof.hadamard_proof,
            &hadamard_aux,
            verifier.transcript,
        );
        ensure!(
            proof.hadamard_clams[2] == identity_eval(&proof.ifft_point, &proof.hadamard_point),
            "Error in Beta evaluation"
        );

        // TODO : 1) Dont forget beta evaluation 2 verification of the last step of delegation
        // Verify fft sumcheck
        IOPVerifierState::<E>::verify(
            proof.hadamard_clams[1],
            &proof.fft_proof,
            &fft_aux,
            verifier.transcript,
        );
        claim = proof.fft_claims[1];

        ensure!(
            delegation_fft_aux.len() == proof.fft_delegation_proof.len(),
            "Inconsistency in FFT delegation proofs/aux size"
        );

        self.verify_fft_delegation(
            verifier,
            claim,
            proof,
            &proof.fft_delegation_proof,
            &proof.fft_delegation_claims,
            &delegation_fft_aux,
            &proof.fft_delegation_points,
            proof.fft_point.clone(),
        )?;

        IOPVerifierState::<E>::verify(
            proof.hadamard_clams[0],
            &proof.fft_proof_weights,
            &fft_aux,
            verifier.transcript,
        );
        claim = proof.fft_weight_claims[1];
        self.verify_fft_delegation(
            verifier,
            claim,
            proof,
            &proof.fft_delegation_proof_weights,
            &proof.fft_delegation_weights_claims,
            &delegation_fft_aux,
            &proof.fft_delegation_weights_points,
            proof.fft_weight_point.clone(),
        )?;

        // Validate the correctness of the padded weights claim
        // using the partial_evals provided by the prover
        let mut weights_point = proof.fft_weight_point.clone();
        let mut v = weights_point.pop().unwrap();
        v = (E::ONE - v).inverse();

        let y_weights = (0..self.real_nw)
            .flat_map(|i| (0..self.real_nw).map(move |j| (i, j)))
            .fold(E::ZERO, |acc, (i, j)| {
                acc + proof.partial_evals[i * self.real_nw + j]
                    * identity_eval(
                        &to_bits(i * self.nw + j, (self.nw.ilog2() as usize) * 2),
                        &weights_point,
                    )
            });

        ensure!(
            proof.fft_weight_claims[0] * v == y_weights,
            "Error in padded_fft evaluation claim"
        );

        let weights_rand: Vec<E> = verifier
            .transcript
            .read_challenges((self.real_nw * self.real_nw).ilog2() as usize);

        let point = [
            proof.hadamard_point.as_slice(),
            &last_claim.point[((self.filter_size).ilog2() as usize)..],
        ]
        .concat();

        let bias_claim = Claim::new(
            last_claim.point[(proof.ifft_delegation_proof.len())..].to_vec(),
            proof.bias_claim,
        );

        let filter_claim = Claim::new(
            [
                weights_rand.clone(),
                point[(2 * self.nw * self.nw).ilog2() as usize..].to_vec(),
            ]
            .concat(),
            proof
                .partial_evals
                .clone()
                .into_mle()
                .evaluate(&weights_rand),
        );
        // Add the common commitment claims to be verified
        let common_claims = {
            let mut claims = HashMap::new();
            claims.insert(self.filter_key.clone(), filter_claim);
            claims.insert(self.bias_key.clone(), bias_claim);
            claims
        };
        verifier.add_common_claims(self.node_id, common_claims);

        let mut input_point = proof.fft_point.clone();
        v = input_point.pop().unwrap();
        v = (E::ONE - v).inverse();
        for point in &mut input_point {
            *point = E::ONE - *point;
        }
        // the output claim for this step that is going to be verified at next step
        Ok(Claim {
            // the new randomness to fix at next layer is the randomness from the sumcheck !
            point: [
                input_point.clone(),
                proof.hadamard_point[((self.filter_size * 2).ilog2() as usize)..].to_vec(),
            ]
            .concat(),
            // the claimed sum for the next sumcheck is MLE of the current vector evaluated at the
            // random point. 1 because vector is secondary.
            eval: proof.fft_claims[0] * v,
        })
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for ConvCtx
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = ConvProof<E>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        Ok(vec![self.verify_convolution(
            verifier,
            last_claims[0],
            proof,
            shape_step,
        )?])
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

/// Contains proof material related to one step of the inference for a convolution layer
#[derive(Default, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct ConvProof<E: ExtensionField> {
    // Sumcheck proof for the FFT layer
    pub(crate) fft_proof: IOPProof<E>,
    pub(crate) fft_proof_weights: IOPProof<E>,
    pub(crate) fft_point: Vec<E>,
    // Proof for the evaluation delegation of the omegas matrix
    // It consists of multiple sumcheck proofs
    pub(crate) fft_delegation_proof: Vec<IOPProof<E>>,
    pub(crate) fft_delegation_proof_weights: Vec<IOPProof<E>>,
    // Likewise for fft, we define ifft proofs
    pub(crate) ifft_proof: IOPProof<E>,
    pub(crate) ifft_point: Vec<E>,
    pub(crate) ifft_delegation_proof: Vec<IOPProof<E>>,
    pub(crate) ifft_delegation_points: Vec<Vec<E>>,
    // Sumcheck proof for the hadamard product
    pub(crate) hadamard_proof: IOPProof<E>,
    pub(crate) hadamard_point: Vec<E>,
    // The evaluation claims produced by the corresponding sumchecks
    pub(crate) fft_claims: Vec<E>,
    pub(crate) fft_weight_claims: Vec<E>,
    pub(crate) ifft_claims: Vec<E>,
    pub(crate) fft_delegation_claims: Vec<Vec<E>>,
    pub(crate) fft_delegation_points: Vec<Vec<E>>,
    pub(crate) fft_delegation_weights_claims: Vec<Vec<E>>,
    pub(crate) fft_delegation_weights_points: Vec<Vec<E>>,
    pub(crate) fft_weight_point: Vec<E>,
    pub(crate) ifft_delegation_claims: Vec<Vec<E>>,
    pub(crate) partial_evals: Vec<E>,
    pub(crate) hadamard_clams: Vec<E>,
    pub(crate) bias_claim: E,
    pub(crate) clearing_proof: hadamard::HadamardProof<E>,
}
