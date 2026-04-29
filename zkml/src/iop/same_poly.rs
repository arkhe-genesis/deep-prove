//! This module contains logic to prove the opening of several claims related to the _same_ polynomial.
//! e.g. a set of (r_i,y_i) such that f(r_i) = y_i for all i's.
//! a_i = randomness() for i:0 -> |r_i|
//! for r_i, compute Beta_{r_i} = [beta_{r_i}(0),(1),...(2^|r_i|)]
//! then Beta_j = SUM_j a_i * Beta_{r_i}
//! final_y = SUM a_i * y_i
//!
//! Note the output of the verifier is a claim that needs to be verified outside of this protocol.
//! It could be via an opening directly OR via an accumulation scheme.

use crate::{
    Claim, VectorTranscript,
    poly_commit::{aggregated_rlc, identity_eval},
};
use anyhow::{Ok, ensure};
use ark_ff::PrimeField;
use dp_crypto::{
    Expression, IntoMLE,
    arkyper::transcript::Transcript,
    poly::{dense::DensePolynomial, eq::evals},
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
    virtual_poly::VPAuxInfo,
    virtual_polys::VirtualPolynomialsBuilder,
};
use either::Either;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Context<F: PrimeField> {
    vp_info: VPAuxInfo<F>,
}

impl<F: PrimeField> Context<F> {
    /// number of variables of the poly in question
    pub fn new(num_vars: usize) -> Self {
        Self {
            vp_info: crate::util::from_mle_list_dimensions(&[vec![num_vars, num_vars]]),
        }
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct Proof<F: PrimeField> {
    sumcheck: IOPProof<F>,
    #[serde(with = "dp_crypto::serialization")]
    eval: F,
}

pub struct Prover<'a, F: PrimeField> {
    claims: Vec<Claim<F>>,
    poly: DensePolynomial<'a, F>,
}

impl<'a, F> Prover<'a, F>
where
    F: PrimeField,
{
    /// The polynomial over which the claims are to be accumulated and proven
    /// Note the prover also _commits_ to this polynomial.
    pub fn new(poly: DensePolynomial<'a, F>) -> Self {
        Self {
            claims: Default::default(),
            poly,
        }
    }
    pub fn add_claim(&mut self, claim: Claim<F>) -> anyhow::Result<()> {
        ensure!(
            claim.point().len() == self.poly.num_vars(),
            format!(
                "Invalid claim length: input.len() = {} vs poly.num_vars = {} ",
                claim.point().len(),
                self.poly.num_vars()
            )
        );
        self.claims.push(claim);
        Ok(())
    }
    pub fn prove<T: Transcript>(self, t: &mut T) -> anyhow::Result<(Proof<F>, Claim<F>)> {
        let challenges = t.read_challenges(self.claims.len());

        let beta_evals = self
            .claims
            .into_par_iter()
            .map(|c_i| evals(c_i.point()).into_mle())
            .collect::<Vec<_>>();
        let num_vars = self.poly.num_vars();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<F>::new(num_threads, num_vars);
        let poly_expr = expr_builder.lift(Either::Left(&self.poly));
        let sum_expr =
            beta_evals
                .iter()
                .enumerate()
                .fold(Expression::Constant(F::ZERO), |acc, (i, p)| {
                    acc + Expression::Challenge(i as u16, 1, F::ONE, F::ZERO)
                        * expr_builder.lift(Either::Left(p))
                });
        let virtual_poly = expr_builder.to_virtual_polys(&[poly_expr * sum_expr], &challenges);
        let (sumcheck_proof, state) = IOPProverState::<F>::prove(virtual_poly, t);
        let eval = state.get_mle_flatten_final_evaluations()[0];
        Ok((
            Proof {
                sumcheck: sumcheck_proof,
                eval,
            },
            Claim::<F>::new(state.collect_raw_challenges(), eval),
        ))
    }
}

pub struct Verifier<'a, F: PrimeField> {
    claims: Vec<Claim<F>>,
    ctx: &'a Context<F>,
}

impl<'a, F: PrimeField> Verifier<'a, F> {
    pub fn new(ctx: &'a Context<F>) -> Self {
        Self {
            claims: Default::default(),
            ctx,
        }
    }

    pub fn add_claim(&mut self, claim: Claim<F>) -> anyhow::Result<()> {
        ensure!(
            claim.point().len() == self.ctx.vp_info.max_num_variables,
            "invalid input len wrt to poly in ctx, claim point length: {}, expected point length: {}",
            claim.point().len(),
            self.ctx.vp_info.max_num_variables
        );
        self.claims.push(claim);
        Ok(())
    }

    pub fn verify<T: Transcript>(self, proof: &Proof<F>, t: &mut T) -> anyhow::Result<Claim<F>> {
        let fs_challenges = t.read_challenges(self.claims.len());
        let (rs, ys): (Vec<_>, Vec<_>) = self.claims.into_iter().map(|c| (c.point, c.eval)).unzip();
        let y_res = aggregated_rlc(&ys, &fs_challenges)?;
        // check sumcheck proof
        let subclaim = IOPVerifierState::<F>::verify(y_res, &proof.sumcheck, &self.ctx.vp_info, t);
        let point = &subclaim.point;
        // check sumcheck output: first check for the betas we can compute
        // for(int i = 0; i < a.size(); i++){y += a[i]*identity_eval(claims[i].first,P.randomness[0]);}
        let computed_y = fs_challenges
            .into_iter()
            .zip(rs)
            .fold(F::ZERO, |acc, (a_i, r_i)| {
                acc + a_i * identity_eval(&r_i, point)
            });

        let calculated_claim = proof.eval * computed_y;
        ensure!(
            calculated_claim == subclaim.expected_evaluation,
            "Same Poly verification failed, calculated claim {:?} did not equal expected claim {:?}",
            calculated_claim,
            subclaim.expected_evaluation
        );

        // here instead of checking this claim via PCS, we actually put it in the output of the verify function.
        // That claims will be accumulated and verified elsewhere in the protocol.
        // Note the claim is only about the actual poly, not the betas since it has been verified just ^
        Ok(Claim::<F>::new(point.clone(), proof.eval))
    }
}

#[cfg(test)]
mod test {
    use dp_crypto::{IntoMLE, arkyper::CommitmentScheme};

    use crate::{
        Claim, default_transcript, rng_from_env_or_random,
        testing::{Pcs, random_field_vector},
    };
    use itertools::Itertools;

    use super::{Context, Prover, Verifier};

    type F = ark_bn254::Fr;

    #[test]
    fn test_pcs() {
        let num_vars = 10;
        let mut rng = rng_from_env_or_random();
        let _param = Pcs::test_setup(&mut rng, num_vars);
    }

    #[test]
    fn test_same_poly_proof() -> anyhow::Result<()> {
        // number of vars
        let num_vars = 10_usize;
        let poly_len = 1 << num_vars;
        let poly = random_field_vector::<F>(poly_len);
        let poly_mle = poly.clone().into_mle();
        // number of clains
        let m = 14;
        let claims = (0..m)
            .map(|_| {
                let r_i = random_field_vector::<F>(num_vars);
                let y_i = poly_mle.evaluate(&r_i).unwrap();
                (r_i, y_i)
            })
            .collect_vec();
        // COMMON PART
        assert_eq!(poly.len(), 1 << num_vars);
        let ctx = Context::new(num_vars);
        // PROVER PART
        let mut t = default_transcript();
        let mut prover = Prover::new(poly_mle.clone());
        for (r_i, y_i) in claims.clone().into_iter() {
            prover.add_claim(Claim::new(r_i, y_i))?;
        }
        let (proof, _) = prover.prove(&mut t)?;
        // VERIFIER PART
        let mut t = default_transcript();
        let mut verifier = Verifier::new(&ctx);
        for (r_i, y_i) in claims.into_iter() {
            verifier.add_claim(Claim::new(r_i, y_i))?;
        }
        let claim = verifier.verify(&proof, &mut t)?;
        let expected = poly_mle.evaluate(&claim.point).unwrap();
        assert_eq!(claim.eval, expected);
        Ok(())
    }
}
