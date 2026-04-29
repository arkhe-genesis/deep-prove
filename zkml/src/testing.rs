use crate::{Element, quantization, rng_from_env_or_random, seed_from_env_or_rng};
use ark_bn254::Bn254;
use ark_ff::PrimeField;
use ark_std::rand::{Rng, SeedableRng, rngs::StdRng};
use dp_crypto::arkyper::HyperKZG;
use itertools::Itertools;

// The hasher type is chosen depending on the feature flag
pub(crate) type Pcs = HyperKZG<Bn254>;

pub fn random_vector(n: usize) -> Vec<Element> {
    let mut rng = rng_from_env_or_random();
    (0..n)
        .map(|_| rng.gen_range(*quantization::MIN..=*quantization::MAX))
        .collect_vec()
}

pub fn random_field_vector<F: PrimeField>(n: usize) -> Vec<F> {
    let mut rng = rng_from_env_or_random();
    (0..n).map(|_| F::rand(&mut rng)).collect_vec()
}

#[allow(unused)]
pub fn random_vector_seed(n: usize, seed: Option<u64>) -> Vec<Element> {
    let seed = seed.unwrap_or_else(seed_from_env_or_rng); // Use provided seed or default
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| rng.gen_range(*quantization::MIN..=*quantization::MAX))
        .collect_vec()
}
