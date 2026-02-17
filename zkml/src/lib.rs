//! Deepprove library
#![recursion_limit = "1024"]
#![feature(iter_next_chunk)]
#![feature(min_specialization)]
#![feature(exact_size_is_empty)]
#![feature(mapped_lock_guards)]

use std::{borrow::Borrow, env, ops::Deref, str::FromStr};

use ark_std::rand::{self, SeedableRng, rngs::StdRng};
use ff_ext::{ExtensionField, FieldFrom};
use itertools::Itertools;
use multilinear_extensions::mle::PointAndEval;
use quantization::ToField;
use serde::{Deserialize, Serialize};
use transcript::{BasicTranscript, Transcript};

mod backend;
mod commit;
mod fft;
pub mod graph;
pub mod inputs;
pub mod iop;
pub mod layers;
pub mod lookup;
pub mod measure;
pub mod model;
pub mod number;
pub mod padding;
pub mod parser;
pub mod quantization;
pub mod shape;
pub mod tensor;
pub use crate::number::Number;

// Re-exports
pub use iop::{
    Proof, ProverContext,
    claim::Claim,
    prover::Prover,
    verifier::{IO, verify},
};
pub use quantization::{ScalingFactor, ScalingStrategy};
pub use shape::Shape;
pub use tensor::Tensor;

#[cfg(feature = "capture-layers-quant")]
pub mod capture;

#[cfg(test)]
mod testing;
pub(crate) mod util;

pub const GIT_VERSION: &str = git_version::git_version!(args = ["--abbrev=6", "--always"]);
pub fn version() -> String {
    GIT_VERSION.to_string()
}

/// We allow higher range to account for overflow. Since we do a requant after each layer, we
/// can support with i128 with 8 bits quant:
/// 16 + log(c) = 64 => c = 2^48 columns in a dense layer
pub type Element = i64;

impl<E: ExtensionField> From<PointAndEval<E>> for Claim<E> {
    fn from(value: PointAndEval<E>) -> Self {
        Claim {
            point: value.point.clone(),
            eval: value.eval,
        }
    }
}

impl<E: ExtensionField> From<&PointAndEval<E>> for Claim<E> {
    fn from(value: &PointAndEval<E>) -> Self {
        Claim {
            point: value.point.clone(),
            eval: value.eval,
        }
    }
}

/// Returns the default transcript the prover and verifier must instantiate to validate a proof.
pub fn default_transcript<E: ExtensionField>() -> BasicTranscript<E> {
    BasicTranscript::new(b"m2vec")
}

/// Returns the bit sequence of num of bit_length length.
pub(crate) fn to_bit_sequence_le(
    num: usize,
    bit_length: usize,
) -> impl DoubleEndedIterator<Item = usize> {
    assert!(
        bit_length as u32 <= usize::BITS,
        "bit_length cannot exceed usize::BITS"
    );
    (0..bit_length).map(move |i| (num >> i) & 1)
}

/// Method to efficiency evaluate the MLE of the zeroifier matrix over a random
/// point. The point is provided already split between coordinates referring to the
/// columns and coordinates referring to the rows of the matrix.
/// Currently, it works only for a square zeroifier matrix
pub fn eval_zeroifier_mle<F: ExtensionField>(column_point: &[F], row_point: &[F]) -> F {
    column_point
        .iter()
        .zip(row_point)
        .fold(F::ONE, |acc, (&c, &r)| {
            acc * (F::ONE - c - r + F::from_canonical_u64(2) * c * r) + (F::ONE - c) * r
        })
}

/// Method to efficiency evaluate the MLE of the infinitizer matrix over a random
/// point. The point is provided already split between coordinates referring to the
/// columns and coordinates referring to the rows of the matrix.
/// Currently, it works only for a square infinitizer matrix
pub fn eval_infinitizer_mle<F: ExtensionField + FieldFrom<u64>>(
    column_point: &[F],
    row_point: &[F],
    minus_infinity: Element,
) -> F {
    <Element as ToField<F>>::to_field(&minus_infinity)
        * (F::ONE - eval_zeroifier_mle(column_point, row_point))
}

pub trait VectorTranscript<E: ExtensionField> {
    fn read_challenges(&mut self, n: usize) -> Vec<E>;
}

impl<T: Transcript<E>, E: ExtensionField> VectorTranscript<E> for T {
    fn read_challenges(&mut self, n: usize) -> Vec<E> {
        (0..n).map(|_| self.read_challenge().elements).collect_vec()
    }
}

pub trait InitTranscript {
    type InitData: Default + From<&'static [u8]>;

    fn new(init_data: Self::InitData) -> Self;
}

impl<E: ExtensionField> InitTranscript for BasicTranscript<E> {
    type InitData = &'static [u8];

    fn new(init_data: Self::InitData) -> Self {
        Self::new(init_data)
    }
}

/// Converts an iterator of elements to the base field.
pub(crate) fn to_base<E, I>(iter: I) -> Vec<E::BaseField>
where
    I: IntoIterator,
    I::Item: Borrow<Element>,
    Element: ToField<E>,
    E: ExtensionField,
{
    iter.into_iter()
        .map(|v| v.borrow().to_field().as_bases()[0])
        .collect()
}

/// Converts an iterator of elements to the extension field.
pub(crate) fn to_field<T, E, I>(iter: I) -> Vec<E>
where
    I: IntoIterator,
    I::Item: Borrow<T>,
    T: ToField<E>,
    E: ExtensionField,
{
    iter.into_iter().map(|v| v.borrow().to_field()).collect()
}

pub trait NextPowerOfTwo {
    /// Returns a new vector where each element is the next power of two.
    fn next_power_of_two(&self) -> Self;
}

// For unsigned integer vectors
impl NextPowerOfTwo for Vec<usize> {
    fn next_power_of_two(&self) -> Self {
        self.iter().map(|&i| i.next_power_of_two()).collect()
    }
}

impl NextPowerOfTwo for Shape {
    fn next_power_of_two(&self) -> Self {
        Shape::new(self.deref().next_power_of_two())
    }
}

impl NextPowerOfTwo for Vec<Shape> {
    fn next_power_of_two(&self) -> Self {
        self.iter().map(|el| el.next_power_of_two()).collect()
    }
}

#[cfg(test)]
static INIT: std::sync::Once = std::sync::Once::new();

#[cfg(test)]
pub fn init_test_logging_default() {
    use tracing_subscriber::EnvFilter;

    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        tracing_subscriber::fmt().with_env_filter(filter).init();
    });
}

#[cfg(test)]
pub fn init_test_logging(default_level: &str) {
    use tracing_subscriber::EnvFilter;

    INIT.call_once(|| {
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));
        tracing_subscriber::fmt().with_env_filter(filter).init();
    });
}

/// Get a rng generator from a seed from env var or generate a random one
pub fn rng_from_env_or_random() -> StdRng {
    let seed = seed_from_env_or_rng();
    StdRng::seed_from_u64(seed)
}

/// Get a seed from env var or generate a random one
pub fn seed_from_env_or_rng() -> u64 {
    env::var("RNG_SEED")
        .map(|val| u64::from_str(&val).expect("RNG_SEED must be a u64"))
        .unwrap_or_else(|_| rand::random::<u64>())
}

#[cfg(test)]
mod test {
    use ark_std::rand::Rng;
    use ceno_p3::field::FieldAlgebra;
    use ff_ext::{FromUniformBytes, GoldilocksExt2};
    use itertools::Itertools;
    use multilinear_extensions::mle::IntoMLE;
    use tenstore::GenStore;
    use transcript::BasicTranscript;

    use crate::{
        iop::{prover::Prover, verifier::verify},
        parser::onnx::FloatOnnxLoader,
        rng_from_env_or_random,
        tensor::Tensor,
        testing::Pcs,
        to_bit_sequence_le,
    };

    type E = GoldilocksExt2;
    type T = BasicTranscript<E>;

    type P<'a, 'b> = Prover<'a, 'b, E, T, Pcs<E>>;

    #[test]
    fn test_model_run() -> anyhow::Result<()> {
        test_model_run_helper()?;
        Ok(())
    }

    use std::path::PathBuf;

    fn workspace_root() -> PathBuf {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        PathBuf::from(manifest_dir).parent().unwrap().to_path_buf()
    }

    fn test_model_run_helper() -> anyhow::Result<()> {
        let filepath = workspace_root().join("zkml/assets/model.onnx");
        let (model, _md) = FloatOnnxLoader::new(&filepath.to_string_lossy()).build()?;

        println!("[+] Loaded onnx file");
        let (prover_ctx, verifier_ctx) = model
            .generate_contexts::<GoldilocksExt2, Pcs<GoldilocksExt2>>()
            .expect("Unable to generate contexts");
        println!("[+] Setup parameters");

        let shapes = model.input_shapes();
        assert_eq!(shapes.len(), 1);
        let shape = &shapes[0];
        assert_eq!(shape.len(), 1);
        let input = Tensor::random(&vec![shape[0]].into());
        println!("input: {:?}", input.data());
        let inputs = model.prepare_inputs(vec![input])?;

        let trace = model.run(inputs, &mut GenStore::default()).unwrap();

        let output = trace.outputs().first().unwrap();
        println!("[+] Run inference. Result: {output:?}");

        let io = trace.to_verifier_io()?;
        println!("[+] Run prover");
        let proof = P::prove(&prover_ctx, trace, &model).expect("unable to generate proof");

        verify::<_, T, _>(&verifier_ctx, proof, io).expect("invalid proof");
        println!("[+] Verify proof: valid");
        Ok(())
    }

    // TODO: move below code to a vector module

    #[test]
    fn test_vector_mle() {
        let n = 10_usize.next_power_of_two();
        let v = (0..n)
            .map(|_| <E as FromUniformBytes>::random(&mut rng_from_env_or_random()))
            .collect_vec();
        let mle = v.clone().into_mle();
        let random_index = rng_from_env_or_random().gen_range(0..v.len());
        let eval = to_bit_sequence_le(random_index, v.len().next_power_of_two().ilog2() as usize)
            .map(|b| E::from_canonical_u64(b as u64))
            .collect_vec();
        let output = mle.evaluate(&eval);

        assert_eq!(output, v[random_index]);
    }
}
