use mpcs::{Basefold, BasefoldRSParams};
use serde::{Deserialize, Serialize};
pub use zkml::inputs::Input;
use zkml::{Element, Proof as ProofG, Tensor, quantization::ScalingStrategyKind};

use super::{llm::LlmProvable, v2::Provable};

pub type E = super::v2::E;
pub type T = super::v2::T;

/// A type of the proof for the `v1` of the protocol
pub type Proof = ProofG<E, Basefold<E, BasefoldRSParams>>;

/// The `v1` proving request
#[derive(Serialize, Deserialize)]
pub struct DeepProveRequest {
    /// The model
    pub model: Vec<u8>,

    /// Optional precomputed hash of the model file.
    #[serde(default)]
    pub model_file_hash: Option<String>,

    /// An array of inputs to run proving for
    pub input: Input,

    /// Model scaling strategy
    pub scaling_strategy: ScalingStrategyKind,

    /// A hash of model scaling strategy input, if any
    pub scaling_input_hash: Option<String>,
}

/// The `v1` proofs that have been computed by the worker
#[derive(Serialize, Deserialize)]
pub struct DeepProveResponse {
    pub outputs: Vec<Output>,
}

#[derive(Serialize, Deserialize)]
pub struct Output {
    /// Model run outputs
    pub outputs: Vec<Tensor<Element>>,
    /// Generated proof
    pub proof: Provable,
}

#[derive(Serialize, Deserialize)]
pub struct LlmOutput {
    /// Model run outputs
    pub outputs: Vec<Tensor<Element>>,
    /// Generated LLM proof with verification context
    pub proof: LlmProvable,
}
