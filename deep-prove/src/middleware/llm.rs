use ff_ext::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams};
use serde::{Deserialize, Serialize};
use zkml::{
    model::llm::{LLMProof, LLMVerifierContext},
    parser::llm::Token,
};

/// Extension field and PCS aliases used for LLM proofs.
pub type F = GoldilocksExt2;
pub type Pcs = Basefold<F, BasefoldRSParams>;

/// Informational payload persisted with the proof; `llm_response` is for logging only.
#[derive(Serialize, Deserialize)]
pub struct LlmOneShotOutput {
    pub model_name: String,
    pub prompt: String,
    pub tokens: Vec<Token>,
    #[serde(default)]
    pub llm_response: Option<String>,
    pub proof: LLMProof<F, Pcs>,
    pub verifier: LLMVerifierContext<F, Pcs>,
}
