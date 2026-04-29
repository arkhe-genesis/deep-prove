use serde::{Deserialize, Serialize};
use zkml::{
    IO, Proof as ZkmlProof,
    model::llm::{LLMProof, LLMVerifierContext},
    parser::llm::Token,
};

pub type F = super::v1::F;
pub type Pcs = super::v1::Pcs;

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

#[derive(Serialize, Deserialize)]
pub struct LlmProvable {
    pub proof: ZkmlProof<F, Pcs>,
    pub io: IO<F>,
    pub ctx: LLMVerifierContext<F, Pcs>,
    pub user_tokens: Vec<Token>,
}

impl LlmProvable {
    pub fn verify(self) -> anyhow::Result<()> {
        self.ctx.verify(self.proof, self.user_tokens, self.io)
    }
}
