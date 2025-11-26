#![allow(clippy::print_stdout)]
use std::path::PathBuf;
use zkml::{
    Element, IO, Proof, ProverContext, Tensor,
    iop::chunking::LLMChunkingStrategy,
    model::{
        exec_graph::InferenceEngine,
        llm::{Driver, LLMVerifierContext},
    },
    parser::{
        file_cache,
        gguf::RawGGUF,
        llm::{LLMTokenizer, Token, models::gpt2::GPT2, tokenizer::TokenizerLoader},
    },
};
mod common {
    include!("common/mod.rs");
}

use common::{F, Pcs};

use crate::common::GraphRuner;

const PRUNED_GPT2: &str = "gpt2.Q2_K.gguf";

fn full_model_path(gguf_path: &str) -> PathBuf {
    let src_model_path = file_cache::cache_path(gguf_path)
        .to_str()
        .unwrap()
        .to_string();
    let mut model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    model_path.push(src_model_path);
    model_path
}

// TODO: Make it generic over the data format and the model type
struct GPT2Runner {
    gguf_path: String,
    user_input: Vec<Token>,
    vk: Option<LLMVerifierContext<F, Pcs>>,
}

impl GPT2Runner {
    pub fn new(model_path: &str, user_input: &str) -> anyhow::Result<Self> {
        let model_path = full_model_path(model_path);
        let tokenizer = GPT2.load_tokenizer(&RawGGUF::new(model_path.clone()))?;
        let user_tokens = tokenizer.tokenize(user_input);
        Ok(Self {
            gguf_path: model_path.to_str().unwrap().to_string(),
            user_input: user_tokens,
            vk: None,
        })
    }
}

impl GraphRuner for GPT2Runner {
    type ChunkingStrategy = LLMChunkingStrategy;
    fn setup(
        &mut self,
    ) -> anyhow::Result<(ProverContext<F, Pcs>, InferenceEngine, Vec<Tensor<Element>>)> {
        let driver =
            Driver::load_from_model(GPT2, &RawGGUF::new(self.gguf_path.clone()), Some(10))?
                .into_provable_llm(None)?;
        let input_tensor = Tensor::new(
            vec![self.user_input.len()].into(),
            self.user_input
                .iter()
                .map(|t| t.as_tensor_type_param::<Element>())
                .collect::<Vec<_>>(),
        )?;
        let (prover_ctx, verifier_ctx) = driver.context::<F, Pcs>()?;
        self.vk = Some(verifier_ctx);
        Ok((prover_ctx, InferenceEngine::LLM(driver), vec![input_tensor]))
    }

    fn chunk_strategy(&self) -> Self::ChunkingStrategy {
        LLMChunkingStrategy
    }

    fn verify_proof(&self, proof: Proof<F, Pcs>, io: IO<F>) -> anyhow::Result<()> {
        self.vk
            .as_ref()
            .unwrap()
            .verify(proof, self.user_input.clone(), io)
            .unwrap();
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let runner = GPT2Runner::new(PRUNED_GPT2, "The sky is")?;
    common::main_loop(6, runner).unwrap();
    println!("Done");
    Ok(())
}
