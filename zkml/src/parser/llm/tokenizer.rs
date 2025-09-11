use ahash::AHashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::quantized::gguf_file::Value;
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::{bpe::BPE, unigram::Unigram},
    pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer,
    tokenizer::Tokenizer as InnerTokenizer,
};

use crate::parser::{
    gguf::FileTensorLoader,
    llm::{LLMVariant, Token as LLMToken},
};

/// A trait for tokenizers that can be used to tokenize and detokenize text.
pub trait LLMTokenizer {
    /// Tokenize a sentence into a vector of tokens
    fn tokenize(&self, sentence: &str) -> Vec<LLMToken>;
    /// Detokenize a vector of tokens back into a string
    fn detokenize(&self, ids: &[LLMToken]) -> String;
}

/// Wrapper around the HuggingFace tokenizers library
pub struct HFTokenizer {
    /// The underlying HuggingFace tokenizer
    tokenizer: InnerTokenizer,
}

impl HFTokenizer {
    pub fn from_gguf_path(path: &Path) -> Result<Self> {
        let loader = FileTensorLoader::from_path(path)?;
        Self::from_loader(&loader)
    }
    /// Create a tokenizer from a GGUF file loader
    pub fn from_loader(loader: &FileTensorLoader) -> Result<Self> {
        let variant = LLMVariant::from_loader(loader)?;
        match variant {
            LLMVariant::GPT2 => Self::bpe_from_loader(loader),
            LLMVariant::Gemma3 => Self::sentencepiece_from_loader(loader),
        }
    }

    /// Create a SentencePiece/Unigram tokenizer from GGUF loader
    pub fn sentencepiece_from_loader(loader: &FileTensorLoader) -> Result<Self> {
        let tokens = loader
            .metadata::<Vec<Value>>("tokenizer.ggml.tokens")
            .context("tokens metadata not found")?
            .into_iter()
            .map(|v| {
                v.to_string()
                    .cloned()
                    .with_context(|| "failed to convert Value to String".to_string())
            })
            .collect::<Result<Vec<_>>>()?;

        let scores = loader
            .metadata::<Vec<Value>>("tokenizer.ggml.scores")
            .context("scores metadata not found")?
            .into_iter()
            .map(|v| Ok(v.to_f32().unwrap() as f64))
            .collect::<Result<Vec<_>>>()?;

        let unk_id = loader
            .metadata::<usize>("tokenizer.ggml.unknown_token_id")
            .context("unknown_token_id not found")?;

        let vocab: Vec<(String, f64)> = tokens.into_iter().zip(scores).collect();

        let Ok(unigram) = Unigram::from(vocab, Some(unk_id), false) else {
            bail!("failed to create Unigram model");
        };
        let mut tokenizer = InnerTokenizer::new(unigram);

        // Add SentencePiece preprocessing for Gemma3 - handle spaces prefix
        use tokenizers::{
            decoders::metaspace::Metaspace as MetaspaceDecoder,
            pre_tokenizers::metaspace::{Metaspace, PrependScheme},
        };
        tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::First, true)));
        tokenizer.with_decoder(Some(MetaspaceDecoder::new('▁', PrependScheme::First, true)));

        Ok(Self { tokenizer })
    }

    /// Create a BPE tokenizer from GGUF loader
    pub fn bpe_from_loader(loader: &FileTensorLoader) -> Result<Self> {
        let tokens = loader
            .metadata::<Vec<Value>>("tokenizer.ggml.tokens")
            .unwrap()
            .into_iter()
            .map(|v| {
                v.to_string()
                    .cloned()
                    .with_context(|| "failed to convert Value to String".to_string())
            })
            .collect::<Result<Vec<_>>>()?;

        let merges = loader
            .metadata::<Vec<Value>>("tokenizer.ggml.merges")
            .unwrap()
            .into_iter()
            .map(|v| {
                v.to_string()
                    .cloned()
                    .with_context(|| "failed to convert Value merges to String".to_string())
            })
            .collect::<Result<Vec<_>>>()?;

        let bos_id = loader
            .metadata::<u32>("tokenizer.ggml.bos_token_id")
            .unwrap();
        let eos_id = loader
            .metadata::<u32>("tokenizer.ggml.eos_token_id")
            .unwrap();

        // Create vocabulary mapping
        let vocab: AHashMap<String, u32> = tokens
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, i as u32))
            .collect();

        let merges_list = parse_merges(&merges);
        let bpe = BPE::new(vocab, merges_list);
        let mut tokenizer = InnerTokenizer::new(bpe);

        // Add ByteLevel pre-tokenizer for proper space handling in GPT2
        tokenizer.with_pre_tokenizer(Some(ByteLevelPreTokenizer::new(true, false, true)));

        // Add ByteLevel decoder for proper detokenization in GPT2
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));

        assert!(
            tokenizer.id_to_token(bos_id).is_some(),
            "no BOS token present"
        );
        assert!(
            tokenizer.id_to_token(eos_id).is_some(),
            "no EOS token present"
        );
        Ok(Self { tokenizer })
    }
}

impl LLMTokenizer for HFTokenizer {
    /// Tokenize a sentence into a vector of tokens
    fn tokenize(&self, sentence: &str) -> Vec<LLMToken> {
        let encoding = self
            .tokenizer
            .encode(sentence.to_string(), true)
            .expect("Failed to tokenize");

        encoding
            .get_ids()
            .iter()
            .map(|&id| LLMToken::from(id as usize))
            .collect()
    }

    /// Detokenize a vector of tokens back into a string
    fn detokenize(&self, ids: &[LLMToken]) -> String {
        let token_ids: Vec<u32> = ids.iter().map(|t| t.0 as u32).collect();
        let decoded = self
            .tokenizer
            .decode(&token_ids, true)
            .expect("Failed to detokenize");

        // GPT2 ByteLevel encoding adds a leading space, remove it if present
        decoded.strip_prefix(' ').unwrap_or(&decoded).to_string()
    }
}

fn parse_merges(merges: &[String]) -> Vec<(String, String)> {
    merges
        .iter()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let first = parts.next()?;
            let second = parts.next()?;
            Some((first.to_string(), second.to_string()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::parser::{
        file_cache,
        gguf::tests::{GEMMA3_Q8, GPT2_Q8_0},
    };

    use super::*;

    #[test]
    fn test_tokenizer_from_gguf_path() -> anyhow::Result<()> {
        let paths = vec![GPT2_Q8_0, GEMMA3_Q8];
        for path in paths {
            let path = file_cache::from_cache(path)?;
            let tokenizer = HFTokenizer::from_gguf_path(&path)?;
            let s = "do or don't. there is no try.";
            let tokens = tokenizer.tokenize(s);
            let s2 = tokenizer.detokenize(&tokens);
            assert_eq!(s, s2, "failing token for model {}", path.display());
        }
        Ok(())
    }
}
