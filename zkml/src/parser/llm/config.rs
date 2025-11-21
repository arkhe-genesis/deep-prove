use crate::{
    layers::transformer::{attention_mask::AttentionSpan, positional::rope::RopeLayout},
    parser::{
        gguf::FileTensorLoader,
        json,
        llm::{Token, transformer::NormType},
    },
};
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};

/// Note some specific models may have additional parameters that are not included in this config.
/// In particular, structural information like which attention type the model is using is not included here,
/// since these information can not be derived from any format; these are default information that the caller
/// must provide.
/// NOTE: can't derive Eq since f32 doesn't implement it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LLMConfig {
    /// The name of the model architecture, e.g. "gpt2", "gemma3", etc.
    pub model_name: String,
    /// The size of an embedding vector (each token gets translated to an embedding vector of this size)
    pub embedding_size: usize,
    /// Size of the attention layer matrices.
    pub hidden_size: usize,
    /// The number of "heads" that are used within each attention layer.
    pub num_heads: usize,
    /// The size of each head. Note it's not always equal to the hidden_size / num_heads.
    pub head_size: usize,
    /// The number of blocks / attention layers there is in the model
    pub num_block: usize,
    /// The maximum size that the tensor containing input + generated token can have. Beyond that, we should not
    /// run the tensor through the model anymore.
    pub context_length: usize,
    /// LayerNorm needs an epsilon value to determine the precision. This is it.
    pub norm_epsilon: f32,
    /// The size of the vocabulary of the model, e.g. each token is an integer in [0, vocab_size)
    pub vocab_size: usize,
    /// The token that signals the end of the sequence.
    pub eos_token: Token,
}

/// It contains the structure of the model, like what kind of attention is used.
/// Non exported struct since it's only used internally to build most LLMs without duplication of code.
/// For some LLM, it may be preferable to fully write the decoding logic if the structure can not be encoded into this config.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LLMStructure {
    pub(crate) generic: LLMConfig,
    /// which norm is the LLM using
    pub(crate) norm_type: NormType,
    /// what's the span of the attention and what kind of attention is used
    pub(crate) attention_config: AttentionConfig,
    /// Global positional encoding after embeddings.
    pub(crate) global_positional: Option<PositionalConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PositionalConfig {
    /// Positional encoding is applied after the embeddings are computed.
    FixedPositional,
    /// Rope positional encoding is applied after the QK attention. The argument is
    /// the maximum sequence length of the model we wanna support.
    Rope(RopeConfig),
}

/// Config items for RoPe
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RopeConfig {
    /// HACK for now to avoid committing to the full length of Rope if we dont need to
    pub max_seq_length: usize,
    /// base frequency to use to construct the sin and cos matrices
    pub base_frequency: f32,
    /// The different ways to compute rope
    pub layout: RopeLayout,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AttentionConfig {
    // A vector, one for each attention layer. Easier to directly store all of them
    // rather than indicating which ones are global and which ones are local.
    pub span: Vec<AttentionSpan>,
    /// The type of attention that is used in the model. Support for MHA and GQA for now.
    pub head: AttentionHeadType,
}

/// The type of attention that is used in the model. Support for MHA and GQA for now.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AttentionHeadType {
    /// Multi-Head Attention
    MHA,
    /// Grouped-Query Attention (GQA) with a specific number of groups.
    GQA(usize),
}

impl LLMConfig {
    pub fn from_gguf(l: &FileTensorLoader, model_name: &str) -> anyhow::Result<Self> {
        let embedding_size = l
            .metadata::<usize>(&l.embedding_size_key(model_name))
            .context("embedding_size_key not found")?;
        let hidden_size = l
            .metadata::<usize>(&l.hidden_size_key(model_name))
            .context("hidden_size_key not found")?;
        let num_heads = l
            .metadata::<usize>(&l.num_heads_key(model_name))
            .context("num_heads_key not found")?;
        let context_length = l
            .metadata::<usize>(&l.context_length_key(model_name))
            .context("context_length_key not found")?;
        let norm_epsilon = l
            .find_norm_epsilon(model_name)
            .context("norm_epsilon_key not found")?;
        let num_block = l
            .metadata::<usize>(&l.num_block_key(model_name))
            .context("num_block_key not found")?;
        let vocab_size = l
            .raw_metadata("tokenizer.ggml.tokens")
            .context("tokens metadata not found")?
            .to_vec()
            .context("tokens metadata not found")?
            .len();

        let head_size = {
            if let Some(head_size) = l.raw_metadata(&l.head_size_key(model_name)) {
                head_size.to_u32().unwrap() as usize
            } else {
                ensure!(
                    hidden_size % num_heads == 0,
                    "hidden_size ({hidden_size}) is not divisible by num_heads ({num_heads})"
                );
                hidden_size / num_heads
            }
        };
        // let attention_config = AttentionConfig::from_loader(l, &variant)?;
        let eos_token = l
            .metadata::<usize>(&l.eos_token_key())
            .context("eos token not found")?
            .into();
        Ok(Self {
            model_name: model_name.to_string(),
            hidden_size,
            embedding_size,
            num_heads,
            head_size,
            context_length,
            norm_epsilon,
            num_block,
            vocab_size,
            eos_token,
        })
    }

    pub fn from_json(l: &json::FileTensorLoader) -> anyhow::Result<Self> {
        let hidden_size = l.metadata_to_u32("hidden_dim")? as usize;
        let embedding_size = hidden_size;
        let num_heads = l.metadata_to_u32("num_attention_heads")? as usize;
        let num_blocks = l.metadata_to_u32("num_hidden_layers")? as usize;
        let context_length = l.metadata_to_u32("max_seq_len")? as usize;
        let norm_epsilon = l.metadata_to_f32("norm_epsilon")?;
        // TODO: fix that, currently it's only used for debugging purposes the JSON format
        // and it doesn't export the vocab size.
        let vocab_size = 3;
        Ok(Self {
            model_name: "gpt2".to_string(),
            embedding_size,
            hidden_size,
            num_heads,
            head_size: hidden_size / num_heads,
            num_block: num_blocks,
            context_length,
            norm_epsilon,
            vocab_size,
            // only support for gpt2 for now
            eos_token: 50256usize.into(),
        })
    }
}

impl AttentionConfig {
    pub fn from_json(l: &json::FileTensorLoader) -> anyhow::Result<Self> {
        let num_attentions = l
            .get_metadata("num_attention_heads")
            .map(|v| v.as_u64().unwrap() as usize)
            .context("num_attention_heads not found")?;
        Ok(Self {
            span: (0..num_attentions).map(|_| AttentionSpan::Full).collect(),
            head: AttentionHeadType::MHA,
        })
    }

    pub fn spans(&self) -> impl Iterator<Item = AttentionSpan> + use<'_> {
        self.span.iter().cloned()
    }
}
