use crate::{
    layers::transformer::attention::attention_mask::AttentionSpan,
    parser::{
        gguf::FileTensorLoader,
        json,
        llm::{LLMModel, Token, transformer::NormType},
    },
};
use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};

pub const GPT2_VARIANTS: &[&str] = &[
    "gpt2",
    "Tmkrzx_X",
    "distilgpt2",
    "toy_gpt2",
    "sshleifer/tiny-gpt2",
];
pub const GEMMA3_VARIANTS: &[&str] = &["gemma-3"];

/// The config of the model containing all parameters that characterize the specific model being
/// extracted from the gguf file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
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
    /// The type of attention that is used in the model. Support for MHA and GQA for now.
    pub attention_config: AttentionConfig,
    /// The specific config for the variant.
    pub variant: LLMVariant,
}

/// The classes of LLM that are supported. These are named according to the literature
/// so GPT2 is the original GPT-2 model, but any model with the same architecture can be used.
/// Gemma3 is the Gemma-3 model but any model with the same architecture can be used.
/// This enum is mostly an entry point for the user and is linked to the `AttentionType` enum.
/// However, multiple different models may use the same attention type but may have however
/// other differences in their configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LLMVariant {
    GPT2,
    Gemma3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    // A vector, one for each attention layer. Easier to directly store all of them
    // rather than indicating which ones are global and which ones are local.
    span: Vec<AttentionSpan>,
    head: AttentionHeadType,
}

/// The type of attention that is used in the model. Support for MHA and GQA for now.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionHeadType {
    /// Multi-Head Attention
    MHA,
    /// Grouped-Query Attention (GQA) with a specific number of groups.
    GQA(usize),
}

impl LLMConfig {
    pub fn from_content(l: &FileTensorLoader) -> anyhow::Result<Self> {
        let variant = LLMVariant::from_loader(l)?;
        let embedding_size = l
            .metadata::<usize>(variant.embedding_size_key())
            .context("embedding_size_key not found")?;
        let hidden_size = l
            .metadata::<usize>(variant.hidden_size_key())
            .context("hidden_size_key not found")?;
        let num_heads = l
            .metadata::<usize>(variant.num_heads_key())
            .context("num_heads_key not found")?;
        let context_length = l
            .metadata::<usize>(variant.context_length_key())
            .context("context_length_key not found")?;
        let norm_epsilon = l
            .metadata::<f32>(variant.norm_epsilon_key())
            .context("norm_epsilon_key not found")?;
        let num_block = l
            .metadata::<usize>(variant.num_block_key())
            .context("num_block_key not found")?;
        let vocab_size = l
            .raw_metadata("tokenizer.ggml.tokens")
            .context("tokens metadata not found")?
            .to_vec()
            .context("tokens metadata not found")?
            .len();

        let attention_config = AttentionConfig::from_loader(l, &variant)?;
        let head_size = match variant {
            LLMVariant::GPT2 => hidden_size / num_heads,
            LLMVariant::Gemma3 => l
                .metadata::<usize>(variant.head_size_key())
                .context("head_size_key not found")?,
        };
        let eos_token = l
            .metadata::<usize>(variant.eos_token_key())
            .context("eos token not found")?
            .into();
        Ok(Self {
            hidden_size,
            embedding_size,
            num_heads,
            head_size,
            context_length,
            norm_epsilon,
            num_block,
            vocab_size,
            attention_config,
            eos_token,
            variant,
        })
    }

    pub fn from_json(l: &json::FileTensorLoader) -> anyhow::Result<Self> {
        let variant = LLMVariant::from_json(l)?;
        let hidden_size = l.metadata_to_u32("hidden_dim")? as usize;
        let embedding_size = hidden_size;
        let num_heads = l.metadata_to_u32("num_attention_heads")? as usize;
        let num_blocks = l.metadata_to_u32("num_hidden_layers")? as usize;
        let context_length = l.metadata_to_u32("max_seq_len")? as usize;
        let norm_epsilon = l.metadata_to_f32("norm_epsilon")?;
        let attention_config = AttentionConfig::from_json(l, &variant)?;
        // TODO: fix that, currently it's only used for debugging purposes the JSON format
        // and it doesn't export the vocab size.
        let vocab_size = 3;
        Ok(Self {
            embedding_size,
            hidden_size,
            num_heads,
            head_size: hidden_size / num_heads,
            num_block: num_blocks,
            context_length,
            norm_epsilon,
            vocab_size,
            // Hardcode for now since the JSON structure only comes from the gpt2 python script
            attention_config,
            variant,
            // only support for gpt2 for now
            eos_token: 50256usize.into(),
        })
    }

    pub fn num_groups(&self) -> usize {
        match self.attention_config.head {
            AttentionHeadType::MHA => self.num_heads,
            AttentionHeadType::GQA(num_groups) => num_groups,
        }
    }

    pub fn model(&self, l: &FileTensorLoader) -> anyhow::Result<LLMModel> {
        self.variant.model(l, self)
    }

    pub fn model_json(&self, l: &json::FileTensorLoader) -> anyhow::Result<LLMModel> {
        self.variant.model_json(l, self)
    }
    pub fn max_sequence_length(&self) -> usize {
        match self.variant {
            LLMVariant::GPT2 => self.context_length,
            LLMVariant::Gemma3 => 2048,
        }
    }
}

impl LLMVariant {
    pub fn from_loader(loader: &FileTensorLoader) -> anyhow::Result<Self> {
        let variant_name = loader
            .metadata::<String>("general.name")
            .or(loader.metadata::<String>("general.architecture"))
            .or(loader.metadata::<String>("general.basename"))
            .or(loader.metadata::<String>("general.base_model.0.name"))
            .map(|v| v.to_string())
            .context("no variant found")?;
        Self::from_name(&variant_name)
    }

    pub fn from_name(name: &str) -> anyhow::Result<Self> {
        match name.to_lowercase() {
            a if GEMMA3_VARIANTS.iter().any(|v| a.contains(v)) => Ok(Self::Gemma3),
            _ if GPT2_VARIANTS.contains(&name) => Ok(Self::GPT2),
            _ => bail!("unsupported architecture variant: {name:?}"),
        }
    }

    pub fn from_json(l: &json::FileTensorLoader) -> anyhow::Result<Self> {
        let variant_value = l
            .get_metadata("model_name")
            .ok_or_else(|| anyhow::anyhow!("Metadata key 'model_name' not found"))?;

        let model_name_str = variant_value
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Metadata 'model_name' is not a string value"))?;

        Self::from_name(model_name_str.trim())
    }
    pub fn model_json(
        &self,
        l: &json::FileTensorLoader,
        config: &LLMConfig,
    ) -> anyhow::Result<LLMModel> {
        match self {
            Self::GPT2 => Ok(LLMModel::from_json(l, config)?),
            Self::Gemma3 => bail!("Gemma3 is not supported yet"),
        }
    }

    pub fn num_heads_key(&self) -> &str {
        match self {
            Self::GPT2 => "gpt2.attention.head_count",
            Self::Gemma3 => "gemma3.attention.head_count",
        }
    }

    pub fn head_size_key(&self) -> &str {
        match self {
            Self::GPT2 => panic!("head_size_key not found for GPT2"),
            Self::Gemma3 => "gemma3.attention.key_length",
        }
    }

    pub fn context_length_key(&self) -> &str {
        match self {
            Self::GPT2 => "gpt2.context_length",
            Self::Gemma3 => "gemma3.context_length",
        }
    }
    pub fn num_block_key(&self) -> &str {
        match self {
            Self::GPT2 => "gpt2.block_count",
            Self::Gemma3 => "gemma3.block_count",
        }
    }
    pub fn embedding_size_key(&self) -> &str {
        match self {
            Self::GPT2 => "gpt2.embedding_length",
            Self::Gemma3 => "gemma3.embedding_length",
        }
    }
    pub fn hidden_size_key(&self) -> &str {
        match self {
            // same size as embedding for gpt2
            Self::GPT2 => self.embedding_size_key(),
            Self::Gemma3 => self.embedding_size_key(),
        }
    }
    pub fn norm_epsilon_key(&self) -> &str {
        match self {
            Self::GPT2 => "gpt2.attention.layer_norm_epsilon",
            Self::Gemma3 => "gemma3.attention.layer_norm_rms_epsilon",
        }
    }

    pub fn eos_token_key(&self) -> &str {
        "tokenizer.ggml.eos_token_id"
    }

    pub fn attention_num_groups_key(&self) -> Option<&str> {
        match self {
            Self::GPT2 => None,
            Self::Gemma3 => Some("gemma3.attention.head_count_kv"),
        }
    }

    pub fn norm_type(&self) -> NormType {
        match self {
            Self::GPT2 => NormType::LayerNorm,
            Self::Gemma3 => NormType::RMSNorm,
        }
    }

    pub fn has_biases(&self) -> bool {
        match self {
            Self::GPT2 => true,
            Self::Gemma3 => false,
        }
    }

    pub fn model(&self, l: &FileTensorLoader, config: &LLMConfig) -> anyhow::Result<LLMModel> {
        match self {
            Self::GPT2 => Ok(LLMModel::from_loader(l, config)?),
            Self::Gemma3 => Ok(LLMModel::from_loader(l, config)?),
        }
    }
}

impl AttentionConfig {
    pub fn from_loader(loader: &FileTensorLoader, variant: &LLMVariant) -> anyhow::Result<Self> {
        let num_attentions = loader
            .metadata::<usize>(variant.num_block_key())
            .context("num_block_key not found")?;
        let span = match variant {
            LLMVariant::GPT2 => vec![AttentionSpan::Full; num_attentions],
            // a ratio of 5:1 local vs global attention span
            LLMVariant::Gemma3 => (1..=num_attentions)
                .map(|i| match i % 6 {
                    0 => AttentionSpan::Full,
                    _ => AttentionSpan::Local(1024),
                })
                .collect(),
        };
        let head = match variant {
            LLMVariant::GPT2 => AttentionHeadType::MHA,
            LLMVariant::Gemma3 => {
                // safe unwrap because we know that the key is present in Gemma3
                let num_groups = loader
                    .metadata::<usize>(variant.attention_num_groups_key().unwrap())
                    .context(format!(
                        "not found: {}",
                        variant.attention_num_groups_key().unwrap()
                    ))?;
                AttentionHeadType::GQA(num_groups)
            }
        };
        Ok(Self { span, head })
    }
    pub fn from_json(l: &json::FileTensorLoader, variant: &LLMVariant) -> anyhow::Result<Self> {
        if let LLMVariant::Gemma3 = variant {
            bail!("Gemma3 is not supported yet for custom JSON format");
        }
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
