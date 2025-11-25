use crate::parser::{
    Load,
    llm::{HFTokenizer, models::LLMModelLoader, tokenizer::TokenizerLoader},
};
use anyhow::Context;

use crate::{
    Shape,
    layers::transformer::attention_mask::AttentionSpan,
    model::Model,
    parser::{
        ModelLoader,
        gguf::RawGGUF,
        llm::{
            LLMConfig, LLMModel,
            config::{AttentionConfig, AttentionHeadType, LLMStructure},
            transformer::{Norm, NormType},
        },
        safe::{self, RawSafeTensors},
    },
};

pub mod decoder;
use decoder::Gemma3Decoder;

/// Loader for the Gemma3 family of models.
/// For more information about Gemma3, see https://ai.google.dev/gemma/docs/core
#[derive(Clone, Debug, Default)]
pub struct Gemma3 {
    /// Current hack to avoid committing to huge rope
    max_ctx_length: Option<usize>,
}

pub const GEMMA3_NAME: &str = "gemma3";

impl Gemma3 {
    pub fn new() -> Self {
        Gemma3 {
            max_ctx_length: None,
        }
    }
    pub fn with_max_context(mut self, max_ctx_length: usize) -> Self {
        self.max_ctx_length = Some(max_ctx_length);
        self
    }

    fn scale_embeddings(llm_config: &LLMConfig, llm_model: &mut LLMModel<Gemma3Decoder>) {
        let normalizer_factor = (llm_config.hidden_size as f32).sqrt();
        llm_model.embeddings.mat = llm_model
            .embeddings
            .mat
            .new_map_tensor(|t| t.scalar_mul_f32(normalizer_factor));
    }

    fn scale_norm(rmsnorm: &mut Norm<f32>) {
        if let Norm::RMSNorm(rmsnorm) = rmsnorm {
            rmsnorm.alpha = rmsnorm
                .alpha
                .take()
                .map(|alpha| alpha.map_tensor(|t| t.map_data(|v| v + 1.0)))
        }
    }
}

pub fn is_gemma3_model(names: &[String]) -> bool {
    let is_gemma3 = names
        .iter()
        .any(|name| name.contains("gemma3") || name.contains("gemma-3"));
    let is_text_only = names.iter().any(|name| {
        name.to_lowercase().contains("270m") || name.to_lowercase().contains("gemma3_text")
    });
    is_gemma3 && is_text_only
}

impl TokenizerLoader<RawGGUF> for Gemma3 {
    fn load_tokenizer(&self, raw: &RawGGUF) -> anyhow::Result<HFTokenizer> {
        let loader = raw.loader()?;
        let tokenizer = HFTokenizer::sentencepiece_from_gguf(&loader)?;
        Ok(tokenizer)
    }
}

impl TokenizerLoader<RawSafeTensors> for Gemma3 {
    fn load_tokenizer(&self, raw: &RawSafeTensors) -> anyhow::Result<HFTokenizer> {
        let tokenizer = HFTokenizer::from_tokenizer_json_path(raw.tokenizer_path())?;
        Ok(tokenizer)
    }
}

impl<DataFormat> LLMModelLoader<DataFormat> for Gemma3
where
    Gemma3: ModelLoader<DataFormat, ModelConfig = LLMConfig>,
{
    fn with_max_context_length(self, max_ctx_length: usize) -> Self
    where
        Self: Sized,
    {
        self.with_max_context(max_ctx_length)
    }
}

impl ModelLoader<RawGGUF> for Gemma3 {
    type ModelConfig = LLMConfig;

    fn model_name(&self) -> String {
        GEMMA3_NAME.to_string()
    }

    fn parse(&self, raw: &RawGGUF) -> anyhow::Result<(Model<f32>, Self::ModelConfig)> {
        let loader = raw.loader()?;
        let config = LLMConfig::from_gguf(&loader, "gemma3")?;

        let sliding_window = loader
            .metadata::<usize>(&loader.sliding_window_key("gemma3"))
            .ok_or(anyhow::anyhow!("sliding window key not found"))?;

        let span = (1..=config.num_block)
            .map(|i| match i % 6 {
                0 => AttentionSpan::Full,
                _ => AttentionSpan::Local(sliding_window),
            })
            .collect();
        let head = AttentionHeadType::GQA(
            loader
                .metadata::<usize>(&loader.num_kv_heads_key("gemma3"))
                .ok_or(anyhow::anyhow!("attention num groups key not found"))?,
        );

        let structure = LLMStructure {
            generic: config.clone(),
            norm_type: NormType::RMSNorm,
            global_positional: None,
            attention_config: AttentionConfig { span, head },
        };

        let mut model = LLMModel::<Gemma3Decoder>::from_loader(&loader, &structure)?;
        // Gemma3 we have to rescale the embedding matrix
        Self::scale_embeddings(&config, &mut model);
        Self::scale_norm(&mut model.final_norm);
        // even though the llm runtime doesn't care about the model input shape, which is designed for "static" input shapes, we still
        // need to provide one.
        let init_user_shape = Shape::from(vec![1]);
        let model = model.into_provable_model(init_user_shape)?;
        Ok((model, config))
    }
}

impl ModelLoader<RawSafeTensors> for Gemma3 {
    type ModelConfig = LLMConfig;

    fn model_name(&self) -> String {
        GEMMA3_NAME.to_string()
    }

    fn parse(&self, raw: &RawSafeTensors) -> anyhow::Result<(Model<f32>, Self::ModelConfig)> {
        // Read HF config.json
        let cfg = raw.read_config_json()?;
        let hidden_size = cfg
            .get::<usize, _>("hidden_size")
            .context("hidden_size not found")?;
        let embedding_size = hidden_size;
        let num_heads = cfg
            .get::<usize, _>("num_attention_heads")
            .context("num_attention_heads not found")?;
        let head_size = cfg
            .get::<usize, _>("head_dim")
            .context("head_dim not found")?;
        let num_block = cfg
            .get::<usize, _>("num_hidden_layers")
            .context("num_hidden_layers not found")?;
        let context_length = cfg
            .get::<usize, _>("max_position_embeddings")
            .context("max_position_embeddings not found")?;
        let norm_epsilon = cfg
            .get::<f32, _>("rms_norm_eps")
            .context("rms_norm_eps not found")?;
        let vocab_size = cfg
            .get::<usize, _>("vocab_size")
            .context("vocab_size not found")?;
        let eos_token = cfg
            .get::<u64, _>("eos_token_id")
            .context("eos_token_id not found")?
            .into();

        let llm_config = LLMConfig {
            model_name: "gemma3".to_string(),
            embedding_size,
            hidden_size,
            num_heads,
            head_size,
            num_block,
            context_length,
            norm_epsilon,
            vocab_size,
            eos_token,
        };

        // Structure: Gemma3 uses RMSNorm + RoPE and no final projection
        let num_groups = cfg
            .get::<usize, _>("num_key_value_heads")
            .context("num_key_value_heads not found")?;
        let raw_spans = cfg
            .get::<Vec<String>, _>("layer_types")
            .context("layer_types not found")?;
        let sliding_window = cfg
            .get::<usize, _>("sliding_window")
            .context("sliding_window not found")?;
        let spans = raw_spans
            .iter()
            .map(|s| {
                if s.contains("sliding_attention") {
                    AttentionSpan::Local(sliding_window)
                } else {
                    AttentionSpan::Full
                }
            })
            .collect();
        let structure = LLMStructure {
            generic: llm_config.clone(),
            norm_type: NormType::RMSNorm,
            global_positional: None,
            attention_config: AttentionConfig {
                span: spans,
                head: AttentionHeadType::GQA(num_groups),
            },
        };

        let loader = safe::FileTensorLoader::from_path(raw.model_path())?;
        let mut llm_model = LLMModel::<Gemma3Decoder>::from_loader(&loader, &(structure, cfg))?;
        // Gemma3 we have to rescale the embedding matrix
        Self::scale_embeddings(&llm_config, &mut llm_model);
        Self::scale_norm(&mut llm_model.final_norm);
        let init_user_shape = Shape::from(vec![1]);
        let model = llm_model.into_provable_model(init_user_shape)?;
        Ok((model, llm_config))
    }
}

#[cfg(test)]
pub mod tests {
    use std::{collections::HashMap, fs::File, io::BufReader};

    use anyhow::{Context, ensure};
    use ff_ext::GoldilocksExt2;
    use serde::{Deserialize, Serialize};
    use tenstore::GenStore;

    use crate::{
        Tensor,
        graph::{Direction, NodeId},
        layers::{Layer, provable::OpInfo, transformer::logits::Logits},
        parser::{file_cache, llm::LLMTokenizer},
    };

    use super::*;

    pub const GEMMA3_Q8: &str = "gemma-3-270m-it-Q8_0.gguf";

    // Tolerance for QKV comparisons - slightly relaxed to account for floating-point precision
    // differences between PyTorch and Burn implementations
    const QKV_TOLERANCE: f32 = 3e-4;

    #[derive(Debug, Clone, Deserialize)]
    struct Gemma3Matrix {
        /// Row-major shape of the serialized tensor.
        shape: Vec<usize>,
        /// Flattened tensor data in row-major order.
        data: Vec<f32>,
    }

    #[derive(Debug, Clone, Deserialize)]
    struct Gemma3QkvLayer {
        /// Query activations captured from the Python trace.
        q: Gemma3Matrix,
        /// Key activations captured from the Python trace.
        k: Gemma3Matrix,
        /// Value activations captured from the Python trace.
        v: Gemma3Matrix,
    }

    #[derive(Debug, Clone, Deserialize)]
    struct Gemma3QkNormalized {
        /// Q after q_norm, pre-RoPE.
        #[serde(default)]
        q: Option<Gemma3Matrix>,
        /// K after k_norm, pre-RoPE.
        #[serde(default)]
        k: Option<Gemma3Matrix>,
    }

    #[derive(Debug, Clone, Deserialize)]
    struct Gemma3Intermediates {
        /// Embedding activations captured from the Python trace.
        #[serde(default)]
        embeddings: Option<Gemma3Matrix>,
        /// Output of input RMSNorm before QKV projection (first layer only).
        #[serde(default)]
        pre_qkv: Vec<Gemma3Matrix>,
        /// List of attention-layer QKV tensors (first layer only).
        #[serde(default)]
        attention_qkv: Vec<Gemma3QkvLayer>,
        /// Q/K after q_norm/k_norm (pre-RoPE), first layer only.
        #[serde(default)]
        attention_qkv_normalized: Vec<Gemma3QkNormalized>,
        /// Q, K, V after RoPE is applied (first layer only).
        #[serde(default)]
        attention_qkv_rope: Vec<Gemma3QkvLayer>,
        // (post-RoPE post-norm is ignored for now; will be parsed via _rest if present)
        /// Static embedding vector for token 0 from the embedding weight matrix.
        #[serde(default)]
        static_embedding_0: Option<Vec<f32>>,
        /// RMSNorm alpha (weight) tensor for the first pre-QKV RMSNorm layer.
        #[serde(default)]
        pre_qkv_rmsnorm_alpha: Option<Vec<f32>>,
        /// Any additional fields emitted by the Python trace.
        #[serde(default, flatten)]
        _rest: HashMap<String, serde_json::Value>,
    }

    #[derive(Debug, Clone, Deserialize)]
    struct Gemma3TraceEntry {
        /// Tokenized input sequence used for the trace example.
        input_token: Vec<u32>,
        /// Optional intermediates emitted when running with `--trace`.
        #[serde(default)]
        intermediates: Option<Gemma3Intermediates>,
    }

    /// Extract the first `num_cols` columns from a 2D tensor
    fn extract_first_cols(tensor: &Tensor<f32>, num_cols: usize) -> anyhow::Result<Tensor<f32>> {
        let shape = tensor.shape();
        ensure!(
            shape.rank() == 2,
            "extract_first_cols requires rank-2 tensor, got {:?}",
            shape
        );
        let rows = shape.dim(0);
        let total_cols = shape.dim(1);
        ensure!(
            num_cols <= total_cols,
            "Cannot extract {} columns from tensor with {} columns",
            num_cols,
            total_cols
        );

        let data = tensor.get_data();
        let mut result = Vec::with_capacity(rows * num_cols);
        for r in 0..rows {
            let row_start = r * total_cols;
            result.extend_from_slice(&data[row_start..row_start + num_cols]);
        }

        Tensor::new(Shape::from(vec![rows, num_cols]), result)
    }

    fn assert_matrices_close(
        label: &str,
        actual: &Tensor<f32>,
        expected: &Gemma3Matrix,
    ) -> anyhow::Result<()> {
        let shape = actual.shape();
        ensure!(
            shape.rank() == 2,
            "{label} output must be rank-2, got {:?}",
            shape
        );

        ensure!(
            expected.shape.len() == 2,
            "{label} serialized matrix shape must have rank 2, got {:?}",
            expected.shape
        );
        let rows = expected.shape[0];
        let cols = expected.shape[1];
        ensure!(
            rows > 0 && cols > 0,
            "{label} expected matrix must be non-empty (rows={rows}, cols={cols})"
        );
        ensure!(
            shape.dim(0) == rows,
            "{label} row count mismatch: actual={}, expected={rows}",
            shape.dim(0)
        );
        ensure!(
            shape.dim(1) == cols,
            "{label} column count mismatch: actual={}, expected={cols}",
            shape.dim(1)
        );

        let stride = shape.dim(1);
        let data = actual.get_data();
        ensure!(
            expected.data.len() == rows * cols,
            "{label} serialized data length {} does not match shape product {}",
            expected.data.len(),
            rows * cols
        );
        ensure!(
            data.len() == rows * cols,
            "{label} tensor data length {} does not match shape product {}",
            data.len(),
            rows * cols
        );
        let mut max_abs = 0f32;

        for r in 0..rows {
            for c in 0..cols {
                let idx = r * stride + c;
                let diff = (data[idx] - expected.data[idx]).abs();
                if diff > max_abs {
                    max_abs = diff;
                }
            }
        }

        ensure!(
            max_abs <= QKV_TOLERANCE,
            "{label} diff too large: max_abs_diff={max_abs:.3e} (tol={tol:.1e})\n first few values actual: {actual:?}\n first few values expected: {expected:?}",
            label = label,
            max_abs = max_abs,
            tol = QKV_TOLERANCE,
            actual = &data[..data.len().min(10)],
            expected = &expected.data[..expected.data.len().min(10)]
        );

        Ok(())
    }

    #[test]
    fn test_gguf_gemma3_load_tokenizer() -> anyhow::Result<()> {
        let model_path = file_cache::from_cache(GEMMA3_Q8)?;
        let mygguf = RawGGUF::new(model_path);
        let tokenizer = Gemma3::new().load_tokenizer(&mygguf)?;
        let s = "do or don't. there is no try.";
        let tokens = tokenizer.tokenize(s);
        let s2 = tokenizer.detokenize(&tokens);
        assert_eq!(s, s2);
        Ok(())
    }

    #[test]
    fn test_gguf_gemma3_load_model() -> anyhow::Result<()> {
        let model_path = file_cache::from_cache(GEMMA3_Q8)?;
        let mygguf = RawGGUF::new(model_path);
        let (model, config) = Gemma3::new().with_max_context(2048).parse(&mygguf)?;
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.num_block, 18);
        assert_eq!(config.embedding_size, 640);
        assert_eq!(config.hidden_size, 640);
        assert_eq!(config.context_length, 32768);
        assert_eq!(config.norm_epsilon, 1e-6);
        assert_eq!(config.vocab_size, 262144);
        assert_eq!(config.eos_token, 106usize.into());
        let input = Tensor::new(vec![1].into(), vec![1562_f32])?;
        model.run_float(vec![input], &mut GenStore::default())?;
        Ok(())
    }

    #[test]
    fn test_gguf_gemma3_logits() -> anyhow::Result<()> {
        let raw = RawSafeTensors::from_hugging_face_cached(super::safe_tests::GEMMA3_SAFE_MODEL)?;
        let (model, _config) = Gemma3::new().with_max_context(2048).parse(&raw)?;
        let (argmax_layer_id, argmax) = model.graph.inner_nodes().last().unwrap();
        ensure!(matches!(argmax, Layer::Logits(Logits::Argmax)));
        // we load the json file that was generated by the python script
        let logits_path = "assets/scripts/llms/gemma3_logits.json";
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct GemmaLogits {
            input_token: Vec<u32>,
            input_text: String,
            logits: Vec<f32>,
        }
        let argmax = |logits: &[f32]| -> usize {
            let max_index = logits
                .iter()
                .enumerate()
                .max_by(|(_, &x), (_, &y)| x.partial_cmp(&y).unwrap())
                .unwrap()
                .0;
            max_index
        };

        let logits: Vec<GemmaLogits> =
            serde_json::from_reader(BufReader::new(File::open(logits_path)?))?;
        for logit in logits.iter() {
            let input_shape = Shape::from(vec![logit.input_token.len()]);
            // let padded_shape = input_shape.next_power_of_two();
            let input = Tensor::new(
                input_shape.clone(),
                logit.input_token.iter().map(|x| *x as f32).collect(),
            )?;
            let mut store = GenStore::default();
            let trace = model.run::<GoldilocksExt2>(vec![input], &mut store)?;
            let deep_prove_logits = trace
                .get_step(&argmax_layer_id)
                .ok_or(anyhow::anyhow!(
                    "Failed to get logits at step: {argmax_layer_id}"
                ))?
                .input_tensors()?[0]
                .clone();
            let vocab_size = logit.logits.len();
            let total_logits = deep_prove_logits.dim(0);
            let skip = (total_logits - 1) * vocab_size;
            // We only care about the logits of the final token
            let computed_new_token = argmax(&deep_prove_logits.get_data()[skip..]);
            let expected_new_token = argmax(&logit.logits);
            // CURRENTLY NOT WORKING BECAUSE GEMMA3 CONSTRUCTION IS INCORRECT
            assert!(
                computed_new_token == expected_new_token,
                "computed_new_token: {:?}, expected_new_token: {:?}; for input: {:?}",
                computed_new_token,
                expected_new_token,
                logit.input_text
            );
            model.reset();
        }
        Ok(())
    }

    #[test]
    fn test_safe_gemma3_intermediates() -> anyhow::Result<()> {
        use crate::layers::Layer;
        use std::path::Path;

        let json_path = {
            let manifest_dir = env!("CARGO_MANIFEST_DIR");
            Path::new(manifest_dir).join("assets/scripts/llms/gemma3_trace.json")
        };

        let entries: Vec<Gemma3TraceEntry> =
            serde_json::from_reader(BufReader::new(File::open(&json_path)?))?;
        let python = entries
            .first()
            .context("gemma3_trace.json has no entries; rerun the Python script with --trace")?;
        ensure!(
            !python.input_token.is_empty(),
            "Python trace must contain at least one input token"
        );
        let intermediates = python.intermediates.as_ref().context(
            "gemma3_trace.json missing intermediates; rerun the Python script with --trace",
        )?;

        ensure!(
            intermediates.attention_qkv.len() == 1,
            "expected exactly one attention layer in Python trace, found {}",
            intermediates.attention_qkv.len()
        );
        let python_qkv = intermediates
            .attention_qkv
            .first()
            .expect("length checked above");

        let seq_len = python.input_token.len();
        ensure!(
            python_qkv.q.shape.len() == 2,
            "Python Q shape must have rank 2, got {:?}",
            python_qkv.q.shape
        );
        ensure!(
            python_qkv.k.shape.len() == 2,
            "Python K shape must have rank 2, got {:?}",
            python_qkv.k.shape
        );
        ensure!(
            python_qkv.v.shape.len() == 2,
            "Python V shape must have rank 2, got {:?}",
            python_qkv.v.shape
        );
        ensure!(
            python_qkv.q.shape[0] == seq_len,
            "Python Q rows ({}) must match input token length ({seq_len})",
            python_qkv.q.shape[0]
        );
        ensure!(
            python_qkv.k.shape[0] == seq_len,
            "Python K rows ({}) must match input token length ({seq_len})",
            python_qkv.k.shape[0]
        );
        ensure!(
            python_qkv.v.shape[0] == seq_len,
            "Python V rows ({}) must match input token length ({seq_len})",
            python_qkv.v.shape[0]
        );
        ensure!(
            python_qkv.q.shape[1] > 0,
            "Python Q must contain at least one column"
        );
        ensure!(
            python_qkv.k.shape[1] > 0,
            "Python K must contain at least one column"
        );
        ensure!(
            python_qkv.v.shape[1] > 0,
            "Python V must contain at least one column"
        );
        ensure!(
            python_qkv.q.data.len() == python_qkv.q.shape[0] * python_qkv.q.shape[1],
            "Python Q data length {} does not match shape {:?}",
            python_qkv.q.data.len(),
            python_qkv.q.shape
        );
        ensure!(
            python_qkv.k.data.len() == python_qkv.k.shape[0] * python_qkv.k.shape[1],
            "Python K data length {} does not match shape {:?}",
            python_qkv.k.data.len(),
            python_qkv.k.shape
        );
        ensure!(
            python_qkv.v.data.len() == python_qkv.v.shape[0] * python_qkv.v.shape[1],
            "Python V data length {} does not match shape {:?}",
            python_qkv.v.data.len(),
            python_qkv.v.shape
        );

        let raw = RawSafeTensors::from_hugging_face_cached(super::safe_tests::GEMMA3_SAFE_MODEL)?;
        let (model, _config) = Gemma3::new().with_max_context(2048).parse(&raw)?;

        // Verify tokens match exactly
        println!("Python input tokens: {:?}", python.input_token);
        let rust_input_tokens: Vec<f32> = python
            .input_token
            .iter()
            .map(|&token| token as f32)
            .collect();
        println!("Rust input tokens (as f32): {:?}", rust_input_tokens);

        let input = Tensor::new(
            Shape::from(vec![python.input_token.len()]),
            rust_input_tokens.clone(),
        )?;

        let mut store = GenStore::default();
        let trace = model.run::<GoldilocksExt2>(vec![input], &mut store)?;

        // Compare embedding outputs
        let python_embeddings = intermediates
            .embeddings
            .as_ref()
            .context("Python trace missing embeddings; rerun the script with --trace")?;
        ensure!(
            python_embeddings.shape.len() == 2,
            "Python embeddings shape must have rank 2, got {:?}",
            python_embeddings.shape
        );
        let num_tokens = python_embeddings.shape[0];
        let hidden_size = python_embeddings.shape[1];
        ensure!(
            num_tokens == seq_len,
            "Python embeddings rows ({}) must match input token length ({})",
            num_tokens,
            seq_len
        );
        ensure!(
            python_embeddings.data.len() == seq_len * hidden_size,
            "Python embeddings data length {} does not match shape {:?}",
            python_embeddings.data.len(),
            python_embeddings.shape
        );

        let (embedding_node, embedding_ref) = model
            .graph
            .forward_inners()
            .find(|(_, layer)| matches!(layer, Layer::Embeddings(_)))
            .context("Gemma3 trace missing an embeddings layer")?;
        let embedding_step = trace
            .get_step(&embedding_node)
            .expect("embedding node id discovered in previous search");
        let embedding_outputs = embedding_step.output_tensors()?;
        ensure!(
            embedding_outputs.len() == 1,
            "Embeddings layer must emit exactly one tensor"
        );

        // Debug: Print first row of embeddings from both Python and Rust
        let rust_emb = &embedding_outputs[0];
        let rust_shape = rust_emb.shape();
        let rust_data = rust_emb.get_data();
        let rust_embedding_output_first_row: Vec<f32> = rust_data[..hidden_size].to_vec();

        let python_first_row: Vec<f32> = python_embeddings.data[..hidden_size].to_vec();
        let first_token_id = python.input_token[0] as usize;

        println!("\n=== Embeddings First Row Comparison ===");
        println!("Python shape: {:?}", python_embeddings.shape);
        println!("Rust shape: {:?}", rust_shape);
        println!("First token ID: {}", first_token_id);
        println!(
            "Python Embeddings output first token (first 10 values): {:?}",
            &python_first_row[..python_first_row.len().min(10)]
        );
        println!(
            "Rust Embeddings output first token (first 10 values): {:?}",
            &rust_embedding_output_first_row[..rust_embedding_output_first_row.len().min(10)]
        );

        // Also check if we can access the embedding matrix weights directly
        if let Layer::Embeddings(emb) = embedding_ref {
            let emb_matrix = emb.embedding_matrix();
            let emb_shape = emb_matrix.shape();
            let emb_data = emb_matrix.get_data();
            println!("\n=== Embedding Matrix Weight Comparison ===");
            println!(
                "Embedding matrix shape: {:?} (should be [vocab_size, hidden_dim])",
                emb_shape
            );

            // Compare static embedding for token 0 from Python vs Rust
            if let Some(ref python_static_emb0) = intermediates.static_embedding_0 {
                ensure!(
                    python_static_emb0.len() == hidden_size,
                    "Python static_embedding_0 length {} does not match hidden_size {}",
                    python_static_emb0.len(),
                    hidden_size
                );

                let rust_row0: Vec<f32> = emb_data[..hidden_size].to_vec();
                println!(
                    "Python static_embedding_0 (first 10): {:?}",
                    &python_static_emb0[..python_static_emb0.len().min(10)]
                );
                println!(
                    "Rust embedding matrix row 0 (first 10): {:?}",
                    &rust_row0[..rust_row0.len().min(10)]
                );

                let matches = python_static_emb0
                    .iter()
                    .zip(rust_row0.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-5);
                println!("Static embedding row 0 matches: {}", matches);

                if !matches {
                    let diff: Vec<f32> = python_static_emb0
                        .iter()
                        .zip(rust_row0.iter())
                        .map(|(a, b)| (a - b).abs())
                        .collect();
                    let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);
                    println!("Max diff in row 0: {:.6e}", max_diff);
                    println!("First 10 diffs: {:?}", &diff[..diff.len().min(10)]);
                }
            } else {
                println!("WARNING: Python trace missing static_embedding_0");
            }

            // The embedding matrix is [vocab_size, hidden_dim], so row i corresponds to token i
            if first_token_id < emb_shape.dim(0) {
                let token_row_start = first_token_id * hidden_size;
                let token_row_end = token_row_start + hidden_size;
                let rust_manual_embedding_output: Vec<f32> =
                    emb_data[token_row_start..token_row_end].to_vec();
                println!(
                    "\nToken {} embedding row from Rust weight matrix (first 10): {:?}",
                    first_token_id,
                    &rust_manual_embedding_output[..rust_manual_embedding_output.len().min(10)]
                );

                // Check if this matches Rust output
                let matches_rust = rust_manual_embedding_output
                    .iter()
                    .zip(rust_embedding_output_first_row.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-5);
                println!(
                    "Token {} row from weight matrix matches Rust embedding output: {}",
                    first_token_id, matches_rust
                );
                if !matches_rust {
                    let diff: Vec<f32> = rust_manual_embedding_output
                        .iter()
                        .zip(rust_embedding_output_first_row.iter())
                        .map(|(a, b)| (a - b).abs())
                        .collect();
                    let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);
                    println!("Max diff: {:.6e}", max_diff);
                    println!("First 10 diffs: {:?}", &diff[..diff.len().min(10)]);
                }
            }
        }

        // Try transpose comparison if shapes allow
        if rust_shape.dim(0) == hidden_size && rust_shape.dim(1) == seq_len {
            println!("WARNING: Shapes suggest possible transpose! Trying transpose comparison...");
            // Would need to transpose rust_emb for comparison
        }

        assert_matrices_close("Embeddings", &embedding_outputs[0], python_embeddings)?;

        // Compare pre-QKV RMSNorm output (first attention layer's input layernorm)
        ensure!(
            intermediates.pre_qkv.len() == 1,
            "expected exactly one pre_qkv layer in Python trace, found {}",
            intermediates.pre_qkv.len()
        );
        let python_pre_qkv = intermediates.pre_qkv.first().expect("length checked above");

        // Find the first RMSNorm node after embeddings (this is the input_layernorm before first QKV)
        let mut found_embeddings = false;
        let (pre_qkv_norm_node, pre_qkv_norm_layer) = model
            .graph
            .forward_inners()
            .find(|(_, layer)| {
                if found_embeddings && matches!(layer, Layer::RMSNorm(_)) {
                    true
                } else if matches!(layer, Layer::Embeddings(_)) {
                    found_embeddings = true;
                    false
                } else {
                    false
                }
            })
            .context("Gemma3 trace missing pre-QKV RMSNorm layer")?;

        let pre_qkv_norm_step = trace
            .get_step(&pre_qkv_norm_node)
            .expect("pre-QKV norm node id discovered in previous search");
        let pre_qkv_norm_outputs = pre_qkv_norm_step.output_tensors()?;
        ensure!(
            pre_qkv_norm_outputs.len() == 1,
            "Pre-QKV RMSNorm layer must emit exactly one tensor"
        );

        println!("\n=== Pre-QKV RMSNorm Comparison ===");
        println!("Python pre_qkv shape: {:?}", python_pre_qkv.shape);
        println!("Rust pre_qkv shape: {:?}", pre_qkv_norm_outputs[0].shape());

        // Compare RMSNorm alpha (weight) tensors
        if let Some(ref python_rmsnorm_alpha) = intermediates.pre_qkv_rmsnorm_alpha {
            if let Layer::RMSNorm(rmsnorm) = pre_qkv_norm_layer {
                if let Some(ref rust_alpha_tensor) = rmsnorm.alpha {
                    let rust_alpha_data = rust_alpha_tensor.get_data();
                    println!("\n=== Pre-QKV RMSNorm Alpha (Weight) Comparison ===");
                    println!("Python alpha length: {}", python_rmsnorm_alpha.len());
                    println!("Rust alpha length: {}", rust_alpha_data.len());
                    println!(
                        "Python alpha (first 10): {:?}",
                        &python_rmsnorm_alpha[..python_rmsnorm_alpha.len().min(10)]
                    );
                    println!(
                        "Rust alpha (first 10): {:?}",
                        &rust_alpha_data[..rust_alpha_data.len().min(10)]
                    );

                    ensure!(
                        python_rmsnorm_alpha.len() == rust_alpha_data.len(),
                        "RMSNorm alpha length mismatch: Python={}, Rust={}",
                        python_rmsnorm_alpha.len(),
                        rust_alpha_data.len()
                    );

                    let matches = python_rmsnorm_alpha
                        .iter()
                        .zip(rust_alpha_data.iter())
                        // +1 since gemma3 adds + 1 during evaluation and we just add + 1 during
                        // initialization
                        .all(|(a, b)| ((a + 1.0) - b).abs() < 1e-5);
                    println!("RMSNorm alpha matches: {}", matches);

                    if !matches {
                        let diff: Vec<f32> = python_rmsnorm_alpha
                            .iter()
                            .zip(rust_alpha_data.iter())
                            .map(|(a, b)| ((a + 1.0) - b).abs())
                            .collect();
                        let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);
                        println!("Max diff in alpha: {:.6e}", max_diff);
                        println!("First 10 diffs: {:?}", &diff[..diff.len().min(10)]);
                    }
                } else {
                    println!("WARNING: Rust RMSNorm has no alpha tensor");
                }
            }
        } else {
            println!("WARNING: Python trace missing pre_qkv_rmsnorm_alpha");
        }

        assert_matrices_close("Pre-QKV RMSNorm", &pre_qkv_norm_outputs[0], python_pre_qkv)?;

        println!("\n✓ Pre-QKV RMSNorm output matches!");
        println!("=== QKV PROJECTION COMPARISON ===");

        let (first_qkv_node, _) = model
            .graph
            .forward_inners()
            .find(|(_, layer)| {
                layer.describe() == "EinSum(X(se)@WQ(ehd):WK(ed):WV(ed)->Q(hsd):K(sd):V(sd))"
            })
            .context("Gemma3 trace missing a QKV layer")?;

        let step = trace
            .get_step(&first_qkv_node)
            .expect("node id discovered in previous search");
        let outputs = step.output_tensors()?;
        assert_eq!(outputs.len(), 3, "QKV layer must emit three outputs");

        println!("Python Q shape: {:?}", python_qkv.q.shape);
        println!("Rust Q shape: {:?}", outputs[0].shape());
        println!("Python K shape: {:?}", python_qkv.k.shape);
        println!("Rust K shape: {:?}", outputs[1].shape());
        println!("Python V shape: {:?}", python_qkv.v.shape);
        println!("Rust V shape: {:?}\n", outputs[2].shape());

        println!("Comparing Q projection...");
        assert_matrices_close(
            "Q",
            &outputs[0]
                .permute3d(&[1, 0, 2])?
                .reshaped(python_qkv.q.shape.clone().into())?,
            &python_qkv.q,
        )?;
        println!("✓ Q projection matches!\n");

        // For K and V, we need to account for GQA expansion in Rust
        // Python captures true GQA dimensions [seq_len, kv_size]
        // Rust expands by num_heads to fake GQA: [seq_len, kv_size * num_heads]
        // We compare only the first kv_size columns since they're repeated
        let kv_size = python_qkv.k.shape[1];
        ensure!(
            kv_size == python_qkv.v.shape[1],
            "K and V must have the same feature dimension"
        );

        println!(
            "Comparing K projection (first {} of {} columns due to GQA expansion)...",
            kv_size,
            outputs[1].shape().dim(1)
        );
        let rust_k_slice = extract_first_cols(&outputs[1], kv_size)?;
        assert_matrices_close("K", &rust_k_slice, &python_qkv.k)?;
        println!("✓ K projection matches!\n");

        println!(
            "Comparing V projection (first {} of {} columns due to GQA expansion)...",
            kv_size,
            outputs[2].shape().dim(1)
        );
        let rust_v_slice = extract_first_cols(&outputs[2], kv_size)?;
        assert_matrices_close("V", &rust_v_slice, &python_qkv.v)?;
        println!("✓ V projection matches!\n");

        // ===== Q/K NORM COMPARISON =====
        println!("=== Q/K NORM COMPARISON ===");
        let is_positional = |op: &Layer<f32>| matches!(op, Layer::Positional(_));
        let is_norm = |op: &Layer<f32>| matches!(op, Layer::RMSNorm(_));
        #[allow(clippy::type_complexity)]
        let next_node =
            |node_id: NodeId, is_layer: Box<dyn Fn(&Layer<f32>) -> bool>| -> Vec<NodeId> {
                model
                    .graph
                    .neighbors(node_id, Direction::Outgoing)
                    .filter_map(|(_, edge)| {
                        model
                            .graph
                            .node(edge.target())
                            .and_then(|node| node.as_inner())
                            .and_then(|layer| is_layer(layer).then(|| edge.target()))
                    })
                    .collect()
            };

        let norm_nodes = next_node(first_qkv_node, Box::new(is_norm));
        ensure!(
            norm_nodes.len() == 2,
            "Expected exactly two norm nodes as child of QKV, found {}",
            norm_nodes.len()
        );
        let q_norm_node = norm_nodes[0];
        let k_norm_node = norm_nodes[1];

        let q_norm_output = trace
            .get_step(&q_norm_node)
            .expect("Q norm node id discovered in previous search")
            .output_tensors()?;
        let k_norm_output = trace
            .get_step(&k_norm_node)
            .expect("K norm node id discovered in previous search")
            .output_tensors()?;
        ensure!(
            q_norm_output.len() == 1 && k_norm_output.len() == 1,
            "Each Q/K norm must emit one tensor"
        );

        if let Some(python_qk_norm) = intermediates.attention_qkv_normalized.first() {
            if let Some(ref expected_q) = python_qk_norm.q {
                println!(
                    "Python Q norm shape: {:?} | Rust Q norm shape: {:?}",
                    expected_q.shape,
                    q_norm_output[0].shape()
                );
                assert_matrices_close(
                    "Q norm",
                    &q_norm_output[0]
                        .permute3d(&[1, 0, 2])?
                        .reshaped(expected_q.shape.clone().into())?,
                    expected_q,
                )?;
                println!("✓ Q norm matches!\n");
            } else {
                println!(
                    "WARNING: Python trace missing Q in attention_qkv_normalized; skipping Q norm compare"
                );
            }

            if let Some(ref expected_k) = python_qk_norm.k {
                println!(
                    "Python K norm shape: {:?} | Rust K norm shape: {:?}",
                    expected_k.shape,
                    k_norm_output[0].shape()
                );
                // Account for Rust GQA expansion on K by slicing first kv_size columns
                let rust_k_norm_slice = extract_first_cols(&k_norm_output[0], expected_k.shape[1])?;
                assert_matrices_close("K norm", &rust_k_norm_slice, expected_k)?;
                println!("✓ K norm matches!\n");
            } else {
                println!(
                    "WARNING: Python trace missing K in attention_qkv_normalized; skipping K norm compare"
                );
            }
        } else {
            println!(
                "WARNING: Python trace missing attention_qkv_normalized; skipping Q/K norm compare"
            );
        }

        // ===== ROPE COMPARISON =====
        println!("=== ROPE (Rotary Position Embedding) COMPARISON ===");

        let python_rope = intermediates
            .attention_qkv_rope
            .first()
            .context("Missing attention_qkv_rope in Python trace")?;

        // Find the two Positional (RoPE) children of the q and k norm
        // first identify the reshape nodes, and then the rope nodes
        let q_rope_out = next_node(q_norm_node, Box::new(is_positional))[0];
        let k_rope_out = next_node(k_norm_node, Box::new(is_positional))[0];

        let rope_nodes = [q_rope_out, k_rope_out];

        // Deterministic mapping: first child = Q RoPE, second = K RoPE
        let q_rope_step = trace
            .get_step(&rope_nodes[0])
            .expect("Q RoPE node id discovered in previous search");
        let k_rope_step = trace
            .get_step(&rope_nodes[1])
            .expect("K RoPE node id discovered in previous search");
        let q_rope_outs = q_rope_step.output_tensors()?;
        let k_rope_outs = k_rope_step.output_tensors()?;
        ensure!(
            q_rope_outs.len() == 1 && k_rope_outs.len() == 1,
            "Each RoPE must emit one tensor"
        );

        println!("Python Q (post-RoPE) shape: {:?}", python_rope.q.shape);
        println!("Rust Q (post-RoPE) shape: {:?}\n", q_rope_outs[0].shape());

        println!("Python K (post-RoPE) shape: {:?}", python_rope.k.shape);
        println!("Rust K (post-RoPE) shape: {:?}\n", k_rope_outs[0].shape());

        println!(
            "------- RUST Q (post-RoPE) [0:2] = {:?}",
            &q_rope_outs[0].get_data()[256..256 + 10]
        );

        // Compare Q post-RoPE
        println!("Comparing Q (post-RoPE)...");
        assert_matrices_close(
            "Q (post-RoPE)",
            &q_rope_outs[0]
                .permute3d(&[1, 0, 2])?
                .reshaped(python_rope.q.shape.clone().into())?,
            &python_rope.q,
        )?;
        println!("✓ Q (post-RoPE) matches!\n");

        // Compare K post-RoPE (need to slice first kv_size columns due to GQA) with the other node
        println!(
            "Comparing K (post-RoPE, first {} of {} columns due to GQA expansion)...",
            kv_size,
            k_rope_outs[0].shape().dim(1)
        );
        let rust_k_rope_slice = extract_first_cols(&k_rope_outs[0], kv_size)?;
        assert_matrices_close("K (post-RoPE)", &rust_k_rope_slice, &python_rope.k)?;
        println!("✓ K (post-RoPE) matches!\n");

        println!("=== ALL TESTS PASSED ===");

        Ok(())
    }
}

#[cfg(test)]
pub mod safe_tests {
    use tenstore::GenStore;

    use super::*;
    use crate::{Tensor, parser::llm::LLMTokenizer};

    pub const GEMMA3_SAFE_MODEL: &str = "google/gemma-3-270m-it";

    #[test]
    fn test_safe_gemma3_load_tokenizer() -> anyhow::Result<()> {
        let raw = RawSafeTensors::from_hugging_face_cached(GEMMA3_SAFE_MODEL)?;
        let tokenizer = Gemma3::new().load_tokenizer(&raw)?;
        let tokens = tokenizer.tokenize("Hello, world!");
        let s = tokenizer.detokenize(&tokens);
        assert_eq!(s, "Hello, world!");
        Ok(())
    }

    #[test]
    fn test_safe_gemma3_load_model() -> anyhow::Result<()> {
        let raw = RawSafeTensors::from_hugging_face_cached(GEMMA3_SAFE_MODEL)?;
        let (model, config) = Gemma3::new().with_max_context(2048).parse(&raw)?;
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.num_block, 18);
        assert_eq!(config.embedding_size, 640);
        assert_eq!(config.hidden_size, 640);
        assert_eq!(config.context_length, 32768);
        assert_eq!(config.norm_epsilon, 1e-6);
        assert_eq!(config.vocab_size, 262144);
        assert_eq!(config.eos_token, 1usize.into());
        let input = Tensor::new(vec![1].into(), vec![1562_f32])?;
        model.run_float(vec![input], &mut GenStore::default())?;
        Ok(())
    }
}
