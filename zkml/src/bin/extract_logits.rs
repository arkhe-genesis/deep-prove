#![allow(clippy::print_stderr)]
#![allow(clippy::print_stdout)]

use anyhow::{Context, Result, bail};
use clap::Parser;
use ff_ext::{ExtensionField, GoldilocksExt2};
use serde::Serialize;
use std::path::{Path, PathBuf};
use tenstore::GenStore;
use zkml::{
    graph::Node,
    layers::Layer,
    model::{Trace, llm::Driver},
    parser::{
        ModelNameProvider,
        llm::{
            HFTokenizer, Token,
            models::{
                gemma3::Gemma3,
                gpt2::{GPT2, is_gpt2_model},
            },
            tokenizer::{LLMTokenizer, TokenizerLoader},
        },
        safe::RawSafeTensors,
    },
};

/// Section separator for debug output
const SECTION_SEPARATOR: &str = "===";

#[derive(Parser, Debug)]
#[command(about = "Extract raw logits from ZKML inference engine")]
struct Args {
    /// Path to model directory containing model.safetensors, tokenizer.json, and config.json
    #[arg(short, long)]
    model: PathBuf,

    /// Input text for inference
    #[arg(short, long, default_value = "The quick brown fox")]
    text: String,

    /// Number of tokens to generate autoregressively (0 = no generation, just return input logits)
    #[arg(short, long, default_value = "0")]
    num_tokens: usize,
}

#[derive(Debug, Serialize)]
struct LogitsOutput {
    /// Logits from float mode inference [seq_len * vocab_size], row-major
    logits_float: Vec<f32>,
    /// Logits from integer/provable mode inference [seq_len_int * vocab_size], row-major
    logits_int: Vec<f32>,
    /// Sequence length (float mode)
    seq_len: usize,
    /// Sequence length (int mode, potentially padded)
    seq_len_int: usize,
    /// Vocabulary size (actual, not padded)
    vocab_size: usize,
}

#[derive(Debug)]
struct LogitsData {
    data: Vec<f32>,
    seq_len: usize,
}

fn write_logits_to_json(
    logits_float: &[f32],
    logits_int: &[f32],
    seq_len: usize,
    seq_len_int: usize,
    vocab_size: usize,
) -> Result<()> {
    let output = LogitsOutput {
        logits_float: logits_float.to_vec(),
        logits_int: logits_int.to_vec(),
        seq_len,
        seq_len_int,
        vocab_size,
    };

    let stdout = std::io::stdout();
    let handle = stdout.lock();
    serde_json::to_writer(handle, &output)?;

    Ok(())
}

fn extract_dimensions(shape: &zkml::Shape) -> (usize, usize) {
    if shape.rank() >= 2 {
        let seq_len = shape.dim(0);
        let vocab_size = shape.dim(shape.rank() - 1);
        (seq_len, vocab_size)
    } else {
        (1, shape.dim(0))
    }
}

/// Validate that required model files exist in the given directory
fn validate_model_files(model_dir: &Path) -> Result<(PathBuf, PathBuf, PathBuf)> {
    if !model_dir.is_dir() {
        bail!("Model path must be a directory: {}", model_dir.display());
    }

    let required_files = [
        ("model.safetensors", "Model file"),
        ("tokenizer.json", "Tokenizer file"),
        ("config.json", "Config file"),
    ];

    let mut paths = Vec::new();
    for (filename, description) in &required_files {
        let path = model_dir.join(filename);
        if !path.exists() {
            bail!("{} not found: {}", description, path.display());
        }
        paths.push(path);
    }

    Ok((paths[0].clone(), paths[1].clone(), paths[2].clone()))
}

/// Get the next token prediction from logits (argmax of last token's logits)
fn get_next_token(logits: &[f32], vocab_size: usize) -> usize {
    // Get the logits for the last token in the sequence
    let last_token_logits = &logits[logits.len() - vocab_size..];

    // Find the token with the highest logit value
    last_token_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn extract_logits_from_trace<E, N, F>(
    driver: &Driver<N>,
    trace: &Trace<E, N>,
    to_float: F,
) -> Result<LogitsData>
where
    E: ExtensionField,
    N: zkml::tensor::TensorTypeParam,
    F: Fn(&[N], zkml::graph::NodeId) -> Result<Vec<f32>>,
{
    for (node_id, node) in driver.model.graph().nodes() {
        let Node::Inner(layer) = node else {
            continue;
        };

        let Some(step) = trace.get_step(node_id) else {
            continue;
        };

        if !matches!(layer, Layer::Logits(_)) {
            continue;
        }

        let step_inputs = step.inputs();
        if step_inputs.is_empty() {
            continue;
        }

        let logit_tensor = step_inputs[0]
            .tensor()
            .ok()
            .context("Failed to get logit tensor")?;
        let shape = logit_tensor.shape();
        let data = to_float(logit_tensor.get_data(), *node_id)?;
        let (seq_len, _vocab_size) = extract_dimensions(shape);

        return Ok(LogitsData { data, seq_len });
    }

    bail!(
        "Could not extract logits from trace. \
        Make sure the model has a Logits layer and was executed correctly."
    )
}

/// Run autoregressive generation for the given driver
fn run_autoregressive_generation<E, N, RunFn, ExtractFn>(
    driver: &Driver<N>,
    tokenizer: &HFTokenizer,
    initial_tokens: Vec<Token>,
    num_tokens: usize,
    mode_name: &str,
    run_fn: RunFn,
    extract_fn: ExtractFn,
) -> Result<LogitsData>
where
    E: ExtensionField,
    N: zkml::tensor::TensorTypeParam,
    RunFn: Fn(&[Token], &mut GenStore) -> Result<Trace<E, N>>,
    ExtractFn: Fn(&[N], zkml::graph::NodeId) -> Result<Vec<f32>>,
{
    let mut generated_tokens = initial_tokens;
    let mut all_logits = Vec::new();
    let mut final_seq_len = 0;
    // Get actual vocab size from driver config (not padded)
    let actual_vocab_size = driver.vocab_size();

    // Run num_tokens + 1 iterations: initial + num_tokens generations
    // Only generate tokens in the first num_tokens iterations
    for i in 0..=num_tokens {
        driver.model.reset();
        let mut store = GenStore::default();

        eprintln!(
            "{} mode: Running inference on {} tokens",
            mode_name,
            generated_tokens.len()
        );

        let trace = run_fn(&generated_tokens, &mut store)?;
        let logits_step = extract_logits_from_trace(driver, &trace, &extract_fn)?;

        if i == 0 {
            // First iteration: store logits for the actual input tokens
            // Note: Integer mode may pad seq_len to power-of-2, but we only want the actual tokens
            let actual_seq_len = generated_tokens.len();
            let logits_to_store = actual_seq_len * actual_vocab_size;
            all_logits = logits_step.data[..logits_to_store].to_vec();
            final_seq_len = actual_seq_len;
        } else {
            // Subsequent iterations: only append the last token's logits
            let last_token_logits = &logits_step.data[logits_step.data.len() - actual_vocab_size..];
            all_logits.extend_from_slice(last_token_logits);
            // Update seq_len to account for the new token (using consistent vocab size)
            final_seq_len += 1;
        }

        if i < num_tokens {
            // Get next token and append to sequence
            let next_token_id = get_next_token(&logits_step.data, actual_vocab_size);
            let next_token: Token = next_token_id.into();
            generated_tokens.push(next_token);
            eprintln!(
                "Generated token {}: {} (id={})",
                i + 1,
                tokenizer.detokenize(&[next_token]),
                next_token_id
            );
        }
    }

    eprintln!("{} logits extracted: seq_len {}", mode_name, final_seq_len);

    let output = tokenizer.detokenize(&generated_tokens);
    eprintln!("{} output: {}", mode_name.to_lowercase(), output);

    Ok(LogitsData {
        data: all_logits,
        seq_len: final_seq_len,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Validate model files exist
    let (model_file, tokenizer_file, config_file) = validate_model_files(&args.model)?;

    let format = RawSafeTensors::new(model_file, tokenizer_file, config_file);

    // Detect model type from metadata
    let model_names = format.model_metadata()?;

    // Determine which model to use based on metadata
    let is_gpt2 = is_gpt2_model(&model_names);
    let is_gemma3 = model_names
        .iter()
        .any(|name| name.to_lowercase().contains("gemma"));

    eprintln!("is_gpt2 {is_gpt2}");

    // Load model, tokenizer, and tokenize input based on detected model type
    let (driver, tokenizer, user_tokens): (Driver<f32>, HFTokenizer, _) = if is_gpt2 {
        let gpt2 = GPT2::new();
        let tokenizer = gpt2.load_tokenizer(&format)?;
        let user_tokens = tokenizer.tokenize(&args.text);
        let driver =
            Driver::load_from_model(gpt2, &format, Some(args.num_tokens + user_tokens.len() + 1))?;
        (driver, tokenizer, user_tokens)
    } else if is_gemma3 {
        let gemma3 = Gemma3::new();
        let tokenizer = gemma3.load_tokenizer(&format)?;
        let user_tokens = tokenizer.tokenize(&args.text);
        let driver = Driver::load_from_model(
            gemma3,
            &format,
            Some(args.num_tokens + user_tokens.len() + 1),
        )?;
        (driver, tokenizer, user_tokens)
    } else {
        bail!(
            "Unsupported model type. Detected names: {:?}. \
            Supported models: GPT-2, Gemma3",
            model_names
        );
    };

    // Run float mode with autoregressive generation
    eprintln!(
        "\n{} Running Float Mode {}",
        SECTION_SEPARATOR, SECTION_SEPARATOR
    );
    let logits_float = run_autoregressive_generation(
        &driver,
        &tokenizer,
        user_tokens.clone(),
        args.num_tokens,
        "Float",
        |tokens, store| {
            let tensor_inputs = vec![driver.tokens_to_tensor(tokens)?];
            driver.run::<GoldilocksExt2>(tensor_inputs, store)
        },
        |data, _node_id| Ok(data.to_vec()),
    )?;
    // Reset cache and convert to provable mode
    eprintln!(
        "\n{} Running Integer/Provable Mode {}",
        SECTION_SEPARATOR, SECTION_SEPARATOR
    );
    driver.model.reset();
    let (driver_int, metadata) = driver.into_provable_llm(None)?;

    // Run provable mode with autoregressive generation
    // vocab_size is obtained from driver_int.vocab_size() which returns the actual (non-padded) size
    let logits_int = run_autoregressive_generation(
        &driver_int,
        &tokenizer,
        user_tokens.clone(),
        args.num_tokens,
        "Integer",
        |tokens, store| {
            let input_tensor = driver_int.tokens_to_tensor(tokens)?;
            driver_int.run_elements::<GoldilocksExt2>(input_tensor, store)
        },
        |data, node_id| {
            let scaling_factors = metadata.layer_input_scaling_factor(node_id);
            let scaling_factor = scaling_factors
                .first()
                .context("Failed to get scaling factor for logits layer")?;
            Ok(data
                .iter()
                .map(|&v| scaling_factor.dequantize(&v))
                .collect())
        },
    )?;

    write_logits_to_json(
        &logits_float.data,
        &logits_int.data,
        logits_float.seq_len,
        logits_int.seq_len,
        driver_int.vocab_size(),
    )?;

    Ok(())
}
