#![allow(clippy::print_stderr)]
#![allow(clippy::print_stdout)]

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Serialize;
use std::path::{Path, PathBuf};
use tenstore::GenStore;
use zkml::{
    graph::Node,
    layers::{Layer, provable::Evaluate},
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
    quantization::Dequantize,
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

fn extract_logits_from_trace<N, F>(
    driver: &Driver<N>,
    trace: &Trace<N>,
    to_float: F,
) -> Result<LogitsData>
where
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

/// Run inference using run_elements for the given driver
fn run_generation<N, ExtractFn>(
    driver: &Driver<N>,
    tokenizer: &HFTokenizer,
    tokens: &[Token],
    mode_name: &str,
    extract_fn: ExtractFn,
) -> Result<LogitsData>
where
    N: zkml::tensor::TensorTypeParam,
    ExtractFn: Fn(&[N], zkml::graph::NodeId) -> Result<Vec<f32>>,
    Layer<N>: Evaluate<N>,
{
    let mut store = GenStore::default();
    let tensor_inputs = driver.tokens_to_tensor(tokens)?;
    let trace: Trace<N> = driver.run_elements(tensor_inputs, &mut store)?;
    let logits_step = extract_logits_from_trace(driver, &trace, &extract_fn)?;

    let input_text = tokenizer.detokenize(tokens);
    eprintln!("{} input: {}", mode_name, input_text);

    let output_tokens = trace
        .outputs()
        .last()
        .unwrap()
        .tensor()
        .unwrap()
        .get_data()
        .iter()
        .skip(tokens.len())
        .map(|t| Token::from(zkml::Number::to_usize(t)))
        .collect::<Vec<_>>();
    let output = tokenizer.detokenize(&output_tokens);
    eprintln!("{} output: {}", mode_name, output);

    Ok(LogitsData {
        data: logits_step.data,
        seq_len: logits_step.seq_len,
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
            Driver::load_from_model(gpt2, &format, Some(args.num_tokens + user_tokens.len()))?;
        (driver, tokenizer, user_tokens)
    } else if is_gemma3 {
        let gemma3 = Gemma3::new();
        let tokenizer = gemma3.load_tokenizer(&format)?;
        let user_tokens = tokenizer.tokenize(&args.text);
        let driver =
            Driver::load_from_model(gemma3, &format, Some(args.num_tokens + user_tokens.len()))?;
        (driver, tokenizer, user_tokens)
    } else {
        bail!(
            "Unsupported model type. Detected names: {:?}. \
            Supported models: GPT-2, Gemma3",
            model_names
        );
    };

    // Run float mode
    eprintln!(
        "\n{} Running Float Mode {}",
        SECTION_SEPARATOR, SECTION_SEPARATOR
    );
    let logits_float = run_generation::<_, _>(
        &driver,
        &tokenizer,
        &user_tokens,
        "Float",
        |data, _node_id| Ok(data.to_vec()),
    )?;
    // Reset cache and convert to provable mode
    eprintln!(
        "\n{} Running Integer/Provable Mode {}",
        SECTION_SEPARATOR, SECTION_SEPARATOR
    );
    driver.model.reset();
    let (driver_int, metadata) = driver.into_provable_llm(None)?;

    // Run provable mode
    let logits_int = run_generation::<_, _>(
        &driver_int,
        &tokenizer,
        &user_tokens,
        "Integer",
        |data, node_id| {
            let scaling_factors = metadata.layer_input_scaling_factor(node_id);
            let scaling_factor = scaling_factors
                .first()
                .context("Failed to get scaling factor for logits layer")?;
            Ok(data.dequantize(scaling_factor))
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
