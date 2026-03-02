#![allow(clippy::print_stderr)]
#![allow(clippy::print_stdout)]

use anyhow::{Context, Result, bail};
use clap::Parser;
use itertools::Itertools;
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
                LLMModelLoader,
                gemma3::Gemma3,
                gpt2::{GPT2, is_gpt2_model},
            },
            tokenizer::{LLMTokenizer, TokenizerLoader},
        },
        safe::RawSafeTensors,
    },
    quantization::{Dequantize, llm_quant::FPTransformModel},
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

    /// Size of each sample to run, if provided with 0 will run a single sample over the whole input
    #[arg(short, long, default_value = "0")]
    sample_size: usize,

    /// The (flat) tokens to use as input instead of text (overrides --text)
    #[arg(long, value_delimiter = ',')]
    tokens: Option<Vec<usize>>,
}

#[derive(Debug, Serialize)]
struct LogitsOutput {
    /// Logits from float mode inference [seq_len * vocab_size], row-major
    logits_float: Vec<Vec<f32>>,
    /// Logits from integer/provable mode inference [seq_len_int * vocab_size], row-major
    logits_int: Vec<Vec<f32>>,
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
    logits_float: &[LogitsData],
    logits_int: &[LogitsData],
    vocab_size: usize,
) -> Result<()> {
    let seq_len = logits_float
        .first()
        .context("No float logits data found")?
        .seq_len;
    let seq_len_int = logits_int
        .first()
        .context("No int logits data found")?
        .seq_len;

    let logits_float = logits_float
        .iter()
        .map(|ld| ld.data.clone())
        .collect::<Vec<Vec<f32>>>();
    let logits_int = logits_int
        .iter()
        .map(|ld| ld.data.clone())
        .collect::<Vec<Vec<f32>>>();

    let output = LogitsOutput {
        logits_float,
        logits_int,
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
        let data = to_float(logit_tensor.data(), *node_id)?;
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
        .data()
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

fn run_multi_sample_generation<N, ExtractFn>(
    driver: &Driver<N>,
    tokenizer: &HFTokenizer,
    token_samples: &[&[Token]],
    mode_name: &str,
    extract_fn: ExtractFn,
) -> Result<Vec<LogitsData>>
where
    N: zkml::tensor::TensorTypeParam,
    ExtractFn: Fn(&[N], zkml::graph::NodeId) -> Result<Vec<f32>>,
    Layer<N>: Evaluate<N>,
{
    // First check all the tokens have the same length
    let all_equal_len = token_samples.iter().map(|t| t.len()).all_equal();
    anyhow::ensure!(
        all_equal_len,
        "All token sequences must have the same length to run multiple samples in mode {mode_name}"
    );
    let mut store = GenStore::default();

    let mut logits_data = Vec::<LogitsData>::with_capacity(token_samples.len());
    for tokens in token_samples {
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
            .data()
            .iter()
            .skip(tokens.len())
            .map(|t| Token::from(zkml::Number::to_usize(t)))
            .collect::<Vec<_>>();
        let output = tokenizer.detokenize(&output_tokens);
        eprintln!("{} output: {}", mode_name, output);
        logits_data.push(logits_step);
    }

    Ok(logits_data)
}

fn internal_generate_logits<M>(format: RawSafeTensors, args: &Args) -> Result<()>
where
    M: LLMModelLoader<RawSafeTensors>
        + FPTransformModel
        + TokenizerLoader<RawSafeTensors>
        + Clone
        + Default,
{
    let model_type = M::default();
    let tokenizer = model_type.load_tokenizer(&format)?;
    let user_tokens = if let Some(tokens) = &args.tokens {
        eprintln!("Using provided tokens!");
        tokens
            .iter()
            .map(|&t| Token::from(t))
            .collect::<Vec<Token>>()
    } else {
        tokenizer.tokenize(&args.text)
    };
    eprintln!("First 10 tokens: {:?}", &user_tokens[..10]);
    let max_context = if args.sample_size != 0 {
        args.num_tokens + args.sample_size
    } else {
        args.num_tokens + user_tokens.len()
    };
    let driver =
        Driver::load_from_model(model_type.clone(), &format, None)?.with_max_context(max_context);
    if args.sample_size != 0 {
        let chunked_tokens = user_tokens
            .chunks_exact(args.sample_size + 1)
            .map(|chunk| &chunk[..args.sample_size])
            .collect::<Vec<&[Token]>>();
        // Run float mode
        eprintln!(
            "\n{} Running Float Mode {}",
            SECTION_SEPARATOR, SECTION_SEPARATOR
        );
        let logits_float = run_multi_sample_generation::<_, _>(
            &driver,
            &tokenizer,
            &chunked_tokens,
            "Float",
            |data, _node_id| Ok(data.to_vec()),
        )?;
        // Reset cache and convert to provable mode
        eprintln!(
            "\n{} Running Integer/Provable Mode {}",
            SECTION_SEPARATOR, SECTION_SEPARATOR
        );
        driver.model.reset();

        let (driver_int, metadata) = driver.into_provable_llm_with_transform::<M>(&tokenizer)?;

        // Run provable mode
        let logits_int = run_multi_sample_generation::<_, _>(
            &driver_int,
            &tokenizer,
            &chunked_tokens,
            "Integer",
            |data, node_id| {
                let scaling_factors = metadata.layer_input_scaling_factor(node_id);
                let scaling_factor = scaling_factors
                    .first()
                    .context("Failed to get scaling factor for logits layer")?;
                Ok(data.dequantize(scaling_factor))
            },
        )?;

        write_logits_to_json(&logits_float, &logits_int, driver_int.vocab_size())?;
    } else {
        let max_context = args.num_tokens + user_tokens.len();

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

        let (driver_int, metadata) = driver.into_provable_llm_with_transform::<M>(&tokenizer)?;

        let driver_int = driver_int.with_max_context(max_context);

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

        write_logits_to_json(&[logits_float], &[logits_int], driver_int.vocab_size())?;
    };
    Ok(())
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

    if is_gpt2 {
        internal_generate_logits::<GPT2>(format, &args)?;
    } else if is_gemma3 {
        internal_generate_logits::<Gemma3>(format, &args)?;
    } else {
        bail!("Unsupported model type. Only GPT-2 and Gemma3 models are supported.");
    }

    Ok(())
}
