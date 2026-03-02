#!/usr/bin/env python3
"""
Model Evaluation Script for Quantization Validation

This script compares a Rust ZKML model against a baseline PyTorch
HuggingFace model using three key metrics:
1. Logit Cosine Similarity
2. Perplexity on WikiText-103 (or custom text)
3. Next-Token Agreement

Currently supports: google/gemma-3-270m-it (safetensors format)

Usage:
    python evaluate_models.py --model google/gemma-3-270m-it --text "Hello world"
"""

import argparse
import json
import subprocess
import sys
import os
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Generator, List, NamedTuple, Tuple


# Disable tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from torchmetrics.text import Perplexity, ROUGEScore
import csv
import datetime

# Path configuration - script is in zkml/assets/scripts/accuracy_evaluation/
SCRIPT_DIR = Path(__file__).parent
ZKML_ROOT = SCRIPT_DIR.parent.parent.parent
MODEL_CACHE_DIR = ZKML_ROOT / "model_cache"

# Constants
MIN_SAMPLE_LENGTH = 100  # Minimum text length for WikiText samples
TYPICAL_LOGIT_RANGE = (-500, 100)  # Typical range for logits
SMOKE_TEST_THRESHOLD_PCT = 10  # Max acceptable percentage difference for smoke tests
SECTION_SEPARATOR = "=" * 60  # Standard section separator


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{SECTION_SEPARATOR}")
    print(title)
    print(SECTION_SEPARATOR)



class ModelOutput(NamedTuple):
    """Output from model inference."""

    logits: np.ndarray
    input_ids: torch.Tensor
    generated_text: str = ""
    generated_only_text: str = ""  # Only generated tokens, no input





def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate ZKML model correctness using safetensors format"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., 'google/gemma-2-2b-it')",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Input text for evaluation. If not provided, uses WikiText-103 sample",
    )
    parser.add_argument(
        "--dataset-sample",
        type=str,
        default="wikitext",
        choices=["wikitext", "custom"],
        help="Dataset to use: 'wikitext' for WikiText-103, 'custom' for --text",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to process for single sample (default: 512)",
    )
    parser.add_argument(
        "--full-test-set",
        action="store_true",
        help="Evaluate on full WikiText-103 test set with sliding window (slower but proper benchmark)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Sliding window stride for full test set evaluation (default: 512)",
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        default=100,
        help="Number of windows to evaluate in full test set mode (default: 100, use 0 for all windows)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output results in markdown format suitable for GitHub comments",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="File to write markdown output to (only used with --markdown)",
    )
    parser.add_argument(
        "--generate-tokens",
        type=int,
        default=0,
        help="Number of tokens to generate autoregressively (0 = no generation, just compare logits)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of samples to evaluate (only for single sample mode)",
    )
    return parser.parse_args()


def get_wikitext_sample(
    tokenizer: AutoTokenizer, max_tokens: int = 512
) -> Tuple[str, List[int], int]:
    """Load a sample from WikiText-103 test set."""
    print("Loading WikiText-103 dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    # Find a substantial text sample (skip empty lines)
    for sample in dataset:
        text = sample["text"].strip()
        if len(text) > MIN_SAMPLE_LENGTH:
            # Tokenize and truncate
            tokens = tokenizer.encode(text)
            final_token = tokens[-1]
            if len(tokens) > max_tokens:
                final_token = tokens[max_tokens]
                tokens = tokens[:max_tokens]
            text = tokenizer.decode(tokens)
            print(f"✓ Loaded WikiText-103 sample ({len(tokens)} tokens)")
            return text, tokens, final_token

    raise ValueError("No suitable WikiText-103 sample found")

def get_wikitext2_samples(tokenizer: AutoTokenizer, max_tokens: int = 512, max_docs=100) -> List[Tuple[str, List[int], int]]:
    """Load samples from WikiText-2 test set."""
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    samples = []
    for doc in dataset["text"]:
        full_text = tokenizer.eos_token + doc
        tokens = tokenizer(full_text, return_tensors="pt").input_ids
       
        tokens = tokens[0].numpy()
        
        if len(tokens) < max_tokens:
            continue
        total_sequences = (len(tokens) - 1) // max_tokens
        for start_idx in range(total_sequences):
            input_ids = tokens[start_idx * max_tokens : (start_idx + 1) * max_tokens]
            final_token = tokens[(start_idx + 1) * max_tokens]
            text = tokenizer.decode(input_ids)
            samples.append((text, input_ids.tolist(), final_token))
            if len(samples) >= max_docs:
               break
        
        if len(samples) >= max_docs:
               break
    
    print(f"✓ Loaded {len(samples)} WikiText-2 samples")
    return samples


def get_wikitext_full_test(
    tokenizer: AutoTokenizer,
    stride: int = 512,
    max_length: int = 1024,
    num_windows: int = 100,
) -> Generator[Tuple[str, List[int]], None, None]:
    """
    Load full WikiText-103 test set for sliding window evaluation.

    Uses sliding window approach as described in HuggingFace documentation:
    https://huggingface.co/transformers/v4.2.2/perplexity.html

    Smaller stride = more context per prediction = better (but slower) perplexity.
    Standard practice: stride=512 for balance of accuracy and speed.

    Args:
        tokenizer: HuggingFace tokenizer
        stride: Stride for sliding window (default: 512)
        max_length: Maximum context length (default: 1024)
        num_windows: Number of windows to yield (default: 100, use 0 for all windows)

    Yields:
        (text, tokens) tuples for each window
    """
    print("Loading full WikiText-103 test set...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    # Concatenate all test samples (generator expression for memory efficiency)
    full_text = " ".join(sample["text"] for sample in dataset if sample["text"].strip())
    print(f"✓ Loaded full test set")

    # Tokenize entire test set
    all_tokens = tokenizer.encode(full_text)
    total_possible_windows = (len(all_tokens) - max_length) // stride + 1
    print(f"  Total tokens: {len(all_tokens)}")
    print(f"  Total possible windows: {total_possible_windows}")

    windows_to_generate = (
        total_possible_windows
        if num_windows == 0
        else min(num_windows, total_possible_windows)
    )
    print(
        f"  Evaluating {windows_to_generate} windows"
        + ("" if num_windows == 0 else f" (of {total_possible_windows} total)")
    )

    # Generate sliding windows using itertools.islice for cleaner iteration
    def window_generator():
        for i in range(0, len(all_tokens) - max_length, stride):
            window_tokens = all_tokens[i : i + max_length]
            window_text = tokenizer.decode(window_tokens)
            yield window_text, window_tokens

    yield from islice(window_generator(), windows_to_generate)

# https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb
def group_texts(examples,block_size=128):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result



def load_wikitext(tokenizer, block_size=128):
    print("Loading WikiText-2...")
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    max_tokens = sum([len(t) for t in data["text"]])
    len_tokens = { }
    for doc in data["text"]:
        if len(doc) not in len_tokens:
            len_tokens[len(doc)] = 0
        len_tokens[len(doc)] += 1
    
    ordered_len_tokens = sorted(len_tokens.items(), key=lambda x: x[0],reverse=True)
    print(f"Sizes: {ordered_len_tokens[:10]}")
    data_len = len(data["text"])
    print(f"Evaluation on {data_len} docs (max {max_tokens} tokens)")
     # => sizes {ordered_len_tokens}")
    
    # tokenize everything 
    tokenized_text = data.map(lambda x: tokenizer(x["text"]),batched=True,num_proc=4,remove_columns=["text"])
    tokenized_text = tokenized_text.map(lambda x: group_texts(x,block_size), batched=True, batch_size=1000,num_proc=4)

    tokenized_len = len(tokenized_text["input_ids"])
    print(f"TOKENIZED TEXT -> number of chunks: {tokenized_len}")
    return tokenized_text["input_ids"]


def run_baseline_model(
    model_name: str, text: str, tokenizer: AutoTokenizer, num_generate: int = 0, tokens: torch.Tensor = None
) -> ModelOutput:
    """
    Run HuggingFace baseline model and return all logits.

    Args:
        model_name: HuggingFace model identifier
        text: Input text
        tokenizer: HuggingFace tokenizer
        num_generate: Number of tokens to generate autoregressively (0 = no generation)

    Returns:
        ModelOutput with logits and input_ids
    """
    print_section_header("Running Baseline HuggingFace Model")

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    input_ids = tokens if tokens is not None else tokenizer.encode(text, return_tensors="pt")
    print(f"Input tokens: {input_ids.shape}")
    print(f"First 10 tokens: {input_ids[0, :10].tolist()}")

    if num_generate > 0:
        print(f"Generating {num_generate} tokens autoregressively...")
        all_logits = []
        current_ids = input_ids.clone()
        all_logits = None
        with torch.no_grad():
            for i in range(num_generate + 1):
                outputs = model(current_ids)
                logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]
                all_logits = np.array(outputs.logits.cpu())

                if i < num_generate:
                    # Get next token (argmax of last position)
                    next_token_ids = torch.argmax(logits[:, -1:, :], dim=-1)
                    print(f"type of next_token_ids: {next_token_ids.shape}")
                    next_tokens = tokenizer.batch_decode(next_token_ids, skip_special_tokens=False)
                    print(f"Generated token {i+1}: {next_tokens!r} (id={next_token_ids})")

                    # Append to sequences
                    print(f"current_ids shape before: {current_ids.shape}")
                    current_ids = torch.cat(
                        [current_ids, torch.tensor(next_token_ids)], dim=1
                    )

        # Concatenate all logits
        logits_np = all_logits # Shape: [seq_len + num_generate, vocab_size]
        final_ids = current_ids

        generated_text = tokenizer.batch_decode(final_ids, skip_special_tokens=False)
        # Decode only the generated tokens (after input)
        input_token_count = input_ids.shape[1]
        generated_only_ids = final_ids[:, input_token_count:]
        generated_only_text = tokenizer.batch_decode(generated_only_ids, skip_special_tokens=False)
        print(f"Final output: {generated_text}")
        print(f"Generated only: {generated_only_text}")
    else:
        # Original behavior: single forward pass
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]
        logits_np = logits.cpu().numpy()
        final_ids = input_ids
        generated_text = tokenizer.decode(final_ids.flatten())
        # Remove batch dimension and convert to numpy
        if input_ids.shape[0] == 1:
            logits_np = logits.squeeze(0).cpu().numpy()  # Shape: [seq_len, vocab_size]
            final_ids = input_ids.squeeze(0)
            generated_text = tokenizer.decode(final_ids)

    print(f"Logits shape: {logits_np.shape}")
    print(
        f"Logits stats: mean={logits_np.mean():.4f}, std={logits_np.std():.4f}, "
        f"min={logits_np.min():.4f}, max={logits_np.max():.4f}"
    )

    print(f"✓ Baseline inference complete")
    # For generation mode, we have generated_only_text; for non-generation, it's empty
    gen_only = generated_only_text if num_generate > 0 else ""
    return ModelOutput(
        logits=logits_np,
        input_ids=final_ids,
        generated_text=generated_text,
        generated_only_text=gen_only,
    )

        


def run_rust_model(
    model_path: Path, text: str, num_generate: int = 0, sample_size: int = 0, bit_len: int = 8, tokens: List[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Rust ZKML model and load both float and int logits from stdout.

    Args:
        model_path: Path to model directory
        text: Input text
        num_generate: Number of tokens to generate autoregressively (0 = no generation)

    Returns:
        Tuple of (logits_float, logits_int)
    """
    print_section_header("Running Rust ZKML Model")

    cmd = [
        "cargo",
        "run",
        "--release",
        "--bin",
        "extract-logits",
        "--",
        "--model",
        str(model_path),
        f"--text={text}",
    ]

    if num_generate > 0:
        cmd.append(f"--num-tokens={num_generate}")
    
    if sample_size > 0:
        cmd.append(f"--sample-size={sample_size}")
    
    if tokens is not None:
        tokens_str = ",".join(str(t) for t in tokens)
        cmd.append(f"--tokens={tokens_str}")

    current_bl = os.environ.get("ZKML_BIT_LEN", None)
    print(f"Current ZKML_BIT_LEN: {current_bl}")
    print(f"Setting ZKML_BIT_LEN to {bit_len} for this run")
    os.environ["ZKML_BIT_LEN"] = str(bit_len)
    print(f"ZKML_BIT_LEN is now: {os.environ.get('ZKML_BIT_LEN', None)}")
    print(f"Executing: {' '.join(cmd)[:100]}")
    print(f"Working directory: {ZKML_ROOT}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=ZKML_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        # Print stderr (contains tracing logs)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"✗ Cargo run failed with exit code {e.returncode}", file=sys.stderr)
        print(f"stdout: {e.stdout}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        raise

    # Parse JSON from stdout
    print("Parsing logits from JSON output...")
    try:
        output_data = json.loads(result.stdout)
        logits_float_flat = np.array(output_data["logits_float"], dtype=np.float32)
        logits_int_flat = np.array(output_data["logits_int"], dtype=np.float32)
        seq_len = output_data["seq_len"]
        vocab_size = output_data["vocab_size"]
        seq_len_int = output_data["seq_len_int"]

        # Reshape to their respective dimensions
        logits_float = logits_float_flat.reshape(logits_float_flat.shape[0], seq_len, vocab_size)

        # Integer mode may have padded BOTH seq_len and vocab_size
        # Calculate the actual padded vocab size from the array
        padded_vocab_size = logits_int_flat.shape[1] // seq_len_int
        logits_int_padded = logits_int_flat.reshape(logits_int_flat.shape[0], seq_len_int, padded_vocab_size)

        # Unpad int logits to match float/baseline shape
        # Int mode may have padded seq_len (e.g., 7 -> 8) and vocab_size (e.g., 50257 -> 65536)
        if seq_len_int != seq_len or padded_vocab_size != vocab_size:
            print(
                f"  Unpadding int mode from ({seq_len_int}, {padded_vocab_size}) to ({seq_len}, {vocab_size})"
            )
            logits_int = logits_int_padded[:, :seq_len, :vocab_size]
        else:
            logits_int = logits_int_padded
    except (json.JSONDecodeError, KeyError) as e:
        print(f"✗ Failed to parse JSON output: {e}", file=sys.stderr)
        print(f"stdout: {result.stdout[:500]}", file=sys.stderr)
        raise

    print(f"\nFloat mode (true float32 inference):")
    print(f"  Logits shape: {logits_float.shape}")
    print(
        f"  Logits stats: mean={logits_float.mean():.4f}, std={logits_float.std():.4f}, "
        f"min={logits_float.min():.4f}, max={logits_float.max():.4f}"
    )

    print(f"\nInt mode (dequantized from quantized inference):")
    print(f"  Logits shape: {logits_int.shape}")
    print(
        f"  Logits stats: mean={logits_int.mean():.4f}, std={logits_int.std():.4f}, "
        f"min={logits_int.min():.4f}, max={logits_int.max():.4f}"
    )

    print(f"✓ Rust inference complete")
    # Restore previous ZKML_BIT_LEN
    if current_bl is not None:
        os.environ["ZKML_BIT_LEN"] = current_bl
    else:
        del os.environ["ZKML_BIT_LEN"]
    return logits_float, logits_int


def compute_generated_text_from_logits(
    logits: np.ndarray,
    tokenizer: AutoTokenizer,
    num_input_tokens: int,
    num_generated_tokens: int,
) -> str:
    """
    Compute generated text from logits using argmax.

    Args:
        logits: Logits array [seq_len, vocab_size] where logits[i] predicts token at position i+1
        tokenizer: HuggingFace tokenizer
        num_input_tokens: Number of input tokens (including BOS if present)
        num_generated_tokens: Number of tokens that were generated

    Returns:
        Generated text (decoded from argmax of logits)
    """
    if logits.shape[0] < num_input_tokens:
        return ""

    # Logits[i] predicts the token at position i+1
    # For num_input_tokens input tokens [0...num_input_tokens-1], the generated tokens are at positions:
    # [num_input_tokens, num_input_tokens+1, ..., num_input_tokens+num_generated_tokens-1]
    # These are predicted by logits at positions:
    # [num_input_tokens-1, num_input_tokens, ..., num_input_tokens+num_generated_tokens-2]
    start_idx = num_input_tokens - 1
    end_idx = start_idx + num_generated_tokens

    generated_logits = logits[start_idx:end_idx]

    # Get token IDs using argmax
    generated_token_ids = np.argmax(generated_logits, axis=1).tolist()

    # Decode to text
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)

    return generated_text

def compute_full_text_from_logits(
    logits: np.ndarray,
    tokenizer: AutoTokenizer,
) -> str:
    """
    Compute generated text from logits using argmax.

    Args:
        logits: Logits array [seq_len, vocab_size] where logits[i] predicts token at position i+1
        tokenizer: HuggingFace tokenizer
        num_input_tokens: Number of input tokens (including BOS if present)
        num_generated_tokens: Number of tokens that were generated

    Returns:
        Generated text (decoded from argmax of logits)
    """
    

    # Get token IDs using argmax
    generated_token_ids = np.argmax(logits, axis=1).tolist()

    # Decode to text
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)

    return generated_text

def compute_full_text_from_batched_logits(
    logits: np.ndarray,
    tokenizer: AutoTokenizer,
) -> List[str]:
    """
    Compute generated texts from logits using argmax.

    Args:
        logits: Logits array [batch_size, seq_len, vocab_size] where logits[i] predicts token at position i+1
        tokenizer: HuggingFace tokenizer

    Returns:
        Generated text (decoded from argmax of logits)
    """
    

    # Get token IDs using argmax
    generated_token_ids = np.argmax(logits, axis=2)
    

    # Decode to text
    all_texts = []
    for i in range(generated_token_ids.shape[0]):
            text_i = tokenizer.decode(generated_token_ids[i], skip_special_tokens=False)
            all_texts.append(text_i)

    return all_texts

def compute_generated_text_from_batched_logits(
    logits: np.ndarray,
    tokenizer: AutoTokenizer,
    num_input_tokens: int,
) -> List[str]:
    """
    Compute generated texts from logits using argmax.

    Args:
        logits: Logits array [batch_size, seq_len, vocab_size] where logits[i] predicts token at position i+1
        tokenizer: HuggingFace tokenizer
        num_input_tokens: Number of input tokens (including BOS if present)
    Returns:
        Generated text (decoded from argmax of logits)
    """
    

    # Get token IDs using argmax
    generated_token_ids = np.argmax(logits, axis=2)
    

    # Decode to text
    return tokenizer.batch_decode(generated_token_ids[..., num_input_tokens:], skip_special_tokens=False)
    


def has_invalid_values(arr: np.ndarray) -> bool:
    """Check if array contains NaN or Inf values."""
    return bool(np.any(np.isnan(arr)) or np.any(np.isinf(arr)))





def ensure_model_downloaded(model_name: str, model_path: Path) -> None:
    """
    Download exact model files from HuggingFace Hub using hf_hub_download.
    Downloads only the three required files: config.json, tokenizer.json, model.safetensors
    Skips downloading files that already exist.

    Args:
        model_name: HuggingFace model identifier
        model_path: Local path where model should be stored

    Raises:
        SystemExit: If download fails
    """
    # Files we need for ZKML
    required_files = ["config.json", "tokenizer.json", "model.safetensors"]

    # Check which files already exist
    existing_files = []
    missing_files = []
    for filename in required_files:
        file_path = model_path / filename
        if file_path.exists():
            existing_files.append(filename)
        else:
            missing_files.append(filename)

    # Skip download if all files exist
    if not missing_files:
        print(f"✓ Model files already exist at: {model_path}")
        return

    print(f"Downloading missing model files from HuggingFace: '{model_name}'...")
    if existing_files:
        print(f"  Already present: {', '.join(existing_files)}")

    try:
        model_path.mkdir(parents=True, exist_ok=True)

        # Download only missing files
        for filename in missing_files:
            print(f"  Downloading {filename}...")
            downloaded_file = hf_hub_download(
                repo_id=model_name,
                filename=filename,
                local_dir=model_path,
                local_dir_use_symlinks=False,  # Copy actual files, not symlinks
            )
            print(f"    ✓ {filename} saved to: {downloaded_file}")

        print(f"  ✓ All files ready at: {model_path}")

    except (OSError, ValueError, RuntimeError) as e:
        print(f"✗ Failed to download model: {e}", file=sys.stderr)
        print(
            f"\nYou may need to authenticate with HuggingFace for gated models:",
            file=sys.stderr,
        )
        print(f"  huggingface-cli login", file=sys.stderr)
        sys.exit(1)


def run_text_generation(
    model_path: Path, tokenizer: AutoTokenizer, args: argparse.Namespace
    ):
    """
    Run evaluation on multiple text samples of a fixed length.

    Returns:
        Tuple of (metrics_float, max_metrics_float, metrics_int, max_metrics_int,
                  smoke_tests_float, smoke_tests_int,
                  generated_text_baseline, generated_text_float, generated_text_int, input_text)
    """
    # Get evaluation text
    sample = load_wikitext(tokenizer, block_size=args.max_tokens)
    array_sample = np.array(sample)[:2]
    full_text = tokenizer.decode(array_sample.flatten().tolist(), skip_special_tokens=True)
    
    
    # Run baseline model
    model_output = run_baseline_model(
            str(model_path), full_text, tokenizer, args.generate_tokens, tokens=torch.tensor(array_sample[:, :-1])
        )

    # Run Rust model - get both float and int logits
    all_logits_int = []
    logits_float, logits_int = run_rust_model(model_path, full_text, num_generate=args.generate_tokens, sample_size=args.max_tokens - 1, bit_len=8, tokens=array_sample.flatten().tolist())

    all_logits_int.append(logits_int)
    for bit_len in [10, 12]:
        _, logits_int = run_rust_model(model_path, full_text, num_generate=args.generate_tokens, sample_size=args.max_tokens - 1, bit_len=bit_len, tokens=array_sample.flatten().tolist())
        all_logits_int.append(logits_int)
    
    
    
    
    # Always compute generated text from logits using argmax for all modes
    # This ensures ROUGE metrics compare argmax predictions across all models
    # Note: model_output.input_ids contains the FULL sequence (input + generated), not just input
    # Calculate the original input token count from the total sequence length
    num_input_tokens = model_output.input_ids.shape[1] - args.generate_tokens
    
    prompts = tokenizer.batch_decode(array_sample[:, :-1], skip_special_tokens=False)
    float_outputs = compute_generated_text_from_batched_logits(logits_float, tokenizer, num_input_tokens=num_input_tokens)
    all_int_outputs = []
    for logits_int in all_logits_int:
        int_outputs = compute_generated_text_from_batched_logits(logits_int, tokenizer, num_input_tokens=num_input_tokens)
        all_int_outputs.append(int_outputs)
    
    for i in range(len(prompts)):
        prompt = prompts[i]
        float_out = float_outputs[i]
        int_outs = [all_int_outputs[j][i] for j in range(len(all_int_outputs))]
        print(f"=== Sample {i+1} ===")
        print(f"Prompt: {prompt}")
        print(f"Baseline Generated: {model_output.generated_only_text}")
        print(f"Float Generated:   {float_out}")
        for bit_len, int_out in zip([8, 10, 12], int_outs):
            print(f"Int_{bit_len} Generated:    {int_out}")

    
    
     
   




def main():
    args = parse_args()

    print_section_header("Model Evaluation - Quantization Validation")
    print(f"Evaluating model: '{args.model}'")

    # Determine model path in cache
    model_path = MODEL_CACHE_DIR / args.model

    # Download model if not present
    ensure_model_downloaded(args.model, model_path)

    print(f"✓ Found model directory: {model_path}")
    print(f"✓ Found safetensors file: {model_path / 'model.safetensors'}")

    # Convert to absolute path
    model_path = model_path.absolute()

    # Load tokenizer from local model path (already downloaded with exact files)
    print(f"\nLoading tokenizer from local cache: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Run evaluation
    run_text_generation(model_path, tokenizer, args)
    


if __name__ == "__main__":
    main()
