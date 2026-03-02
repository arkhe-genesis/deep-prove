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


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    cosine_similarity: float
    perplexity_baseline: float
    perplexity_test: float
    perplexity_last_token_base_line: float
    perplexity_last_token: float
    agreement: float
    rouge1_f1: float = 0.0  # ROUGE-1 F1 score (unigram overlap)
    rouge2_f1: float = 0.0  # ROUGE-2 F1 score (bigram overlap)
    rougeL_f1: float = 0.0  # ROUGE-L F1 score (longest common subsequence)

    @property
    def perplexity_delta_pct(self) -> float:
        """Compute percentage change in perplexity."""
        return (
            (self.perplexity_test - self.perplexity_baseline) / self.perplexity_baseline
        ) * 100


class ModelOutput(NamedTuple):
    """Output from model inference."""

    logits: np.ndarray
    input_ids: torch.Tensor
    generated_text: str = ""
    generated_only_text: str = ""  # Only generated tokens, no input


def compute_average_metrics(
    metrics_list: List[EvaluationMetrics],
) -> Tuple[EvaluationMetrics, EvaluationMetrics]:
    """
    Compute average and max metrics from a list of EvaluationMetrics.

    Args:
        metrics_list: List of EvaluationMetrics objects

    Returns:
        Tuple of (mean_metrics, max_metrics)
    """
    mean_metrics = EvaluationMetrics(
        cosine_similarity=np.mean([m.cosine_similarity for m in metrics_list]),
        perplexity_baseline=np.mean([m.perplexity_baseline for m in metrics_list]),
        perplexity_test=np.mean([m.perplexity_test for m in metrics_list]),
        perplexity_last_token=np.mean([m.perplexity_last_token for m in metrics_list]),
        perplexity_last_token_base_line=np.mean([m.perplexity_last_token_base_line for m in metrics_list]),
        agreement=np.mean([m.agreement for m in metrics_list]),
        rouge1_f1=np.mean([m.rouge1_f1 for m in metrics_list]),
        rouge2_f1=np.mean([m.rouge2_f1 for m in metrics_list]),
        rougeL_f1=np.mean([m.rougeL_f1 for m in metrics_list]),
    )

    max_metrics = EvaluationMetrics(
        cosine_similarity=np.max([m.cosine_similarity for m in metrics_list]),
        perplexity_baseline=np.max([m.perplexity_baseline for m in metrics_list]),
        perplexity_test=np.max([m.perplexity_test for m in metrics_list]),
        perplexity_last_token=np.max([m.perplexity_last_token for m in metrics_list]),
        perplexity_last_token_base_line=np.max([m.perplexity_last_token_base_line for m in metrics_list]),
        agreement=np.max([m.agreement for m in metrics_list]),
        rouge1_f1=np.max([m.rouge1_f1 for m in metrics_list]),
        rouge2_f1=np.max([m.rouge2_f1 for m in metrics_list]),
        rougeL_f1=np.max([m.rougeL_f1 for m in metrics_list]),
    )

    return mean_metrics, max_metrics


def run_smoke_tests_with_warning(
    baseline_logits: np.ndarray, test_logits: np.ndarray, mode_name: str
) -> bool:
    """
    Run smoke tests and print warning if they fail.

    Args:
        baseline_logits: Baseline model logits
        test_logits: Test model logits
        mode_name: Name of the mode being tested (e.g., "Float", "Integer")

    Returns:
        True if tests pass, False otherwise
    """
    print(f"\n--- {mode_name} Mode Smoke Tests ---")
    smoke_tests_passed = run_smoke_tests(baseline_logits, test_logits)
    if not smoke_tests_passed:
        print(
            f"\n⚠ Warning: {mode_name} mode smoke tests failed, but continuing with metric calculations...\n"
        )
    return smoke_tests_passed


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
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")

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
        print(f"tokens shape, no indexing: {tokens.shape}")
        tokens = tokens[0].numpy()
        print(f"tokens shape, after indexing: {tokens.shape}")
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
    data = data.shuffle()
    # data = test_data.shuffle()
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

    if num_generate > 0:
        print(f"Generating {num_generate} tokens autoregressively...")
        all_logits = []
        current_ids = input_ids.clone()

        with torch.no_grad():
            for i in range(num_generate + 1):
                outputs = model(current_ids)
                logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]

                if i == 0:
                    # First iteration: keep all logits
                    all_logits.append(logits.squeeze(0).cpu().numpy())
                else:
                    # Subsequent iterations: only append last token's logits
                    last_logits = logits[0, -1:, :].cpu().numpy()
                    all_logits.append(last_logits)

                if i < num_generate:
                    # Get next token (argmax of last position)
                    next_token_id = logits[0, -1, :].argmax().item()
                    next_token = tokenizer.decode([next_token_id])
                    print(f"Generated token {i+1}: {next_token!r} (id={next_token_id})")

                    # Append to sequence
                    current_ids = torch.cat(
                        [current_ids, torch.tensor([[next_token_id]])], dim=1
                    )

        # Concatenate all logits
        logits_np = np.vstack(all_logits)  # Shape: [seq_len + num_generate, vocab_size]
        final_ids = current_ids.squeeze(0)

        generated_text = tokenizer.decode(final_ids)
        # Decode only the generated tokens (after input)
        input_token_count = input_ids.shape[1]
        generated_only_ids = final_ids[input_token_count:]
        generated_only_text = tokenizer.decode(generated_only_ids)
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
    model_path: Path, text: str, num_generate: int = 0, sample_size: int = 0
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

    print(f"Executing: {' '.join(cmd)}")
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


def has_invalid_values(arr: np.ndarray) -> bool:
    """Check if array contains NaN or Inf values."""
    return bool(np.any(np.isnan(arr)) or np.any(np.isinf(arr)))


def compute_cosine_similarity(
    logits_base: np.ndarray, logits_test: np.ndarray, skip_first: bool = True
) -> float:
    """
    Compute average cosine similarity between logit vectors using PyTorch.

    Args:
        logits_base: Baseline logits [seq_len, vocab_size]
        logits_test: Test logits [seq_len, vocab_size]
        skip_first: Skip first token (no context)

    Returns:
        Average cosine similarity
    """
    start_idx = 1 if skip_first else 0

    # Convert to torch tensors
    base_torch = torch.from_numpy(logits_base[start_idx:])
    test_torch = torch.from_numpy(logits_test[start_idx:])

    # Compute cosine similarity for all positions at once
    # F.cosine_similarity computes along dim=1 (vocab dimension)
    similarities = F.cosine_similarity(base_torch, test_torch, dim=1)

    return similarities.mean().item()


def compute_perplexity(logits: np.ndarray, target_ids: torch.Tensor) -> float:
    """
    Compute perplexity given logits and target token IDs using torchmetrics.

    Args:
        logits: Logits array [seq_len, vocab_size]
        target_ids: Target token IDs [seq_len]

    Returns:
        Perplexity value
    """
    # Convert to torch and add batch dimension
    logits_torch = torch.from_numpy(logits).unsqueeze(0)  # [1, seq_len, vocab_size]
    target_ids_batch = target_ids.unsqueeze(0)  # [1, seq_len]

    # Initialize perplexity metric (no ignore_index needed - we have no padding)
    metric = Perplexity()

    # Compute perplexity
    # We use logits[:-1] to predict targets[1:] (standard language modeling)
    # This skips the first position (no context) and the last logit (no target)
    perplexity = metric(logits_torch[:, :-1, :], target_ids_batch[:, 1:])

    return perplexity.item()

def compute_perplexity_last_token(logits: np.ndarray, target_id: int) -> float:
    """
    Compute perplexity given logits and target token IDs using torchmetrics.

    Args:
        logits: Logits array [seq_len, vocab_size]
        target_ids: Target token IDs [seq_len]

    Returns:
        Perplexity value
    """
    # Convert to torch and add batch dimension
    logits_torch = torch.from_numpy(logits).unsqueeze(0)  # [1, seq_len, vocab_size]
    target_id_batch = torch.tensor([[target_id]])  # [1, 1]

    # Initialize perplexity metric (no ignore_index needed - we have no padding)
    metric = Perplexity()

    # Compute perplexity
    # We use logits[:-1] to predict targets[1:] (standard language modeling)
    # This skips the first position (no context) and the last logit (no target)
    perplexity = metric(logits_torch[:, -2:-1, :], target_id_batch[:, :])

    return perplexity.item()


def compute_next_token_agreement(
    logits_base: np.ndarray, logits_test: np.ndarray, skip_first: bool = True
) -> float:
    """
    Compute next-token prediction agreement rate.

    Args:
        logits_base: Baseline logits [seq_len, vocab_size]
        logits_test: Test logits [seq_len, vocab_size]
        skip_first: Skip first token (no context)

    Returns:
        Agreement rate (0.0 to 1.0)
    """
    start_idx = 1 if skip_first else 0

    # Get predicted token IDs
    pred_base = np.argmax(logits_base[start_idx:], axis=-1)
    pred_test = np.argmax(logits_test[start_idx:], axis=-1)

    # Compute agreement directly - fraction of matching predictions
    # Note: Using numpy instead of sklearn to avoid warnings about
    # large vocabulary size (262k classes) vs small sample size (~512 tokens)
    agreement = float(np.mean(pred_base == pred_test))

    return agreement


def compute_rouge_scores(
    text_baseline: str, text_test: str
) -> Tuple[float, float, float]:
    """
    Compute ROUGE scores between baseline and test generated text using torchmetrics.

    Args:
        text_baseline: Baseline generated text
        text_test: Test model generated text

    Returns:
        Tuple of (rouge1_f1, rouge2_f1, rougeL_f1)
    """
    rouge = ROUGEScore()
    scores = rouge(preds=[text_test], target=[text_baseline])

    return (
        scores["rouge1_fmeasure"].item(),
        scores["rouge2_fmeasure"].item(),
        scores["rougeL_fmeasure"].item(),
    )


def compute_all_metrics(
    baseline_logits: np.ndarray,
    rust_logits: np.ndarray,
    input_ids: torch.Tensor,
    final_token: int = None,
    skip_first: bool = True,
    baseline_text: str = "",
    rust_text: str = "",
) -> EvaluationMetrics:
    """
    Compute all evaluation metrics.

    Args:
        baseline_logits: Baseline model logits [seq_len, vocab_size]
        rust_logits: Test model logits [seq_len, vocab_size]
        input_ids: Input token IDs [seq_len]
        skip_first: Skip first token when computing similarity/agreement
        baseline_text: Baseline generated text (optional, for ROUGE)
        rust_text: Test model generated text (optional, for ROUGE)

    Returns:
        EvaluationMetrics object containing all computed metrics
    """
    # Compute ROUGE scores if both texts are provided
    rouge1_f1, rouge2_f1, rougeL_f1 = 0.0, 0.0, 0.0
    if baseline_text and rust_text:
        rouge1_f1, rouge2_f1, rougeL_f1 = compute_rouge_scores(baseline_text, rust_text)
    if final_token is None:
        final_token = input_ids[-1].item()
    return EvaluationMetrics(
        cosine_similarity=compute_cosine_similarity(
            baseline_logits, rust_logits, skip_first
        ),
        perplexity_baseline=compute_perplexity(baseline_logits, input_ids),
        perplexity_test=compute_perplexity(rust_logits, input_ids),
        perplexity_last_token_base_line=compute_perplexity_last_token(baseline_logits, final_token),
        perplexity_last_token=compute_perplexity_last_token(rust_logits, final_token),
        agreement=compute_next_token_agreement(
            baseline_logits, rust_logits, skip_first
        ),
        rouge1_f1=rouge1_f1,
        rouge2_f1=rouge2_f1,
        rougeL_f1=rougeL_f1,
    )


def run_smoke_tests(baseline_logits: np.ndarray, rust_logits: np.ndarray) -> bool:
    """
    Run smoke tests to verify data integrity and reasonable similarity.

    Args:
        baseline_logits: Baseline model logits [seq_len, vocab_size]
        rust_logits: Rust model logits [seq_len, vocab_size]

    Returns:
        True if all tests pass, False otherwise
    """
    print_section_header("Running Smoke Tests")

    # Test 1: Check for NaN/Inf values
    if has_invalid_values(baseline_logits):
        print("✗ Baseline logits contain NaN or Inf values", file=sys.stderr)
        return False
    print("✓ Baseline logits: No NaN/Inf values")

    if has_invalid_values(rust_logits):
        print("✗ Rust logits contain NaN or Inf values", file=sys.stderr)
        return False
    print("✓ Rust logits: No NaN/Inf values")

    # Test 2: Check value ranges
    baseline_min, baseline_max = baseline_logits.min(), baseline_logits.max()
    rust_min, rust_max = rust_logits.min(), rust_logits.max()
    min_range, max_range = TYPICAL_LOGIT_RANGE

    if baseline_min < min_range or baseline_max > max_range:
        print(
            f"⚠ Warning: Baseline logits outside typical range: [{baseline_min:.2f}, {baseline_max:.2f}]"
        )
    else:
        print(
            f"✓ Baseline logits in reasonable range: [{baseline_min:.2f}, {baseline_max:.2f}]"
        )

    if rust_min < min_range or rust_max > max_range:
        print(
            f"⚠ Warning: Rust logits outside typical range: [{rust_min:.2f}, {rust_max:.2f}]"
        )
    else:
        print(f"✓ Rust logits in reasonable range: [{rust_min:.2f}, {rust_max:.2f}]")

    # Test 3: Check statistical similarity
    baseline_mean, baseline_std = baseline_logits.mean(), baseline_logits.std()
    rust_mean, rust_std = rust_logits.mean(), rust_logits.std()

    mean_diff_pct = abs(baseline_mean - rust_mean) / abs(baseline_mean) * 100
    std_diff_pct = abs(baseline_std - rust_std) / baseline_std * 100

    print(f"  Baseline: mean={baseline_mean:.4f}, std={baseline_std:.4f}")
    print(f"  Rust:     mean={rust_mean:.4f}, std={rust_std:.4f}")

    if mean_diff_pct > SMOKE_TEST_THRESHOLD_PCT:
        print(f"✗ Mean difference too large: {mean_diff_pct:.2f}%", file=sys.stderr)
        return False
    print(f"✓ Mean difference acceptable: {mean_diff_pct:.2f}%")

    if std_diff_pct > SMOKE_TEST_THRESHOLD_PCT:
        print(
            f"✗ Std deviation difference too large: {std_diff_pct:.2f}%",
            file=sys.stderr,
        )
        return False
    print(f"✓ Std deviation difference acceptable: {std_diff_pct:.2f}%")

    print("✓ All smoke tests passed")
    return True


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


def print_evaluation_results(
    metrics: EvaluationMetrics,
    max_metrics: EvaluationMetrics = None,
    num_windows: int = 1,
) -> None:
    """
    Print formatted evaluation results.

    Args:
        metrics: Computed evaluation metrics (mean values)
        max_metrics: Max evaluation metrics (optional)
        num_windows: Number of windows evaluated (for full test set mode)
    """
    if num_windows > 1:
        print_section_header("EVALUATION RESULTS (Full Test Set)")
        print(f"Total windows evaluated: {num_windows}")
    else:
        print_section_header("EVALUATION RESULTS")

    if max_metrics:
        print(
            f"Cosine similarity: {metrics.cosine_similarity:.6f} (mean) / {max_metrics.cosine_similarity:.6f} (max)"
        )
        print(
            f"Perplexity (baseline): {metrics.perplexity_baseline:.4f} (mean) / {max_metrics.perplexity_baseline:.4f} (max)"
        )
        print(
            f"Perplexity (test): {metrics.perplexity_test:.4f} (mean) / {max_metrics.perplexity_test:.4f} (max)"
        )
        print(f"Perplexity (last token baseline): {metrics.perplexity_last_token_base_line:.4f} (mean) / {max_metrics.perplexity_last_token_base_line:.4f} (max)")
        print(f"Perplexity (last token test): {metrics.perplexity_last_token:.4f} (mean) / {max_metrics.perplexity_last_token:.4f} (max)")
        print(
            f"Perplexity Δ: {metrics.perplexity_delta_pct:+.4f}% (mean) / {max_metrics.perplexity_delta_pct:+.4f}% (max)"
        )
        print(
            f"Next-token match: {metrics.agreement * 100:.2f}% (mean) / {max_metrics.agreement * 100:.2f}% (max)"
        )
        print(
            f"ROUGE-1 F1: {metrics.rouge1_f1:.4f} (mean) / {max_metrics.rouge1_f1:.4f} (max)"
        )
        print(
            f"ROUGE-2 F1: {metrics.rouge2_f1:.4f} (mean) / {max_metrics.rouge2_f1:.4f} (max)"
        )
        print(
            f"ROUGE-L F1: {metrics.rougeL_f1:.4f} (mean) / {max_metrics.rougeL_f1:.4f} (max)"
        )
    else:
        print(f"Cosine similarity: {metrics.cosine_similarity:.6f}")
        print(f"Perplexity (baseline): {metrics.perplexity_baseline:.4f}")
        print(f"Perplexity (test): {metrics.perplexity_test:.4f}")
        print(f"Perplexity (last token baseline): {metrics.perplexity_last_token_base_line:.4f}")
        print(f"Perplexity (last token test): {metrics.perplexity_last_token:.4f}")
        print(f"Perplexity Δ: {metrics.perplexity_delta_pct:+.4f}%")
        print(f"Next-token match: {metrics.agreement * 100:.2f}%")
        print(f"ROUGE-1 F1: {metrics.rouge1_f1:.4f}")
        print(f"ROUGE-2 F1: {metrics.rouge2_f1:.4f}")
        print(f"ROUGE-L F1: {metrics.rougeL_f1:.4f}")

    print(f"{'='*60}")


def format_metrics_as_markdown(
    model_name: str,
    metrics_float: EvaluationMetrics,
    metrics_int: EvaluationMetrics,
    smoke_tests_float: bool,
    smoke_tests_int: bool,
    num_windows: int = 1,
    generated_text_baseline: str = "",
    generated_text_float: str = "",
    generated_text_int: str = "",
    input_text: str = "",
    max_metrics_float: EvaluationMetrics = None,
    max_metrics_int: EvaluationMetrics = None,
) -> str:
    """
    Format evaluation results as markdown for GitHub comments.

    Args:
        model_name: Name of the model being evaluated
        metrics_float: Metrics for float mode (mean values)
        metrics_int: Metrics for int mode (mean values)
        smoke_tests_float: Whether float mode passed smoke tests
        smoke_tests_int: Whether int mode passed smoke tests
        num_windows: Number of windows evaluated
        generated_text_baseline: Baseline generated text (optional, includes input)
        generated_text_float: Float mode generated text (optional, includes input)
        generated_text_int: Integer mode generated text (optional, includes input)
        input_text: Input text to strip from generated text (optional)
        max_metrics_float: Max metrics for float mode (optional)
        max_metrics_int: Max metrics for int mode (optional)

    Returns:
        Markdown-formatted string
    """
    # Determine status emojis
    float_status = (
        "✅" if smoke_tests_float and metrics_float.agreement >= 0.99 else "⚠️"
    )
    int_status = "✅" if smoke_tests_int and metrics_int.agreement >= 0.50 else "⚠️"

    md = f"""## Model Evaluation Results: `{model_name}`

| Metric | Float Mode (Dequantized) {float_status} | Integer Mode (Quantized) {int_status} |
|--------|----------------------------------------|---------------------------------------|
"""

    # Format float mode column
    def format_float_col(val, max_val=None, fmt=".6f", is_pct=False, bold=False):
        suffix = "%" if is_pct else ""
        if max_val is not None:
            formatted = f"`{val:{fmt}}{suffix} (mean) / {max_val:{fmt}}{suffix} (max)`"
        else:
            formatted = f"`{val:{fmt}}{suffix}`"
        return f"**{formatted}**" if bold else formatted

    # Format each metric row
    cosine_float = format_float_col(
        metrics_float.cosine_similarity,
        max_metrics_float.cosine_similarity if max_metrics_float else None,
    )
    cosine_int = format_float_col(
        metrics_int.cosine_similarity,
        max_metrics_int.cosine_similarity if max_metrics_int else None,
    )

    perp_base_float = format_float_col(
        metrics_float.perplexity_baseline,
        max_metrics_float.perplexity_baseline if max_metrics_float else None,
        ".4f",
    )
    perp_base_int = format_float_col(
        metrics_int.perplexity_baseline,
        max_metrics_int.perplexity_baseline if max_metrics_int else None,
        ".4f",
    )

    perp_test_float = format_float_col(
        metrics_float.perplexity_test,
        max_metrics_float.perplexity_test if max_metrics_float else None,
        ".4f",
    )
    perp_test_int = format_float_col(
        metrics_int.perplexity_test,
        max_metrics_int.perplexity_test if max_metrics_int else None,
        ".4f",
    )

    perp_delta_float = format_float_col(
        metrics_float.perplexity_delta_pct,
        max_metrics_float.perplexity_delta_pct if max_metrics_float else None,
        "+.4f",
        is_pct=True,
    )
    perp_delta_int = format_float_col(
        metrics_int.perplexity_delta_pct,
        max_metrics_int.perplexity_delta_pct if max_metrics_int else None,
        "+.4f",
        is_pct=True,
    )

    agreement_float = format_float_col(
        metrics_float.agreement * 100,
        max_metrics_float.agreement * 100 if max_metrics_float else None,
        ".2f",
        is_pct=True,
        bold=True,
    )
    agreement_int = format_float_col(
        metrics_int.agreement * 100,
        max_metrics_int.agreement * 100 if max_metrics_int else None,
        ".2f",
        is_pct=True,
        bold=True,
    )

    rouge1_float = format_float_col(
        metrics_float.rouge1_f1,
        max_metrics_float.rouge1_f1 if max_metrics_float else None,
        ".4f",
    )
    rouge1_int = format_float_col(
        metrics_int.rouge1_f1,
        max_metrics_int.rouge1_f1 if max_metrics_int else None,
        ".4f",
    )

    rouge2_float = format_float_col(
        metrics_float.rouge2_f1,
        max_metrics_float.rouge2_f1 if max_metrics_float else None,
        ".4f",
    )
    rouge2_int = format_float_col(
        metrics_int.rouge2_f1,
        max_metrics_int.rouge2_f1 if max_metrics_int else None,
        ".4f",
    )

    rougeL_float = format_float_col(
        metrics_float.rougeL_f1,
        max_metrics_float.rougeL_f1 if max_metrics_float else None,
        ".4f",
    )
    rougeL_int = format_float_col(
        metrics_int.rougeL_f1,
        max_metrics_int.rougeL_f1 if max_metrics_int else None,
        ".4f",
    )

    md += f"""| Cosine Similarity | {cosine_float} | {cosine_int} |
| Perplexity (Baseline) | {perp_base_float} | {perp_base_int} |
| Perplexity (ZKML) | {perp_test_float} | {perp_test_int} |
| Perplexity Δ | {perp_delta_float} | {perp_delta_int} |
| Next-Token Match | {agreement_float} | {agreement_int} |
| Smoke Tests | `{"PASS" if smoke_tests_float else "FAIL"}` | `{"PASS" if smoke_tests_int else "FAIL"}` |
| ROUGE-1 F1 | {rouge1_float} | {rouge1_int} |
| ROUGE-2 F1 | {rouge2_float} | {rouge2_int} |
| ROUGE-L F1 | {rougeL_float} | {rougeL_int} |

"""

    # Add generated text if available
    if generated_text_baseline or generated_text_float or generated_text_int:
        md += "### Generated Text\n\n"

        # All generated text strings now contain only generated portion (no input)
        if generated_text_baseline:
            md += f"**Baseline (PyTorch):** {generated_text_baseline}\n\n"
        if generated_text_float:
            md += f"**Float Mode (Dequantized):** {generated_text_float}\n\n"
        if generated_text_int:
            md += f"**Integer Mode (Quantized):** {generated_text_int}\n\n"

    if num_windows > 1:
        md += f"\n*Evaluated on {num_windows} sliding windows*\n"

    return md


def run_full_test_set_evaluation(
    model_path: Path, tokenizer: AutoTokenizer, args: argparse.Namespace
) -> Tuple[
    EvaluationMetrics,
    EvaluationMetrics,
    EvaluationMetrics,
    EvaluationMetrics,
    bool,
    bool,
]:
    """
    Run evaluation on full WikiText-103 test set with sliding window.

    Returns:
        Tuple of (avg_metrics_float, max_metrics_float, avg_metrics_int,
                  max_metrics_int, smoke_tests_float, smoke_tests_int)
    """
    if args.text:
        print("⚠ Warning: --text ignored when using --full-test-set", file=sys.stderr)

    print_section_header("FULL TEST SET EVALUATION (Sliding Window)")
    print(f"Stride: {args.stride}")
    print(f"Max length: {args.max_tokens}")
    print(
        f"Windows: {'ALL (evaluate entire test set)' if args.num_windows == 0 else f'{args.num_windows} (sample)'}"
    )

    # Accumulate metrics across all windows
    all_metrics_float = []
    all_metrics_int = []
    smoke_tests_float = True
    smoke_tests_int = True

    for num_window, (text, tokens) in enumerate(
        get_wikitext_full_test(
            tokenizer,
            stride=args.stride,
            max_length=args.max_tokens,
            num_windows=args.num_windows,
        ),
        start=1,
    ):
        # Run both models on this window
        model_output = run_baseline_model(
            str(model_path), text, tokenizer, args.generate_tokens
        )
        logits_float, logits_int = run_rust_model(
            model_path, text, args.generate_tokens
        )
        logits_float = logits_float[0]
        logits_int = logits_int[0]

        # Always compute generated text from logits using argmax for all modes
        # This ensures ROUGE metrics compare argmax predictions across all models
        num_input_tokens = len(model_output.input_ids)

        baseline_generated_from_logits = ""
        if args.generate_tokens > 0:
            # Compute baseline generated text from baseline logits using argmax
            baseline_generated_from_logits = compute_generated_text_from_logits(
                model_output.logits, tokenizer, num_input_tokens, args.generate_tokens
            )
            # Compute ZKML float generated text from float logits using argmax
            generated_text_float = compute_generated_text_from_logits(
                logits_float, tokenizer, num_input_tokens, args.generate_tokens
            )
            # Compute ZKML int generated text from int logits using argmax
            generated_text_int = compute_generated_text_from_logits(
                logits_int, tokenizer, num_input_tokens, args.generate_tokens
            )

        # Verify shapes match
        if model_output.logits.shape != logits_float.shape:
            print(
                f"\n✗ Float mode shape mismatch in window {num_window}!",
                file=sys.stderr,
            )
            continue

        if model_output.logits.shape != logits_int.shape:
            print(
                f"\n✗ Int mode shape mismatch in window {num_window}!", file=sys.stderr
            )
            continue

        # Compute metrics for this window
        metrics_float = compute_all_metrics(
            model_output.logits,
            logits_float,
            model_output.input_ids,
            skip_first=True,
            baseline_text=baseline_generated_from_logits,
            rust_text=generated_text_float,
        )
        metrics_int = compute_all_metrics(
            model_output.logits,
            logits_int,
            model_output.input_ids,
            skip_first=True,
            baseline_text=baseline_generated_from_logits,
            rust_text=generated_text_int,
        )
        all_metrics_float.append(metrics_float)
        all_metrics_int.append(metrics_int)

        if num_window % 10 == 0:
            print(f"  Processed {num_window} windows...")

    # Compute average and max metrics
    avg_metrics_float, max_metrics_float = compute_average_metrics(all_metrics_float)
    avg_metrics_int, max_metrics_int = compute_average_metrics(all_metrics_int)

    # Print results
    print_section_header("FLOAT MODE (Full Test Set)")
    print_evaluation_results(
        avg_metrics_float, max_metrics_float, num_windows=len(all_metrics_float)
    )

    print_section_header("INTEGER MODE (Full Test Set)")
    print_evaluation_results(
        avg_metrics_int, max_metrics_int, num_windows=len(all_metrics_int)
    )

    return (
        avg_metrics_float,
        max_metrics_float,
        avg_metrics_int,
        max_metrics_int,
        smoke_tests_float,
        smoke_tests_int,
    )


def run_single_sample_evaluation(
    model_path: Path, tokenizer: AutoTokenizer, args: argparse.Namespace
) -> Tuple[
    EvaluationMetrics,
    EvaluationMetrics,
    EvaluationMetrics,
    EvaluationMetrics,
    bool,
    bool,
    str,
    str,
    str,
    str,
]:
    """
    Run evaluation on a single text sample (fast, for debugging).

    Returns:
        Tuple of (metrics_float, max_metrics_float, metrics_int, max_metrics_int,
                  smoke_tests_float, smoke_tests_int,
                  generated_text_baseline, generated_text_float, generated_text_int, input_text)
    """
    # Get evaluation text
    if args.text:
        text = args.text
        tokens = tokenizer.encode(text)
        if len(tokens) > args.max_tokens:
            tokens = tokens[: args.max_tokens]
            text = tokenizer.decode(tokens)
        print(f"Using custom text ({len(tokens)} tokens)")
    elif args.dataset_sample == "wikitext":
        text, tokens, final_token = get_wikitext_sample(tokenizer, max_tokens=args.max_tokens)
    else:
        print("✗ Must provide --text or use --dataset-sample wikitext", file=sys.stderr)
        sys.exit(1)

    print(f"\nEvaluation text preview: {text[:100]}...")
    print(f"Total tokens: {len(tokens)}")


    # Run baseline model
    model_output = run_baseline_model(
        str(model_path), text, tokenizer, args.generate_tokens
    )

    # Run Rust model - get both float and int logits
    logits_float, logits_int = run_rust_model(model_path, text, args.generate_tokens)
    logits_float = logits_float[0]
    logits_int = logits_int[0]

    # Always compute generated text from logits using argmax for all modes
    # This ensures ROUGE metrics compare argmax predictions across all models
    # Note: model_output.input_ids contains the FULL sequence (input + generated), not just input
    # Calculate the original input token count from the total sequence length
    total_tokens = len(model_output.input_ids)
    num_input_tokens = total_tokens - args.generate_tokens

    baseline_generated_from_logits = ""
    generated_text_float = ""
    generated_text_int = ""
    if args.generate_tokens > 0:
        # Compute baseline generated text from baseline logits using argmax
        baseline_generated_from_logits = compute_generated_text_from_logits(
            model_output.logits, tokenizer, num_input_tokens, args.generate_tokens
        )
        # Compute ZKML float generated text from float logits using argmax
        generated_text_float = compute_generated_text_from_logits(
            logits_float, tokenizer, num_input_tokens, args.generate_tokens
        )
        # Compute ZKML int generated text from int logits using argmax
        generated_text_int = compute_generated_text_from_logits(
            logits_int, tokenizer, num_input_tokens, args.generate_tokens
        )

    #Print the full text 
    full_text_float = compute_full_text_from_logits(logits_float, tokenizer)
    full_text_int = compute_full_text_from_logits(logits_int, tokenizer)
    print_section_header("FULL GENERATED TEXT (FLOAT MODE)")
    print(full_text_float)
    print_section_header("FULL GENERATED TEXT (INT MODE)")
    print(full_text_int)

    # Verify shapes match
    if model_output.logits.shape != logits_float.shape:
        print(f"\n✗ Float mode shape mismatch!", file=sys.stderr)
        print(f"  Baseline: {model_output.logits.shape}", file=sys.stderr)
        print(f"  Float: {logits_float.shape}", file=sys.stderr)
        sys.exit(1)

    if model_output.logits.shape != logits_int.shape:
        print(f"\n✗ Int mode shape mismatch!", file=sys.stderr)
        print(f"  Baseline: {model_output.logits.shape}", file=sys.stderr)
        print(f"  Int: {logits_int.shape}", file=sys.stderr)
        sys.exit(1)

    # Run smoke tests
    smoke_tests_passed_float = run_smoke_tests_with_warning(
        model_output.logits, logits_float, "Float"
    )
    smoke_tests_passed_int = run_smoke_tests_with_warning(
        model_output.logits, logits_int, "Integer"
    )

    # Compute metrics
    print_section_header("Computing Evaluation Metrics")

    # Float mode metrics
    metrics_float = compute_all_metrics(
        model_output.logits,
        logits_float,
        model_output.input_ids,
        final_token,
        skip_first=True,
        baseline_text=baseline_generated_from_logits,
        rust_text=generated_text_float,
    )

    # Int mode metrics
    metrics_int = compute_all_metrics(
        model_output.logits,
        logits_int,
        model_output.input_ids,
        final_token,
        skip_first=True,
        baseline_text=baseline_generated_from_logits,
        rust_text=generated_text_int,
    )

    # Print results for float mode
    print_section_header("FLOAT MODE (Dequantized from Quantized Inference)")
    print_evaluation_results(metrics_float, max_metrics=metrics_float)

    # Print results for int mode
    print_section_header("INTEGER MODE (Quantized)")
    print_evaluation_results(metrics_int, max_metrics=metrics_int)

    return (
        metrics_float,
        metrics_float,  # max = mean for single sample
        metrics_int,
        metrics_int,  # max = mean for single sample
        smoke_tests_passed_float,
        smoke_tests_passed_int,
        model_output.generated_only_text,
        generated_text_float,
        generated_text_int,
        text,
    )


def run_comparison_sample_evaluations(
    model_path: Path, tokenizer: AutoTokenizer, args: argparse.Namespace
) -> Tuple[
    EvaluationMetrics,
    EvaluationMetrics,
    EvaluationMetrics,
    EvaluationMetrics,
]:
    """
    Run evaluation on multiple text samples of a fixed length.

    Returns:
        Tuple of (metrics_float, max_metrics_float, metrics_int, max_metrics_int,
                  smoke_tests_float, smoke_tests_int,
                  generated_text_baseline, generated_text_float, generated_text_int, input_text)
    """
    # Get evaluation text
    sample = load_wikitext(tokenizer, block_size=args.max_tokens)
    full_text = tokenizer.decode(sample)
    samples = get_wikitext2_samples(
        tokenizer, max_tokens=args.max_tokens, max_docs=args.samples
    )
    all_metrics_float = []
    all_metrics_int = []
    all_metrics_zkgpt = []

    for i, (text, tokens, final_token) in enumerate(samples, start=1):
        print_section_header(f"SAMPLE {i} EVALUATION (Length: {len(tokens)} tokens)")
        print(f"\nEvaluation text preview: {text[:100]}...")
    
        # Run baseline model
        model_output = run_baseline_model(
             str(model_path), text, tokenizer, 0
         )

        # Run Rust model - get both float and int logits
        logits_float, logits_int = run_rust_model(model_path, text, 0)
        logits_float = logits_float[:-1]
        logits_int = logits_int[:-1]
        

        # Always compute generated text from logits using argmax for all modes
        # This ensures ROUGE metrics compare argmax predictions across all models
        # Note: model_output.input_ids contains the FULL sequence (input + generated), not just input
        # Calculate the original input token count from the total sequence length
        total_tokens = len(model_output.input_ids)
        num_input_tokens = total_tokens - args.generate_tokens

        baseline_generated_from_logits = ""
        
        if args.generate_tokens > 0:
            # Compute baseline generated text from baseline logits using argmax
            baseline_generated_from_logits = compute_generated_text_from_logits(
                model_output.logits, tokenizer, 1, 31
            )   
            # Compute ZKML float generated text from float logits using argmax
            generated_text_float = compute_generated_text_from_logits(
                logits_float, tokenizer, 1, 31
            )
            # Compute ZKML int generated text from int logits using argmax
            generated_text_int = compute_generated_text_from_logits(
                logits_int, tokenizer, 1, 31
            )
            

            # Verify shapes match
        if model_output.logits.shape != logits_float.shape:
            print(f"\n✗ Float mode shape mismatch!", file=sys.stderr)
            print(f"  Baseline: {model_output.logits.shape}", file=sys.stderr)
            print(f"  Float: {logits_float.shape}", file=sys.stderr)
            sys.exit(1)
        if model_output.logits.shape != logits_int.shape:
            print(f"\n✗ Int mode shape mismatch!", file=sys.stderr)
            print(f"  Baseline: {model_output.logits.shape}", file=sys.stderr)
            print(f"  Int: {logits_int.shape}", file=sys.stderr)
            sys.exit(1)

        

        # Compute metrics
        print_section_header(f"Computing Evaluation Metrics Sample {i}")

        # Float mode metrics
        metrics_float = compute_all_metrics(
            model_output.logits,
            logits_float,
            model_output.input_ids,
            final_token,
            skip_first=True,
            baseline_text=baseline_generated_from_logits,
            rust_text=generated_text_float,
        )

        # Int mode metrics
        metrics_int = compute_all_metrics(
            model_output.logits,
            logits_int,
            model_output.input_ids,
            final_token,
            skip_first=True,
            baseline_text=baseline_generated_from_logits,
            rust_text=generated_text_int,
        )

        
        all_metrics_float.append(metrics_float)
        all_metrics_int.append(metrics_int)
        

        if i % 10 == 0:
            print(f"  Processed {i} samples...")
            # Compute average and max metrics
            tmp_avg_metrics_float, tmp_max_metrics_float = compute_average_metrics(all_metrics_float)
            tmp_avg_metrics_int, tmp_max_metrics_int = compute_average_metrics(all_metrics_int)
            print_section_header(f"INTERIM RESULTS AFTER {i} SAMPLES")
            print_evaluation_results(
                tmp_avg_metrics_float, tmp_max_metrics_float, num_windows=len(all_metrics_float)
            )
            print_evaluation_results(
                tmp_avg_metrics_int, tmp_max_metrics_int, num_windows=len(all_metrics_int)
            )
            


    # Compute average and max metrics
    avg_metrics_float, max_metrics_float = compute_average_metrics(all_metrics_float)
    avg_metrics_int, max_metrics_int = compute_average_metrics(all_metrics_int)
    

    # Print results
    print_section_header("FLOAT MODE (Full Test Set)")
    print_evaluation_results(
        avg_metrics_float, max_metrics_float, num_windows=len(all_metrics_float)
    )

    print_section_header("INTEGER MODE (Full Test Set)")
    print_evaluation_results(
        avg_metrics_int, max_metrics_int, num_windows=len(all_metrics_int)
    )

    

    return (
        avg_metrics_float,
        max_metrics_float,
        avg_metrics_int,
        max_metrics_int,
    )



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
    if args.full_test_set:
        (
            avg_metrics_float,
            max_metrics_float,
            avg_metrics_int,
            max_metrics_int,
            smoke_float,
            smoke_int,
        ) = run_full_test_set_evaluation(model_path, tokenizer, args)

        # Output markdown if requested
        if args.markdown:
            num_windows = args.num_windows if args.num_windows > 0 else "all"
            markdown_output = format_metrics_as_markdown(
                args.model,
                avg_metrics_float,
                avg_metrics_int,
                smoke_float,
                smoke_int,
                num_windows=num_windows if isinstance(num_windows, int) else 0,
                max_metrics_float=max_metrics_float,
                max_metrics_int=max_metrics_int,
            )

            if args.output_file:
                output_path = Path(args.output_file)
                output_path.write_text(markdown_output)
                print(f"\n✓ Markdown output written to: {args.output_file}")
            else:
                print_section_header("MARKDOWN OUTPUT")
                print(markdown_output)
    elif args.samples > 1:
        _ = run_comparison_sample_evaluations(model_path, tokenizer, args)
        
    else:
        (
            metrics_float,
            max_metrics_float,
            metrics_int,
            max_metrics_int,
            smoke_float,
            smoke_int,
            gen_text_baseline,
            gen_text_float,
            gen_text_int,
            input_text,
        ) = run_single_sample_evaluation(model_path, tokenizer, args)

        # Output markdown if requested
        if args.markdown:
            markdown_output = format_metrics_as_markdown(
                args.model,
                metrics_float,
                metrics_int,
                smoke_float,
                smoke_int,
                num_windows=1,
                generated_text_baseline=gen_text_baseline,
                generated_text_float=gen_text_float,
                generated_text_int=gen_text_int,
                input_text=input_text,
                max_metrics_float=max_metrics_float,
                max_metrics_int=max_metrics_int,
            )

            if args.output_file:
                output_path = Path(args.output_file)
                output_path.write_text(markdown_output)
                print(f"\n✓ Markdown output written to: {args.output_file}")
            else:
                print_section_header("MARKDOWN OUTPUT")
                print(markdown_output)

    print(f"\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
