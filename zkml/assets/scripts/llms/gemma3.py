"""
Run Gemma 3 (PyTorch/Transformers) and dump final-token logits for 3 input sizes.

Outputs JSON array to assets/scripts/llms/gemma3_logits_output.json with objects:
  { "input_token": [u32], "input_text": str, "logits": [f32] }
"""

import argparse
import json
import os
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_tokens_and_text(tokenizer: AutoTokenizer, target_tokens: int) -> Tuple[List[int], str]:
    seed = (
        "The morning sky over the city carried a calm brightness. "
        "People walked to work and cafés opened their doors. "
        "A developer reviewed code, thinking about clearer names and simpler designs. "
        "A teacher prepared a short lesson about how maps tell stories and why scale matters. "
        "In the background, a radio show explained how memory and attention work together when we read."
    )
    filler = (
        " Artificial intelligence systems help summarize long texts, highlight key ideas, "
        "and generate drafts that humans later refine. Good prompts clarify goals and constraints. "
        "When we test models, we compare outputs across different lengths to study stability and calibration. "
        "We care about correctness, clarity, and the ability to reason through intermediate steps."
    )

    text = seed
    while True:
        input_ids = tokenizer.encode(text, add_special_tokens=True)
        if len(input_ids) >= target_tokens:
            trimmed = input_ids[: target_tokens]
            decoded = tokenizer.decode(trimmed, skip_special_tokens=True)
            return trimmed, decoded
        text += "\n\n" + filler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-3-270m-it",
        help="Hugging Face model id (e.g. google/gemma-3-270m-it)",
    )
    parser.add_argument(
        "--token-lens",
        type=str,
        default="10,100,500",
        help="Comma-separated target token lengths",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.to(device)
    model.eval()

    token_targets = [int(x) for x in args.token_lens.split(",") if x.strip()]
    results = []
    with torch.no_grad():
        for tlen in token_targets:
            input_ids_list, input_text = build_tokens_and_text(tokenizer, tlen)
            input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, return_dict=True)
            last_logits = outputs.logits[0, -1, :].float().detach().cpu().tolist()
            results.append(
                {
                    "input_token": input_ids_list,
                    "input_text": input_text,
                    "logits": last_logits,
                }
            )

    out_path = os.path.join(os.path.dirname(__file__), "gemma3_logits_output.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote logits to {out_path}")


if __name__ == "__main__":
    main()
