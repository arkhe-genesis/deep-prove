"""
Run Llama2 (PyTorch/Transformers) and dump final-token logits for validation.

Outputs to llama2_logits.json with data including embeddings, attention
intermediates, and final logits for comparison with Rust implementation.
"""

import argparse
import json
import os
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_tokens_and_text(
    tokenizer: AutoTokenizer, target_tokens: int
) -> Tuple[List[int], str]:
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
            trimmed = input_ids[:target_tokens]
            decoded = tokenizer.decode(trimmed, skip_special_tokens=True)
            return trimmed, decoded
        text += "\n\n" + filler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model id (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    parser.add_argument(
        "--token-lens",
        type=str,
        default="10,50",
        help="Comma-separated target token lengths",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.to(device)
    model.eval()

    # Just use one token length for now
    tlen = int(args.token_lens.split(",")[0].strip())
    print(f"Processing {tlen} tokens...")

    input_ids_list, input_text = build_tokens_and_text(tokenizer, tlen)
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)

    with torch.no_grad():
        # Get embeddings first
        embeddings = model.model.embed_tokens(input_ids)
        print(f"Embeddings shape: {embeddings.shape}")  # [1, seq_len, hidden_size]

        # Get first decoder block intermediates
        hidden_states = embeddings
        first_block = model.model.layers[0]
        attn = first_block.self_attn

        # Input layernorm -> Q, K, V projections
        after_input_norm = first_block.input_layernorm(hidden_states)

        # Q, K, V projections (before reshape/RoPE)
        q_proj_raw = attn.q_proj(after_input_norm)  # [batch, seq, num_heads * head_dim]
        k_proj_raw = attn.k_proj(after_input_norm)  # [batch, seq, num_kv_heads * head_dim]
        v_proj_raw = attn.v_proj(after_input_norm)  # [batch, seq, num_kv_heads * head_dim]

        # Get config values
        num_heads = model.config.num_attention_heads  # 32
        num_kv_heads = model.config.num_key_value_heads  # 4
        head_dim = model.config.hidden_size // num_heads  # 64
        heads_per_group = num_heads // num_kv_heads  # 8
        seq_len = q_proj_raw.shape[1]

        # Reshape to match Rust layout: [heads_per_group, num_groups, seq, head_dim]
        # Python raw: [batch, seq, num_heads * head_dim]
        # First reshape to [batch, seq, num_groups, heads_per_group, head_dim]
        # Then permute to [batch, heads_per_group, num_groups, seq, head_dim]
        q_proj = q_proj_raw.view(1, seq_len, num_kv_heads, heads_per_group, head_dim)
        q_proj = q_proj.permute(0, 3, 2, 1, 4)  # [batch, heads_per_group, num_groups, seq, head_dim]

        # K and V: [batch, seq, num_kv_heads * head_dim] -> [batch, num_groups, seq, head_dim]
        k_proj = k_proj_raw.view(1, seq_len, num_kv_heads, head_dim)
        k_proj = k_proj.permute(0, 2, 1, 3)  # [batch, num_groups, seq, head_dim]

        v_proj = v_proj_raw.view(1, seq_len, num_kv_heads, head_dim)
        v_proj = v_proj.permute(0, 2, 1, 3)  # [batch, num_groups, seq, head_dim]

        print(f"Q proj shape (reshaped): {q_proj.shape}")
        print(f"K proj shape (reshaped): {k_proj.shape}")
        print(f"V proj shape (reshaped): {v_proj.shape}")

        # Apply RoPE to Q and K
        # Need to reshape for HF's apply_rotary_pos_emb
        # HF expects [batch, num_heads, seq_len, head_dim]
        q_for_rope = q_proj_raw.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        k_for_rope = k_proj_raw.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # Get position ids
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # Get rotary embeddings (rotary_emb is on model.model, not attention)
        cos, sin = model.model.rotary_emb(k_for_rope, position_ids)

        # Apply RoPE
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        q_after_rope, k_after_rope = apply_rotary_pos_emb(q_for_rope, k_for_rope, cos, sin)
        print(f"Q after RoPE shape: {q_after_rope.shape}")  # [batch, num_heads, seq, head_dim]
        print(f"K after RoPE shape: {k_after_rope.shape}")  # [batch, num_kv_heads, seq, head_dim]

        # Reshape to match Rust layout: Q=[heads_per_group, num_groups, seq, head_dim], K=[num_groups, seq, head_dim]
        q_rope_reshaped = q_after_rope.view(1, num_kv_heads, heads_per_group, seq_len, head_dim)
        q_rope_reshaped = q_rope_reshaped.permute(0, 2, 1, 3, 4)  # [batch, heads_per_group, num_groups, seq, head_dim]
        k_rope_reshaped = k_after_rope  # Already [batch, num_groups, seq, head_dim]
        print(f"Q after RoPE (reshaped): {q_rope_reshaped.shape}")
        print(f"K after RoPE (reshaped): {k_rope_reshaped.shape}")

        # Compute attention scores and softmax
        # Q: [batch, num_heads, seq, head_dim], K: [batch, num_kv_heads, seq, head_dim]
        # Need to repeat K for GQA: each KV head serves multiple Q heads
        k_expanded = k_after_rope.unsqueeze(2).expand(-1, -1, heads_per_group, -1, -1)
        k_expanded = k_expanded.reshape(1, num_heads, seq_len, head_dim)  # [batch, num_heads, seq, head_dim]

        # Attention scores: Q @ K^T / sqrt(head_dim)
        attn_weights = torch.matmul(q_after_rope, k_expanded.transpose(-2, -1)) / (head_dim ** 0.5)
        print(f"Attention weights (before softmax) shape: {attn_weights.shape}")  # [batch, num_heads, seq, seq]

        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        # Softmax
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
        print(f"Attention probs (after softmax) shape: {attn_probs.shape}")

        # Reshape to match Rust layout: [heads_per_group, num_groups, seq, seq]
        attn_probs_reshaped = attn_probs.view(1, num_kv_heads, heads_per_group, seq_len, seq_len)
        attn_probs_reshaped = attn_probs_reshaped.permute(0, 2, 1, 3, 4)
        print(f"Attention probs (reshaped): {attn_probs_reshaped.shape}")

        # VERIFICATION: Compare our manual computation with model's actual attention output
        # Run first block's attention and compare
        v_for_attn = v_proj_raw.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v_expanded = v_for_attn.unsqueeze(2).expand(-1, -1, heads_per_group, -1, -1)
        v_expanded = v_expanded.reshape(1, num_heads, seq_len, head_dim)

        # Manual attention output: softmax @ V
        manual_attn_output = torch.matmul(attn_probs, v_expanded)  # [batch, num_heads, seq, head_dim]
        manual_attn_output = manual_attn_output.transpose(1, 2).reshape(1, seq_len, num_heads * head_dim)

        # Apply output projection
        manual_after_o_proj = attn.o_proj(manual_attn_output)

        # Get position embeddings (cos, sin) for the attention
        position_embeddings = (cos, sin)

        # Create causal attention mask
        attention_mask_4d = torch.zeros(1, 1, seq_len, seq_len, device=device)
        attention_mask_4d = attention_mask_4d.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool(),
            float('-inf')
        )

        # Get the actual attention output from the model
        attn_result = first_block.self_attn(
            hidden_states=after_input_norm,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask_4d,
        )
        actual_attn_output = attn_result[0]

        # Compare
        diff = (manual_after_o_proj - actual_attn_output).abs()
        max_diff = diff.max().item()
        print(f"VERIFICATION: Manual vs Model attention output max_diff: {max_diff:.8f}")
        if max_diff > 1e-4:
            print(f"  WARNING: Large difference! Manual first 5: {manual_after_o_proj[0, 0, :5].tolist()}")
            print(f"  WARNING: Model first 5: {actual_attn_output[0, 0, :5].tolist()}")
        else:
            print("  Manual computation matches model output!")

        # First residual add: embeddings + attention_output
        after_first_residual = hidden_states + actual_attn_output
        print(f"After first residual add shape: {after_first_residual.shape}")

        # Forward pass with native SiLU activation
        outputs = model(input_ids=input_ids, return_dict=True)
        last_logits = outputs.logits[0, -1, :].float().detach().cpu()
        print(f"Model logits shape: {outputs.logits.shape}")

    result = {
        "input_token": input_ids_list,
        "input_text": input_text,
        "embeddings": embeddings[0].float().detach().cpu().tolist(),  # [seq_len, hidden_size]
        "q_proj_0": q_proj[0].float().detach().cpu().numpy().flatten().tolist(),  # flattened
        "k_proj_0": k_proj[0].float().detach().cpu().numpy().flatten().tolist(),  # flattened
        "v_proj_0": v_proj[0].float().detach().cpu().numpy().flatten().tolist(),  # flattened
        "q_rope_0": q_rope_reshaped[0].float().detach().cpu().numpy().flatten().tolist(),  # flattened
        "k_rope_0": k_rope_reshaped[0].float().detach().cpu().numpy().flatten().tolist(),  # flattened
        "attn_softmax_0": attn_probs_reshaped[0].float().detach().cpu().numpy().flatten().tolist(),  # flattened
        # softmax @ V (before output projection) - reshape to [heads_per_group, num_groups, seq, head_dim]
        # HuggingFace: [1, 32, seq, 64] where head i = group (i//8), head_in_group (i%8)
        # Rust expects: [8, 4, seq, 64] = [heads_per_group, num_groups, seq, head_dim]
        # So reshape to [4, 8, seq, 64] then permute to [8, 4, seq, 64]
        "attn_value_0": manual_attn_output.reshape(1, num_kv_heads, heads_per_group, seq_len, head_dim).permute(0, 2, 1, 3, 4).contiguous()[0].float().detach().cpu().numpy().flatten().tolist(),
        "attn_output_0": actual_attn_output[0].float().detach().cpu().numpy().flatten().tolist(),  # [seq, hidden_size] - use actual model output
        "after_first_residual_0": after_first_residual[0].float().detach().cpu().numpy().flatten().tolist(),  # [seq, hidden_size]
        "final_proj_output": outputs.logits[0, -1, :].float().detach().cpu().tolist(),  # last token logits (same as logits)
        "logits": last_logits.tolist(),
    }

    results = [result]

    # Choose output filename
    out_filename = "llama2_logits.json"
    out_path = os.path.join(os.path.dirname(__file__), out_filename)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote output to {out_path}")


if __name__ == "__main__":
    main()
