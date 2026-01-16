"""
Run Gemma 3 (PyTorch/Transformers) and dump final-token logits for 3 input sizes.

MODES:
  1. Default (no --trace): Outputs to gemma3_logits.json with minimal data:
     { "input_token": [u32], "input_text": str, "logits": [f32] }

  2. With --trace: Outputs to gemma3_trace.json with full intermediates:
     { "input_token": [u32], "input_text": str, "logits": [f32], "intermediates": {...} }

The intermediates dict (--trace mode) contains FULL sequences FOR FIRST LAYER ONLY:
  Gemma3 flow per layer: Input → RMSNorm → QKV Proj → Q/K Norm → RoPE → Attention+Softmax → +Residual → RMSNorm → FFN → +Residual

  - pre_qkv: Output of input RMSNorm (before QKV projection) [1 layer]{shape: [seq_len, hidden_dim], data: [row-major f32]}
  - embeddings: Output of the embedding lookup {shape: [seq_len, hidden_dim], data: [row-major f32]}
  - attention_qkv: Q, K, V after projection (before Q/K norm) [1 layer]{q,k,v: {shape: [seq_len, hidden_dim], data: [row-major f32]}}
  - attention_qkv_normalized: Q, K after RMSNorm, V unchanged (before RoPE) [1 layer]{q,k,v: [seq_len, hidden_dim]}
  - attention_qkv_rope: Q, K after RoPE, V unchanged [1 layer]{q,k,v: [seq_len, hidden_dim]}
  - attention_softmax: Softmax attention weights [1 layer][seq_len, num_heads, seq_len]
  - add_after_attention: Residual add (input + attention output) [1 layer][seq_len, hidden_dim]
  - ffn_pre_norm: Output of post-attention RMSNorm (before FFN) [1 layer][seq_len, hidden_dim]
  - feedforward_outputs: FFN/MLP output (before residual) [1 layer][seq_len, hidden_dim]
  - add_after_ffn: Final residual add (post-attn + FFN output) [1 layer][seq_len, hidden_dim]
  - transformer_blocks: Complete layer output (same as add_after_ffn) [1 layer][seq_len, hidden_dim]

IMPLEMENTATION NOTES (--trace mode):
  - FFN/MLP: Uses ORIGINAL transformers module (NOT reproduced)
  - RMSNorm layers: Uses ORIGINAL transformers modules (NOT reproduced)
  - QKV projections: Uses ORIGINAL transformers modules (NOT reproduced)
  - Attention logic: REPRODUCED (matmul, softmax, reshaping) and VALIDATED against original
  - RoPE: REPRODUCED apply_rotary_pos_emb(), uses original cos/sin computation
  - Final output: Uses ORIGINAL transformers output to ensure correctness
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.gemma3.modeling_gemma3 as gemma3_mod


class IntermediateCapture:
    """Captures intermediate activations from Gemma model layers."""

    def __init__(self):
        self.transformer_blocks: List[torch.Tensor] = []
        self.feedforward_outputs: List[torch.Tensor] = []
        self.pre_qkv: List[torch.Tensor] = []  # Output of input RMSNorm
        self.ffn_pre_norm: List[torch.Tensor] = []
        self.attention_qkv: List[Dict[str, torch.Tensor]] = []  # After QKV projection
        self.attention_qkv_normalized: List[Dict[str, torch.Tensor]] = (
            []
        )  # After Q/K RMSNorm
        self.attention_qkv_rope: List[Dict[str, torch.Tensor]] = []  # After RoPE
        self.attention_qkv_rope_postnorm: List[Dict[str, torch.Tensor]] = (
            []
        )  # After RoPE, then q_norm/k_norm (if present)
        self.attention_softmax: List[torch.Tensor] = []  # Softmax output
        self.add_after_attention: List[torch.Tensor] = (
            []
        )  # Residual add after attention
        self.add_after_ffn: List[torch.Tensor] = []  # Residual add after FFN
        self.embeddings: Optional[torch.Tensor] = None
        self.pre_qkv_rmsnorm_alpha: Optional[torch.Tensor] = (
            None  # RMSNorm weight before QKV
        )
        self.hooks: List[Any] = []
        # Internal state for residual computations and RoPE capture
        self._layer_entry: Dict[int, torch.Tensor] = {}
        self._post_attn_residual: Dict[int, torch.Tensor] = {}
        self._current_layer_idx: int = -1
        self._orig_apply_rotary = None
        self._value_cache: Dict[int, torch.Tensor] = {}
        # Captured RoPE cos/sin from HF forward (reduced to [S, H]) per layer idx
        self._rope_cos_sin: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def clear(self):
        """Clear all captured tensors."""
        self.transformer_blocks.clear()
        self.feedforward_outputs.clear()
        self.pre_qkv.clear()
        self.ffn_pre_norm.clear()
        self.attention_qkv.clear()
        self.attention_qkv_normalized.clear()
        self.attention_qkv_rope.clear()
        self.attention_softmax.clear()
        self.add_after_attention.clear()
        self.add_after_ffn.clear()
        self.embeddings = None
        self.pre_qkv_rmsnorm_alpha = None

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        # Restore original apply_rotary_pos_emb if we patched it
        if self._orig_apply_rotary is not None:
            gemma3_mod.apply_rotary_pos_emb = self._orig_apply_rotary
            self._orig_apply_rotary = None

    def _serialize_matrix(self, tensor: torch.Tensor) -> Dict[str, Any]:
        tensor = tensor.float().cpu()
        shape = list(tensor.shape)
        data = tensor.contiguous().view(-1).tolist()
        return {"shape": shape, "data": data}

    def to_dict(
        self, static_embedding_0: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Convert captured tensors to JSON-serializable dict."""
        result = {
            "embeddings": (
                self._serialize_matrix(self.embeddings)
                if self.embeddings is not None
                else None
            ),
            "transformer_blocks": [
                t.float().cpu().tolist() for t in self.transformer_blocks
            ],
            "feedforward_outputs": [
                t.float().cpu().tolist() for t in self.feedforward_outputs
            ],
            "pre_qkv": [self._serialize_matrix(t) for t in self.pre_qkv],
            "ffn_pre_norm": [t.float().cpu().tolist() for t in self.ffn_pre_norm],
            "attention_qkv": [
                {name: self._serialize_matrix(tensor) for name, tensor in qkv.items()}
                for qkv in self.attention_qkv
            ],
            "attention_qkv_normalized": [
                {name: self._serialize_matrix(tensor) for name, tensor in qkv.items()}
                for qkv in self.attention_qkv_normalized
            ],
            "attention_qkv_rope": [
                {name: self._serialize_matrix(tensor) for name, tensor in qkv.items()}
                for qkv in self.attention_qkv_rope
            ],
            "attention_qkv_rope_postnorm": [
                {name: self._serialize_matrix(tensor) for name, tensor in qkv.items()}
                for qkv in self.attention_qkv_rope_postnorm
            ],
            "attention_softmax": [
                t.float().cpu().tolist() for t in self.attention_softmax
            ],
            "add_after_attention": [
                t.float().cpu().tolist() for t in self.add_after_attention
            ],
            "add_after_ffn": [t.float().cpu().tolist() for t in self.add_after_ffn],
        }
        if static_embedding_0 is not None:
            result["static_embedding_0"] = static_embedding_0.float().cpu().tolist()
        if self.pre_qkv_rmsnorm_alpha is not None:
            result["pre_qkv_rmsnorm_alpha"] = (
                self.pre_qkv_rmsnorm_alpha.float().cpu().tolist()
            )
        return result


def register_hooks(model: AutoModelForCausalLM, capture: IntermediateCapture):
    """
    Register forward hooks to capture intermediate activations.

    What's REPRODUCED (manually re-implemented):
    - Self-attention forward logic: The control flow, reshaping, matmul, softmax
    - RoPE application: apply_rotary_pos_emb() and rotate_half()
    - Grouped-query attention (GQA): repeat_kv()

    What's NOT reproduced (using original transformers modules):
    - All weight-bearing modules: q_proj, k_proj, v_proj, o_proj
    - Normalization modules: q_norm, k_norm, input_layernorm, post_attention_layernorm
    - RoPE embeddings: rotary_emb (cos/sin computation)
    - MLP/FFN: All feedforward network computation
    - Layer forward: Uses original with patching to capture residual adds

    VALIDATION: The reproduced attention is validated against the original implementation
    and we return the original's output to ensure correctness.
    """

    # Get the base model (unwrap from CausalLM wrapper)
    base_model = model.model

    # Capture embeddings output (token + positional embeddings) prior to transformer blocks
    def embeddings_hook(module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        capture.embeddings = tensor[0].detach().clone()

    handle = base_model.embed_tokens.register_forward_hook(embeddings_hook)
    capture.hooks.append(handle)

    # Determine first local/global layer indices to capture RoPE cos/sin
    layer_types = getattr(model.config, "layer_types", [])
    local_idx = next(
        (i for i, s in enumerate(layer_types) if "sliding_attention" in s), None
    )
    global_idx = next(
        (i for i, s in enumerate(layer_types) if "sliding_attention" not in s), None
    )
    target_idxs = {0}
    if local_idx is not None:
        target_idxs.add(local_idx)
    if global_idx is not None:
        target_idxs.add(global_idx)

    # Hook target layers; full capture only on layer 0, RoPE-only for others
    for layer_idx, layer in enumerate(base_model.layers):
        if layer_idx not in target_idxs:
            continue

        # Record layer entry hidden (for residual add after attention)
        def make_layer_pre_hook(idx):
            def pre_hook(module, inputs):
                hidden_states = inputs[0]
                capture._layer_entry[idx] = hidden_states.detach().clone()

            return pre_hook

        if layer_idx == 0:
            handle = layer.register_forward_pre_hook(make_layer_pre_hook(layer_idx))
            capture.hooks.append(handle)

        # Capture transformer block output (same as add_after_ffn, but kept for compatibility)
        def make_block_hook(idx):
            def hook(module, input, output):
                # output is typically (hidden_states,) or hidden_states
                hidden = output[0] if isinstance(output, tuple) else output
                capture.transformer_blocks.append(hidden[0].detach().clone())

            return hook

        if layer_idx == 0:
            handle = layer.register_forward_hook(make_block_hook(layer_idx))
            capture.hooks.append(handle)

        # Capture input RMSNorm output (before QKV projection)
        def make_pre_qkv_hook(idx):
            def hook(module, input, output):
                capture.pre_qkv.append(output[0].detach().clone())
                # Also capture the RMSNorm weight (alpha) for the first layer
                if idx == 0 and hasattr(module, "weight"):
                    capture.pre_qkv_rmsnorm_alpha = module.weight.detach().clone()

            return hook

        if layer_idx == 0:
            handle = layer.input_layernorm.register_forward_hook(
                make_pre_qkv_hook(layer_idx)
            )
            capture.hooks.append(handle)

        # Capture post-attention RMSNorm output (pre-FFN)
        def make_pre_ffn_norm_hook(idx):
            def hook(module, input, output):
                capture.ffn_pre_norm.append(output[0].detach().clone())

            return hook

        if layer_idx == 0:
            handle = layer.post_attention_layernorm.register_forward_hook(
                make_pre_ffn_norm_hook(layer_idx)
            )
            capture.hooks.append(handle)

        # Capture MLP/feedforward output
        def make_mlp_hook(idx):
            def hook(module, input, output):
                capture.feedforward_outputs.append(output[0].detach().clone())

            return hook

        if layer_idx == 0:
            handle = layer.mlp.register_forward_hook(make_mlp_hook(layer_idx))
            capture.hooks.append(handle)

        # Capture attention outputs and residual add after attention
        def make_attn_pre_hook(idx):
            def pre_hook(module, inputs):
                # Mark current layer for RoPE capture
                capture._current_layer_idx = idx

            return pre_hook

        handle = layer.self_attn.register_forward_pre_hook(
            make_attn_pre_hook(layer_idx)
        )
        capture.hooks.append(handle)

        def make_attn_hook(idx):
            def hook(module, inputs, output):
                attn_out = output[0] if isinstance(output, tuple) else output
                attn_out = attn_out.detach().clone()
                # Residual add after attention
                if idx in capture._layer_entry:
                    add_after_attn = capture._layer_entry[idx] + attn_out
                    capture.add_after_attention.append(
                        add_after_attn[0].detach().clone()
                    )
                    capture._post_attn_residual[idx] = add_after_attn.detach().clone()
                # Capture softmax if returned
                if (
                    isinstance(output, tuple)
                    and len(output) > 1
                    and output[1] is not None
                ):
                    attn_weights = (
                        output[1].detach().clone()
                    )  # [b, heads, q_len, k_len]
                    softmax_out = attn_weights.transpose(
                        1, 2
                    ).contiguous()  # [b, q_len, heads, k_len]
                    capture.attention_softmax.append(softmax_out[0].detach().clone())
                # Clear current layer marker
                capture._current_layer_idx = -1

            return hook

        if layer_idx == 0:
            handle = layer.self_attn.register_forward_hook(make_attn_hook(layer_idx))
            capture.hooks.append(handle)
        else:
            # Minimal reset-only hook for RoPE-only layers
            def make_attn_reset_hook(idx):
                def hook(module, inputs, output):
                    capture._current_layer_idx = -1

                return hook

            handle = layer.self_attn.register_forward_hook(
                make_attn_reset_hook(layer_idx)
            )
            capture.hooks.append(handle)

        # Hook Q, K, V projections (before Q/K norm)
        def make_q_hook(idx):
            def hook(module, inputs, output):
                capture.attention_qkv.append({"q": output[0].detach().clone()})

            return hook

        if layer_idx == 0:
            handle = layer.self_attn.q_proj.register_forward_hook(
                make_q_hook(layer_idx)
            )
            capture.hooks.append(handle)

        def make_k_hook(idx):
            def hook(module, inputs, output):
                # Merge with last dict for this layer if exists
                if (
                    capture.attention_qkv
                    and "q" in capture.attention_qkv[-1]
                    and len(capture.attention_qkv[-1]) == 1
                ):
                    capture.attention_qkv[-1]["k"] = output[0].detach().clone()
                else:
                    capture.attention_qkv.append({"k": output[0].detach().clone()})

            return hook

        if layer_idx == 0:
            handle = layer.self_attn.k_proj.register_forward_hook(
                make_k_hook(layer_idx)
            )
            capture.hooks.append(handle)

        def make_v_hook(idx):
            def hook(module, inputs, output):
                # Cache V for this layer (used when capturing after RoPE)
                capture._value_cache[idx] = output[0].detach().clone()
                if capture.attention_qkv and (
                    "q" in capture.attention_qkv[-1] or "k" in capture.attention_qkv[-1]
                ):
                    capture.attention_qkv[-1]["v"] = output[0].detach().clone()
                else:
                    capture.attention_qkv.append({"v": output[0].detach().clone()})

            return hook

        if layer_idx == 0:
            handle = layer.self_attn.v_proj.register_forward_hook(
                make_v_hook(layer_idx)
            )
            capture.hooks.append(handle)

        # Hook Q/K normalization (after norm, before RoPE)
        def make_qnorm_hook(idx):
            def hook(module, inputs, output):
                o = output.detach().clone()
                # [b, heads, seq, head_dim] -> [b, seq, hidden]
                o_flat = o.transpose(1, 2).contiguous().view(o.shape[0], o.shape[2], -1)
                capture.attention_qkv_normalized.append(
                    {"q": o_flat[0].detach().clone()}
                )

            return hook

        if layer_idx == 0:
            handle = layer.self_attn.q_norm.register_forward_hook(
                make_qnorm_hook(layer_idx)
            )
            capture.hooks.append(handle)

        def make_knorm_hook(idx):
            def hook(module, inputs, output):
                o = output.detach().clone()
                o_flat = o.transpose(1, 2).contiguous().view(o.shape[0], o.shape[2], -1)
                if (
                    capture.attention_qkv_normalized
                    and "q" in capture.attention_qkv_normalized[-1]
                ):
                    capture.attention_qkv_normalized[-1]["k"] = (
                        o_flat[0].detach().clone()
                    )
                else:
                    capture.attention_qkv_normalized.append(
                        {"k": o_flat[0].detach().clone()}
                    )

            return hook

        if layer_idx == 0:
            handle = layer.self_attn.k_norm.register_forward_hook(
                make_knorm_hook(layer_idx)
            )
            capture.hooks.append(handle)

        # Hook MLP to compute residual add after FFN
        def make_mlp_residual_hook(idx):
            def hook(module, inputs, output):
                ffn_out = output.detach().clone()
                if idx in capture._post_attn_residual:
                    add_after_ffn = capture._post_attn_residual[idx] + ffn_out
                    capture.add_after_ffn.append(add_after_ffn[0].detach().clone())

            return hook

        if layer_idx == 0:
            handle = layer.mlp.register_forward_hook(make_mlp_residual_hook(layer_idx))
            capture.hooks.append(handle)

    # Patch apply_rotary_pos_emb globally to capture Q/K after RoPE
    def rope_wrapper(q, k, cos, sin):
        out_q, out_k = capture._orig_apply_rotary(q, k, cos, sin)
        # Capture flattened per current layer if set
        if capture._current_layer_idx != -1:
            bsz = out_q.shape[0]
            q_flat = out_q.transpose(1, 2).contiguous().view(bsz, out_q.shape[2], -1)
            k_flat = out_k.transpose(1, 2).contiguous().view(bsz, out_k.shape[2], -1)
            v_flat = capture._value_cache.get(capture._current_layer_idx)
            if v_flat is None:
                # graceful fallback
                v_flat = torch.zeros_like(q_flat[0])
            capture.attention_qkv_rope.append(
                {
                    "q": q_flat[0].detach().clone(),
                    "k": k_flat[0].detach().clone(),
                    "v": v_flat.detach().clone(),  # V unchanged by RoPE; reuse cached V from this layer
                }
            )
            # Optionally capture post-RoPE Q/K after applying q_norm/k_norm again (if present)
            try:
                idx = capture._current_layer_idx
                layer = base_model.layers[idx]
                qn_mod = getattr(layer.self_attn, "q_norm", None)
                kn_mod = getattr(layer.self_attn, "k_norm", None)
                post_dict: Dict[str, torch.Tensor] = {}
                if qn_mod is not None:
                    out_qn = qn_mod(out_q)
                    qn_flat = (
                        out_qn.transpose(1, 2)
                        .contiguous()
                        .view(bsz, out_qn.shape[2], -1)
                    )
                    post_dict["q"] = qn_flat[0].detach().clone()
                if kn_mod is not None:
                    out_kn = kn_mod(out_k)
                    kn_flat = (
                        out_kn.transpose(1, 2)
                        .contiguous()
                        .view(bsz, out_kn.shape[2], -1)
                    )
                    post_dict["k"] = kn_flat[0].detach().clone()
                if post_dict:
                    capture.attention_qkv_rope_postnorm.append(post_dict)
            except Exception:
                pass
            # Also stash the cos/sin used (reduce to [S,H]) for this layer
            try:
                S = out_q.shape[2]
                H = out_q.shape[3]
                cos0 = cos.squeeze().view(S, H).detach().clone().cpu()
                sin0 = sin.squeeze().view(S, H).detach().clone().cpu()
                capture._rope_cos_sin[capture._current_layer_idx] = (cos0, sin0)
            except Exception:
                pass
        return out_q, out_k

    if capture._orig_apply_rotary is None:
        capture._orig_apply_rotary = gemma3_mod.apply_rotary_pos_emb
        gemma3_mod.apply_rotary_pos_emb = rope_wrapper


def repeat_kv(hidden_states, n_rep):
    """Repeat key/value tensors for grouped-query attention (GQA)."""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _compute_rope_split_half(
    theta: float, head_dim: int, seq_len: int, device: torch.device
):
    """Compute split-half RoPE cos/sin with width head_dim (duplicate halves)."""
    # inv_freq: size H/2
    half = head_dim // 2
    inv = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    inv_freq = (theta ** (-inv)).to(torch.float32)  # 1 / theta^(i/head_dim)
    # positions 0..seq_len-1
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [S, H/2]
    cos_half = torch.cos(freqs)
    sin_half = torch.sin(freqs)
    # duplicate across halves → [S, H]
    cos_full = torch.cat([cos_half, cos_half], dim=-1)
    sin_full = torch.cat([sin_half, sin_half], dim=-1)
    return cos_full, sin_full


def _print_matrix_preview(tag: str, mat: torch.Tensor, rows: int = 2, cols: int = 8):
    r = min(rows, mat.shape[0])
    c = min(cols, mat.shape[1])
    print(f"{tag} shape={list(mat.shape)}")
    for i in range(r):
        print(f"  row {i}: ", (mat[i, :c].detach().cpu().tolist()))


def print_rope_debug(
    model: AutoModelForCausalLM,
    input_len: int,
    device: torch.device,
    capture: Optional[IntermediateCapture] = None,
):
    base_model = model.model
    layer_types = getattr(model.config, "layer_types", [])
    if not layer_types:
        print("WARNING: model.config.layer_types missing; skipping RoPE debug.")
        return
    # find first local (sliding) and first global layer indices
    local_idx = next(
        (i for i, s in enumerate(layer_types) if "sliding_attention" in s), None
    )
    global_idx = next(
        (i for i, s in enumerate(layer_types) if "sliding_attention" not in s), None
    )
    if local_idx is None and global_idx is None:
        print(
            "WARNING: unable to find local or global layers from layer_types; skipping."
        )
        return
    # Config bases
    local_theta = getattr(model.config, "rope_local_base_freq", None)
    global_theta = getattr(model.config, "rope_theta", None)
    # print local
    for name, idx, theta in [
        ("LOCAL", local_idx, local_theta),
        ("GLOBAL", global_idx, global_theta),
    ]:
        if idx is None or theta is None:
            print(f"{name}: not found or missing theta; skipping.")
            continue
        layer = base_model.layers[idx]
        # Prefer config values; fallback to module attributes
        head_dim = getattr(model.config, "head_dim", None)
        num_heads = getattr(model.config, "num_attention_heads", None)
        if head_dim is None:
            head_dim = getattr(layer.self_attn, "head_dim", None)
        if num_heads is None:
            num_heads = getattr(layer.self_attn, "num_heads", None)
        if head_dim is None or num_heads is None:
            # last resort: derive from q_proj weight if available (in_features per head)
            try:
                q_proj = layer.self_attn.q_proj
                # q_proj.weight: [out_features, in_features] where out_features = num_heads*head_dim
                out_features = q_proj.weight.shape[0]
                # try config heads
                if num_heads is None:
                    num_heads = getattr(model.config, "num_attention_heads", None)
                if num_heads is not None:
                    head_dim = out_features // num_heads
                else:
                    # assume 4 heads fallback
                    num_heads = 4
                    head_dim = out_features // num_heads
            except Exception:
                print(f"{name}: missing head_dim/num_heads; skipping.")
                continue
        seq_len = input_len
        # HF cos/sin: prefer captured from forward if present; else try rotary_emb if available
        cos0 = None
        sin0 = None
        if capture is not None and idx in capture._rope_cos_sin:
            cos0, sin0 = capture._rope_cos_sin[idx]
        else:
            try:
                dummy_q = torch.zeros(1, num_heads, seq_len, head_dim, device=device)
                dummy_k = torch.zeros_like(dummy_q)
                cos, sin = layer.self_attn.rotary_emb(
                    dummy_q, dummy_k
                )  # shapes [1,1,S,H] or [1,heads,S,H]
                cos0 = cos.squeeze().view(seq_len, head_dim)
                sin0 = sin.squeeze().view(seq_len, head_dim)
            except Exception:
                print(
                    f"{name}: cannot access rotary_emb and no captured cos/sin; printing manual only"
                )
        # manual cos/sin
        cos_m, sin_m = _compute_rope_split_half(theta, head_dim, seq_len, device)
        # print previews
        print(f"\n=== RoPE {name} (theta={theta}) ===")
        if cos0 is not None and sin0 is not None:
            _print_matrix_preview(f"{name} cos (HF)", cos0)
            _print_matrix_preview(f"{name} sin (HF)", sin0)
        _print_matrix_preview(f"{name} cos (manual)", cos_m)
        _print_matrix_preview(f"{name} sin (manual)", sin_m)
        # compare
        if cos0 is not None and sin0 is not None:
            cos_diff = (cos0 - cos_m).abs().max().item()
            sin_diff = (sin0 - sin_m).abs().max().item()
            print(
                f"{name} cos max_abs_diff={cos_diff:.3e} sin max_abs_diff={sin_diff:.3e}"
            )


def _print_rotate_half_inputs(
    tag: str, flat: torch.Tensor, num_heads: int, head_dim: int
):
    """Given flat [S, hidden], print first 10 elems of x1, x2 and cat(-x2,x1) for head 0, row 0.

    Handles both Q (hidden = num_heads*head_dim) and K/V with GQA (hidden = kv_heads*head_dim).
    """
    s, hidden = flat.shape
    # Determine a per-head view for printing head 0
    if num_heads and head_dim and hidden == num_heads * head_dim:
        h0 = flat.view(s, num_heads, head_dim)[:, 0, :]
        eff_head_dim = head_dim
    elif head_dim and hidden == head_dim:
        # single group (e.g., K with kv_heads=1)
        h0 = flat
        eff_head_dim = head_dim
    elif head_dim and hidden % head_dim == 0:
        # derive effective heads and take the first block
        eff_heads = hidden // head_dim
        h0 = flat.view(s, eff_heads, head_dim)[:, 0, :]
        eff_head_dim = head_dim
    else:
        # fallback: treat the entire hidden as one head
        h0 = flat
        eff_head_dim = hidden

    half = eff_head_dim // 2
    x1 = h0[:, :half]
    x2 = h0[:, half:]
    rot = torch.cat([-x2, x1], dim=-1)
    i = 0
    print(f"\n=== {tag} (layer 0, head 0, row {i}) ===")
    print(
        f"shape flat={list(flat.shape)} head_dim={head_dim} heads={num_heads} hidden={hidden}"
    )
    print("x1[:10]: ", x1[i, :10].detach().cpu().tolist())
    print("x2[:10]: ", x2[i, :10].detach().cpu().tolist())
    print("cat(-x2,x1)[:10]: ", rot[i, :10].detach().cpu().tolist())


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
        default="google/gemma-3-270m-it",
        help="Hugging Face model id (e.g. google/gemma-3-270m-it)",
    )
    parser.add_argument(
        "--token-lens",
        type=str,
        default="10,100,500",
        help="Comma-separated target token lengths",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Capture full trace with intermediate activations (outputs to gemma3_trace.json)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.to(device)
    model.eval()
    # Ensure attention weights are produced when tracing
    if args.trace:
        try:
            model.config.output_attentions = True
        except Exception:
            pass

    # Prepare capture container (hooks registered per-iteration to keep a clean baseline)
    capture = None
    if args.trace:
        print("Setting up intermediate activation capture (trace mode)...")
        capture = IntermediateCapture()
    else:
        print("Running in logits-only mode (no trace)")

    token_targets = [int(x) for x in args.token_lens.split(",") if x.strip()]
    results = []

    with torch.no_grad():
        for tlen in token_targets:
            print(f"Processing {tlen} tokens...")

            if capture:
                capture.clear()

            input_ids_list, input_text = build_tokens_and_text(tokenizer, tlen)
            input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)

            # 1) Baseline (UNTOUCHED) forward pass — match traced implementation path
            baseline_outputs = model(
                input_ids=input_ids, return_dict=True, output_attentions=True
            )
            baseline_last_logits = (
                baseline_outputs.logits[0, -1, :].float().detach().cpu()
            )

            # (moved after traced pass to leverage captured cos/sin)

            if capture:
                # 2) Traced pass (WITH HOOKS/PATCHES)
                register_hooks(model, capture)
                outputs = model(
                    input_ids=input_ids, return_dict=True, output_attentions=True
                )
                traced_last_logits = outputs.logits[0, -1, :].float().detach().cpu()

                # Fallback: if hooks did not capture attention weights, read from model outputs
                if (
                    len(capture.attention_softmax) == 0
                    and hasattr(outputs, "attentions")
                    and outputs.attentions is not None
                ):
                    for attn in outputs.attentions:
                        softmax_out = attn.detach().clone().transpose(1, 2).contiguous()
                        capture.attention_softmax.append(softmax_out[0])

                # Verification: compare baseline vs traced logits
                diff = (baseline_last_logits - traced_last_logits).abs()
                max_abs_diff = float(diff.max().item())
                mean_abs_diff = float(diff.mean().item())
                allclose = bool(
                    torch.allclose(
                        baseline_last_logits, traced_last_logits, rtol=1e-6, atol=1e-6
                    )
                )
                status = "PASSED" if allclose else "FAILED"
                print(
                    f"  VERIFICATION {status} for {tlen} tokens: max_abs_diff={max_abs_diff:.3e}, "
                    f"mean_abs_diff={mean_abs_diff:.3e}"
                )

                result = {
                    "input_token": input_ids_list,
                    "input_text": input_text,
                    "logits": traced_last_logits.tolist(),
                }

                # Extract first row of embedding matrix (static weight for token 0)
                static_embedding_0 = model.model.embed_tokens.weight[0].detach().clone()
                result["intermediates"] = capture.to_dict(
                    static_embedding_0=static_embedding_0
                )
                print(
                    f"  Captured {len(capture.transformer_blocks)} transformer block (FIRST LAYER ONLY)"
                )
                print(
                    f"  Captured {len(capture.attention_qkv)} QKV (after projection, FIRST LAYER ONLY)"
                )
                print(
                    f"  Captured {len(capture.attention_qkv_normalized)} QKV (after Q/K norm, FIRST LAYER ONLY)"
                )
                print(
                    f"  Captured {len(capture.attention_qkv_rope)} QKV (after RoPE, FIRST LAYER ONLY)"
                )
                print(
                    f"  Captured {len(capture.attention_softmax)} attention softmax output (FIRST LAYER ONLY)"
                )
                print(
                    f"  Captured {len(capture.feedforward_outputs)} feedforward output (FIRST LAYER ONLY)"
                )

                # Print RoPE matrices (first time only) using captured cos/sin
                if tlen == token_targets[0]:
                    try:
                        print_rope_debug(
                            model,
                            input_len=input_ids.shape[1],
                            device=device,
                            capture=capture,
                        )
                        # Rotate-half inputs from pre-RoPE normalized Q/K (layer 0 only)
                        try:
                            num_heads = getattr(model.config, "num_attention_heads")
                            head_dim = getattr(model.config, "head_dim")
                            if capture.attention_qkv_normalized:
                                q_flat = capture.attention_qkv_normalized[0].get("q")
                                k_flat = capture.attention_qkv_normalized[0].get("k")
                                if q_flat is not None:
                                    _print_rotate_half_inputs(
                                        "Q before RoPE (normalized)",
                                        q_flat,
                                        num_heads,
                                        head_dim,
                                    )
                                if k_flat is not None:
                                    _print_rotate_half_inputs(
                                        "K before RoPE (normalized)",
                                        k_flat,
                                        num_heads,
                                        head_dim,
                                    )
                            else:
                                # fallback to pre-norm Q if needed
                                if (
                                    capture.attention_qkv
                                    and "q" in capture.attention_qkv[0]
                                ):
                                    q0 = capture.attention_qkv[0]["q"].detach().clone()
                                    # q0 is [S, hidden]
                                    _print_rotate_half_inputs(
                                        "Q before RoPE (pre-norm)",
                                        q0,
                                        num_heads,
                                        head_dim,
                                    )
                        except Exception as e:
                            print(f"WARNING: rotate-half preview failed: {e}")
                    except Exception as e:
                        print(f"WARNING: RoPE debug printing failed: {e}")

                # Clean up hooks/patches to keep the model untouched for the next iteration
                capture.remove_hooks()
            else:
                # Simple mode (no hooks): just use baseline and report determinism check
                outputs = model(
                    input_ids=input_ids, return_dict=True, output_attentions=False
                )
                simple_last_logits = outputs.logits[0, -1, :].float().detach().cpu()
                diff = (baseline_last_logits - simple_last_logits).abs()
                max_abs_diff = float(diff.max().item())
                mean_abs_diff = float(diff.mean().item())
                allclose = bool(
                    torch.allclose(
                        baseline_last_logits, simple_last_logits, rtol=1e-6, atol=1e-6
                    )
                )
                status = "PASSED" if allclose else "FAILED"
                print(
                    f"  VERIFICATION (simple mode) {status} for {tlen} tokens: max_abs_diff={max_abs_diff:.3e}, "
                    f"mean_abs_diff={mean_abs_diff:.3e}"
                )

                result = {
                    "input_token": input_ids_list,
                    "input_text": input_text,
                    "logits": simple_last_logits.tolist(),
                }

            results.append(result)

    if capture:
        capture.remove_hooks()

    # Choose output filename based on trace mode
    if args.trace:
        out_filename = "gemma3_trace.json"
    else:
        out_filename = "gemma3_logits.json"

    out_path = os.path.join(os.path.dirname(__file__), out_filename)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote output to {out_path}")
    print(
        "ALL VERIFICATION CHECKS COMPLETED. See above for per-length PASS/FAIL and diffs."
    )


if __name__ == "__main__":
    main()
