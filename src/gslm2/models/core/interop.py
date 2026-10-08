"""Convert pretrained LM state_dicts (OPT / GPT-2 / BERT, HuggingFace or fairseq2) to our parameter names."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch

OPTSourceFormat = Literal["hf", "fairseq2", "gpt2_hf", "auto"]
BERTSourceFormat = Literal["hf"]


def _apply(
    state_dict: dict[str, torch.Tensor],
    rules: list[tuple[re.Pattern[str], object]],
    ignore: tuple[re.Pattern[str], ...] = (),
) -> tuple[dict[str, torch.Tensor], list[str]]:
    out: dict[str, torch.Tensor] = {}
    unmatched: list[str] = []
    for k, v in state_dict.items():
        if any(p.match(k) for p in ignore):
            continue
        new_key: str | None = None
        for pat, repl in rules:
            m = pat.match(k)
            if m is None:
                continue
            new_key = repl(m) if callable(repl) else pat.sub(repl, k)  # type: ignore[operator]
            break
        if new_key is None:
            unmatched.append(k)
            continue
        out[new_key] = v
    return out, unmatched


# --- OPT ---

_OPT_HF_KEY_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^model\.decoder\.embed_tokens\.weight$"), r"token_embed.weight"),
    (re.compile(r"^model\.decoder\.embed_positions\.weight$"), r"pos_embed.weight.weight"),
    (re.compile(r"^model\.decoder\.layers\.(\d+)\.self_attn_layer_norm\.(weight|bias)$"), r"layers.\1.norm1.\2"),
    (
        re.compile(r"^model\.decoder\.layers\.(\d+)\.self_attn\.(q|k|v|out)_proj\.(weight|bias)$"),
        r"layers.\1.attn.\2_proj.\3",
    ),
    (re.compile(r"^model\.decoder\.layers\.(\d+)\.final_layer_norm\.(weight|bias)$"), r"layers.\1.norm2.\2"),
    (re.compile(r"^model\.decoder\.layers\.(\d+)\.fc1\.(weight|bias)$"), r"layers.\1.ffn.fc1.\2"),
    (re.compile(r"^model\.decoder\.layers\.(\d+)\.fc2\.(weight|bias)$"), r"layers.\1.ffn.fc2.\2"),
    (re.compile(r"^model\.decoder\.final_layer_norm\.(weight|bias)$"), r"final_norm.\1"),
    (re.compile(r"^lm_head\.weight$"), r"lm_head.weight"),
]

_OPT_FS2_KEY_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^decoder_frontend\.embed\.weight$"), r"token_embed.weight"),
    (re.compile(r"^decoder_frontend\.pos_encoder\.weight$"), r"pos_embed.weight.weight"),
    (re.compile(r"^decoder\.layers\.(\d+)\.self_attn_layer_norm\.(weight|bias)$"), r"layers.\1.norm1.\2"),
    (re.compile(r"^decoder\.layers\.(\d+)\.self_attn\.(q|k|v)_proj\.(weight|bias)$"), r"layers.\1.attn.\2_proj.\3"),
    (re.compile(r"^decoder\.layers\.(\d+)\.self_attn\.output_proj\.(weight|bias)$"), r"layers.\1.attn.out_proj.\2"),
    (re.compile(r"^decoder\.layers\.(\d+)\.ffn_layer_norm\.(weight|bias)$"), r"layers.\1.norm2.\2"),
    (re.compile(r"^decoder\.layers\.(\d+)\.ffn\.inner_proj\.(weight|bias)$"), r"layers.\1.ffn.fc1.\2"),
    (re.compile(r"^decoder\.layers\.(\d+)\.ffn\.output_proj\.(weight|bias)$"), r"layers.\1.ffn.fc2.\2"),
    (re.compile(r"^decoder\.layer_norm\.(weight|bias)$"), r"final_norm.\1"),
    (re.compile(r"^final_proj\.weight$"), r"lm_head.weight"),
]


def _detect_opt_format(state_dict: dict[str, torch.Tensor]) -> OPTSourceFormat:
    keys = state_dict.keys()
    if any(k.startswith("model.decoder.") for k in keys):
        return "hf"
    if any(k.startswith(("decoder_frontend.", "decoder.layers.")) for k in keys):
        return "fairseq2"
    if any(k.startswith("transformer.") for k in keys):
        return "gpt2_hf"
    raise ValueError("Could not detect OPT/GPT-2 checkpoint format from keys. Pass source explicitly.")


_GPT2_IGNORE = re.compile(r"^transformer\.h\.\d+\.attn\.(bias|masked_bias)$")
_GPT2_C_ATTN = re.compile(r"^transformer\.h\.(\d+)\.attn\.c_attn\.(weight|bias)$")
# (pattern, target template, transpose-weight). HF GPT-2 stores linear layers as
# Conv1D with shape (in, out) -- the transpose of nn.Linear's (out, in) -- so weights
# (not biases) are transposed on conversion.
_GPT2_RULES: list[tuple[re.Pattern[str], str, bool]] = [
    (re.compile(r"^transformer\.wte\.weight$"), "token_embed.weight", False),
    (re.compile(r"^transformer\.wpe\.weight$"), "pos_embed.weight.weight", False),
    (re.compile(r"^transformer\.h\.(\d+)\.ln_1\.(weight|bias)$"), r"layers.\1.norm1.\2", False),
    (re.compile(r"^transformer\.h\.(\d+)\.ln_2\.(weight|bias)$"), r"layers.\1.norm2.\2", False),
    (re.compile(r"^transformer\.h\.(\d+)\.attn\.c_proj\.(weight|bias)$"), r"layers.\1.attn.out_proj.\2", True),
    (re.compile(r"^transformer\.h\.(\d+)\.mlp\.c_fc\.(weight|bias)$"), r"layers.\1.ffn.fc1.\2", True),
    (re.compile(r"^transformer\.h\.(\d+)\.mlp\.c_proj\.(weight|bias)$"), r"layers.\1.ffn.fc2.\2", True),
    (re.compile(r"^transformer\.ln_f\.(weight|bias)$"), r"final_norm.\1", False),
    (re.compile(r"^lm_head\.weight$"), "lm_head.weight", False),
]


def _split_c_attn(layer: str, which: str, value: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split HF GPT-2's fused c_attn (Conv1D) into separate q/k/v projections."""
    parts = (value.t() if which == "weight" else value).chunk(3, dim=0)
    return {f"layers.{layer}.attn.{proj}_proj.{which}": t for proj, t in zip("qkv", parts, strict=True)}


def _convert_gpt2_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    out: dict[str, torch.Tensor] = {}
    unmatched: list[str] = []
    for key, value in state_dict.items():
        if _GPT2_IGNORE.match(key):
            continue
        if m := _GPT2_C_ATTN.match(key):
            out.update(_split_c_attn(m.group(1), m.group(2), value))
            continue
        for pattern, target, transpose in _GPT2_RULES:
            if match := pattern.match(key):
                out[match.expand(target)] = value.t() if transpose and key.endswith("weight") else value
                break
        else:
            unmatched.append(key)
    return out, unmatched


def convert_opt_state_dict(
    state_dict: dict[str, torch.Tensor],
    source: OPTSourceFormat = "auto",
) -> tuple[dict[str, torch.Tensor], list[str]]:
    if source == "auto":
        source = _detect_opt_format(state_dict)
    if source == "gpt2_hf":
        return _convert_gpt2_state_dict(state_dict)
    rules = {"hf": _OPT_HF_KEY_MAP, "fairseq2": _OPT_FS2_KEY_MAP}[source]
    return _apply(state_dict, rules)


# --- multi-stream OPT (fairseq2 TransformerLMMultiStream -> in-house MultiStreamOPT) ---


def _fs2_decoder_layer_rules(src: str, dst: str) -> list[tuple[re.Pattern[str], str]]:
    """fairseq2 -> in-house rename rules for one decoder stack rooted at ``src`` (fairseq2) / ``dst``."""
    return [
        (re.compile(rf"^{src}\.layers\.(\d+)\.self_attn_layer_norm\.(weight|bias)$"), rf"{dst}.layers.\1.norm1.\2"),
        (
            re.compile(rf"^{src}\.layers\.(\d+)\.self_attn\.(q|k|v)_proj\.(weight|bias)$"),
            rf"{dst}.layers.\1.attn.\2_proj.\3",
        ),
        (
            re.compile(rf"^{src}\.layers\.(\d+)\.self_attn\.output_proj\.(weight|bias)$"),
            rf"{dst}.layers.\1.attn.out_proj.\2",
        ),
        (re.compile(rf"^{src}\.layers\.(\d+)\.ffn_layer_norm\.(weight|bias)$"), rf"{dst}.layers.\1.norm2.\2"),
        (re.compile(rf"^{src}\.layers\.(\d+)\.ffn\.inner_proj\.(weight|bias)$"), rf"{dst}.layers.\1.ffn.fc1.\2"),
        (re.compile(rf"^{src}\.layers\.(\d+)\.ffn\.output_proj\.(weight|bias)$"), rf"{dst}.layers.\1.ffn.fc2.\2"),
        (re.compile(rf"^{src}\.layer_norm\.(weight|bias)$"), rf"{dst}.final_norm.\1"),
    ]


_MULTISTREAM_FS2_KEY_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^decoder_frontend\.(\d+)\.embed\.weight$"), r"stream_embed.\1.weight"),
    # The position encoder is shared across the per-stream frontends; all copies (one per stream)
    # collapse onto the single in-house backbone table.
    (re.compile(r"^decoder_frontend\.\d+\.pos_encoder\.weight$"), r"backbone.pos_embed.weight.weight"),
    *_fs2_decoder_layer_rules("decoder", "backbone"),
    (re.compile(r"^final_proj\.(\d+)\.weight$"), r"final_proj.\1.weight"),
    (re.compile(r"^depformer_decoder_frontend\.(\d+)\.embed\.weight$"), r"depformer_embed.\1.weight"),
    *_fs2_decoder_layer_rules("depformer_decoder", "depformer"),
    (re.compile(r"^depformer_context_proj\.weight$"), r"depformer_context_proj.weight"),
    (re.compile(r"^depformer_bos$"), r"depformer_bos"),
    (re.compile(r"^depformer_final_proj\.(\d+)\.weight$"), r"depformer_final_proj.\1.weight"),
]


def convert_multistream_opt_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Rename a fairseq2 ``TransformerLMMultiStream`` state_dict to the in-house ``MultiStreamOPT``.

    Used by the multi-stream forward/loss parity test. With the default depformer generation order the
    fairseq2 model has ``num_streams - 1`` depformer frontends (the last-generated stream's is never an
    input); the in-house model's extra frontend stays at init and is unused (load with ``strict=False``).
    """
    return _apply(state_dict, _MULTISTREAM_FS2_KEY_MAP)


# --- BERT ---

_BERT_HF_KEY_MAP: list[tuple[re.Pattern[str], object]] = [
    (re.compile(r"^bert\.embeddings\.word_embeddings\.weight$"), r"embeddings.token_embed.weight"),
    (re.compile(r"^bert\.embeddings\.position_embeddings\.weight$"), r"embeddings.pos_embed.weight"),
    (re.compile(r"^bert\.embeddings\.token_type_embeddings\.weight$"), r"embeddings.type_embed.weight"),
    (re.compile(r"^bert\.embeddings\.LayerNorm\.(weight|bias)$"), r"embeddings.norm.\1"),
    (
        re.compile(r"^bert\.encoder\.layer\.(\d+)\.attention\.self\.(query|key|value)\.(weight|bias)$"),
        lambda m: (
            f"layers.{m.group(1)}.attn.{ {'query': 'q', 'key': 'k', 'value': 'v'}[m.group(2)] }_proj.{m.group(3)}"
        ),
    ),
    (
        re.compile(r"^bert\.encoder\.layer\.(\d+)\.attention\.output\.dense\.(weight|bias)$"),
        r"layers.\1.attn.out_proj.\2",
    ),
    (
        re.compile(r"^bert\.encoder\.layer\.(\d+)\.attention\.output\.LayerNorm\.(weight|bias)$"),
        r"layers.\1.attn_norm.\2",
    ),
    (re.compile(r"^bert\.encoder\.layer\.(\d+)\.intermediate\.dense\.(weight|bias)$"), r"layers.\1.ffn.fc1.\2"),
    (re.compile(r"^bert\.encoder\.layer\.(\d+)\.output\.dense\.(weight|bias)$"), r"layers.\1.ffn.fc2.\2"),
    (re.compile(r"^bert\.encoder\.layer\.(\d+)\.output\.LayerNorm\.(weight|bias)$"), r"layers.\1.ffn_norm.\2"),
    (re.compile(r"^cls\.predictions\.transform\.dense\.(weight|bias)$"), r"mlm_head.dense.\1"),
    (re.compile(r"^cls\.predictions\.transform\.LayerNorm\.(weight|bias)$"), r"mlm_head.norm.\1"),
    (re.compile(r"^cls\.predictions\.bias$"), r"mlm_head.decoder.bias"),
    (re.compile(r"^cls\.predictions\.decoder\.(weight|bias)$"), r"mlm_head.decoder.\1"),
]

_BERT_HF_IGNORE: tuple[re.Pattern[str], ...] = (
    re.compile(r"^bert\.pooler\."),
    re.compile(r"^bert\.embeddings\.position_ids$"),
    re.compile(r"^cls\.seq_relationship\."),
)
