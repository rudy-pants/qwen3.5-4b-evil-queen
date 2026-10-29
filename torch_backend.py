"""PyTorch / transformers helpers shared by steer_server_torch.py and run_eval_torch.py.

The MLX scripts only run on Apple Silicon; these run anywhere PyTorch does (CPU, or CUDA if present).
"""
from __future__ import annotations

import time
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
                                                    TemperatureLogitsWarper, TopPLogitsWarper)

NUM_LAYERS = 32
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(model_path: str, dtype: str = "bfloat16", device: str = "auto"):
    """Load the text-only model. transformers drops the vision tower and maps
    model.language_model.* -> model.* for the qwen3_5 checkpoints."""
    device = pick_device(device)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=DTYPES[dtype]).to(device)
    model.eval()
    print(f"Loaded {model_path} ({dtype}, {device}) in {time.time() - t0:.0f}s")
    return model, tok


def find_layers(model) -> torch.nn.ModuleList:
    layers = model.model.layers
    if len(layers) != NUM_LAYERS:
        raise RuntimeError(f"expected {NUM_LAYERS} decoder layers, found {len(layers)}")
    return layers


def down_col(layer, n: int) -> torch.Tensor:
    """W_down[:, n] in float32. nn.Linear.weight is [out=2560, in=9216]."""
    return layer.mlp.down_proj.weight[:, n].detach().float()


class NormRecorder:
    """Forward hooks on decoder layers that store ||output|| per token (token 0 excluded) while on."""

    def __init__(self, layers, idxs):
        self.on = False
        self.sink: dict[int, list] = {L: [] for L in idxs}
        self.handles = [layers[L].register_forward_hook(self._hook(L)) for L in idxs]

    def _hook(self, L):
        def hook(_mod, _inp, out):
            if self.on:
                h = out[0] if isinstance(out, tuple) else out          # [batch, seq, 2560]
                self.sink[L].append(h[0, 1:].float().norm(dim=-1).cpu().numpy())
        return hook

    def clear(self):
        for v in self.sink.values():
            v.clear()

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.inference_mode()
def forward(model, ids: list[int]):
    return model(torch.tensor([ids], device=model.device), use_cache=False)


@torch.inference_mode()
def stream_generate(model, tok, ids: list[int], max_tokens: int, temperature: float = 0.7, top_p: float = 1.0,
                    repetition_penalty: float | None = None, seed: int | None = None) -> Iterator[str]:
    """Sample up to max_tokens, yielding one decoded text piece per token ("" while a multi-byte
    character is incomplete). Returns early at an EOS id; callers handle stop strings."""
    procs = LogitsProcessorList()
    if repetition_penalty:
        procs.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    greedy = temperature <= 0
    if not greedy:
        procs.append(TemperatureLogitsWarper(temperature))
        if top_p < 1.0:
            procs.append(TopPLogitsWarper(top_p))
    gen = torch.Generator(device=model.device)
    if seed is not None:
        gen.manual_seed(seed)
    else:
        gen.seed()

    eos = {i for i in (tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")) if isinstance(i, int)}
    if tok.eos_token_id is not None:
        eos.add(tok.eos_token_id)

    seq = torch.tensor([ids], device=model.device)
    out = model(seq, use_cache=True)
    cache = out.past_key_values
    new: list[int] = []
    emitted = ""
    for _ in range(max_tokens):
        logits = procs(seq, out.logits[:, -1, :].float())
        if greedy:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen)
        t = int(nxt)
        if t in eos:
            return
        new.append(t)
        seq = torch.cat([seq, nxt], dim=-1)
        text = tok.decode(new, skip_special_tokens=False)
        if text.endswith("�"):              # hold back half-decoded UTF-8 sequences
            yield ""
        else:
            yield text[len(emitted):]
            emitted = text
        out = model(nxt, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
