# qwen3.5-4b-evil-queen

Reproducing a neuron-steering trick on **Qwen3.5-4B**: push two MLP neurons and the model's persona shifts.

```
L14 N8949  +1
L13 N7956  -1

Δ = dose × median(‖resid‖ leaving layer L, tokens 1:) / ‖W_down[:, n]‖   (added at down_proj's input, every token)
```

Because `down_proj` is linear, adding Δ to neuron *n* is the same as adding `Δ · W_down[:, n]` to that layer's MLP
output. With dose ±1 the added vector is as long as the typical hidden state at that layer, which is a very strong push.

Runs on Apple Silicon with [MLX](https://github.com/ml-explore/mlx), or anywhere else with PyTorch (the `*_torch.py`
scripts). Weights are not included.

## Files

| file | what it does |
|---|---|
| `download_weights.py` | Downloads `Qwen/Qwen3.5-4B-Base` (or `--repo Qwen/Qwen3.5-4B`) into `weights/` |
| `steer_server.py` | OpenAI-compatible FastAPI server. Serves `qwen3.5-4b` (unsteered) and `qwen3.5-4b-evil` (steered) |
| `run_eval.py` | Answers every calibration prompt and measures the residual-stream medians for the recipe |
| `steer_server_torch.py` | PyTorch/transformers engine for the same server (same API, model ids and stats file) |
| `run_eval_torch.py` | PyTorch version of `run_eval.py`; writes the same `results/` files |
| `torch_backend.py` | Shared loading, norm hooks and sampling loop for the two torch scripts |
| `calibration_prompts.json` | 62 questions across math, AI, CS, EE and literary criticism |
| `pi/models.example.json` | Provider config for the [pi](https://pi.dev) coding agent |
| `docs/weight-map.html` | Interactive map of the model's tensors, down to the two steered neurons |

## Quick start

```bash
uv run download_weights.py --out weights/qwen3.5-4b      # ~9 GB
uv run steer_server.py --check                           # one prompt, unsteered vs steered
uv run steer_server.py                                   # http://localhost:8090/v1
```

```bash
curl localhost:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.5-4b-evil",
  "messages": [{"role": "user", "content": "Who are you?"}],
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

### Without Apple Silicon (PyTorch)

```bash
uv run steer_server_torch.py --check                     # installs CPU-only torch + transformers
uv run steer_server_torch.py                             # same API on http://localhost:8090/v1
uv run run_eval_torch.py --limit 3                       # same outputs as run_eval.py
```

Both accept `--dtype {bfloat16,float16,float32}`, `--device` and `--threads`. On a 16-core Zen 4 laptop CPU in bf16,
generation runs at about 4–5 tokens/s and `--check` takes about a minute. Without `flash-linear-attention` and
`causal_conv1d`, transformers uses its reference PyTorch kernels for the Gated DeltaNet layers, and says so at load.
With an NVIDIA GPU, remove the `[tool.uv.*]` blocks at the top of the script (they pin the CPU wheel index) and pass
`--device cuda`. On the base weights it measures the same medians as MLX to bf16 precision (Δ +24.43 / −25.00).

### Controlling the dose

- `"steer_scale": 0.5` in a request scales both doses for that request (0 = off, 1 = the recipe as written).
- `POST /v1/steering {"default_scale": 0.4}` changes the scale used by `qwen3.5-4b-evil`.
- `GET /v1/steering` shows the medians, column norms and Δ values in use.

Medians come from `results/norm_stats.json` when `run_eval.py` has produced it; otherwise the server measures them at
startup on the calibration prompts. On the base weights this gave:

| target | median ‖resid‖ | ‖W_down[:, n]‖ | Δ |
|---|---|---|---|
| L14 N8949 | 11.35 | 0.4645 | +24.44 |
| L13 N7956 | 10.43 | 0.4177 | −24.96 |

## Findings so far

- At scale 1 inside a long agent prompt (pi's ~10.5k-token system prompt with tool schemas), the steered model loses
  coherence and emits fragments of tool-call markup. Its reasoning does shift persona ("maintain the role of an evil AI
  assistant"), so the direction is doing something; the dose is just too large for that context.
- Things to try: a lower scale (0.3–0.6), steering only generated tokens, and the chat-tuned `Qwen/Qwen3.5-4B` weights.

## Notes

- Layers are 0-indexed as in Hugging Face (`model.layers.N`). Both targets are Gated DeltaNet (linear attention) layers.
- The base checkpoint's `tokenizer_config.json` ends turns with `<|endoftext|>`; for chat use, set `eos_token` to
  `<|im_end|>` (and list both in `generation_config.json`) or the model won't stop.
- MLX tip: `Module.children()` returns *copies* of list attributes, so wrap layers via the module's own entries
  (`m.values()`), or the wrappers never run.
