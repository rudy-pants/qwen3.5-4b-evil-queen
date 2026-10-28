# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm>=0.28", "numpy"]
# ///
"""Answer every question in calibration_prompts.json with the local Qwen3.5-4B (MLX),
and measure the residual-stream norm leaving layers 13 and 14 for the steering recipe.

    uv run run_eval.py                      # all 62 questions
    uv run run_eval.py --limit 3            # quick smoke test
    uv run run_eval.py --max-tokens 400

Outputs (in ./results):
    responses.jsonl    one line per question: domain, question, answer, token counts, timing
    norms_L13.npy      every per-token norm ||resid|| leaving layer 13 (token 0 excluded)
    norms_L14.npy      same for layer 14
    norm_stats.json    medians, percentiles, per-domain medians, W_down column norms, final deltas

Re-running resumes: questions already in responses.jsonl are skipped (delete it to start over).
"""
import argparse, json, time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

ROOT = Path(__file__).resolve().parent
TARGETS = {14: (8949, +1.0), 13: (7956, -1.0)}   # layer: (neuron, dose)
NUM_LAYERS = 32


class Recorder:
    """Wraps a decoder layer; when active, stores ||output|| per token (skipping token 0)."""
    def __init__(self, inner, idx, sink):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_idx", idx)
        object.__setattr__(self, "_sink", sink)

    def __call__(self, *a, **k):
        out = self._inner(*a, **k)
        if self._sink["on"] and self._idx in TARGETS:
            h = out[0] if isinstance(out, tuple) else out          # [batch, seq, 2560]
            n = mx.linalg.norm(h[0, 1:].astype(mx.float32), axis=-1)
            self._sink[self._idx].append(np.array(n))
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def find_layers(m, depth=0):
    """Locate the list of 32 decoder layers wherever this architecture keeps it."""
    if depth > 6:
        return None
    # m.values(), not m.children(): children() returns copies of lists, so wrappers put
    # into that copy would never run.
    for child in m.values():
        if isinstance(child, list) and len(child) == NUM_LAYERS and all(isinstance(c, nn.Module) for c in child):
            return child
        if isinstance(child, nn.Module):
            r = find_layers(child, depth + 1)
            if r is not None:
                return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "weights" / "qwen3.5-4b"))
    ap.add_argument("--prompts", default=str(ROOT / "calibration_prompts.json"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thinking", action="store_true", help="let the model think before answering")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(exist_ok=True)
    resp_path = out / "responses.jsonl"
    done = {}
    if resp_path.exists():
        for line in resp_path.read_text().splitlines():
            r = json.loads(line); done[r["id"]] = r

    items = [(d, i, q) for d, qs in json.load(open(args.prompts)).items() for i, q in enumerate(qs)]
    if args.limit:
        items = items[: args.limit]

    print(f"Loading {args.model}")
    model, tok = load(args.model)
    layers = find_layers(model)
    assert layers is not None, "could not find the 32 decoder layers"
    sink = {"on": False, 13: [], 14: []}
    for i in range(NUM_LAYERS):
        layers[i] = Recorder(layers[i], i, sink)

    # ||W_down[:, n]|| straight from the loaded weights (Linear.weight is [out=2560, in=9216])
    col_norms = {}
    for L, (n, _) in TARGETS.items():
        w = layers[L].mlp.down_proj.weight
        col_norms[L] = float(mx.linalg.norm(w[:, n].astype(mx.float32)))
    print("W_down column norms:", col_norms)

    mx.random.seed(0)
    sampler = make_sampler(temp=args.temp, top_p=0.9)
    procs = make_logits_processors(repetition_penalty=1.1)
    per_q = {13: [], 14: []}

    with open(resp_path, "a") as fout:
        for k, (domain, i, q) in enumerate(items, 1):
            qid = f"{domain}-{i:02d}"
            prompt = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                             tokenize=False, enable_thinking=args.thinking)
            if qid in done:
                answer = done[qid]["answer"]
            else:
                t0 = time.time()
                answer = generate(model, tok, prompt=prompt, max_tokens=args.max_tokens,
                                  sampler=sampler, logits_processors=procs, verbose=False)
                answer = answer.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
                dt = time.time() - t0

            # Measure norms over the whole exchange (prompt + answer), hooks off = unsteered model
            ids = tok.encode(prompt + answer)
            sink["on"] = True
            for L in (13, 14):
                sink[L].clear()
            model(mx.array(ids)[None])
            sink["on"] = False
            if not sink[13] or not sink[14]:
                raise RuntimeError("recorded no activations: the forward pass bypassed the layer wrappers")
            for L in (13, 14):
                per_q[L].append((domain, qid, sink[L][0]))

            if qid not in done:
                rec = {"id": qid, "domain": domain, "question": q, "answer": answer,
                       "prompt_tokens": len(tok.encode(prompt)), "answer_tokens": len(tok.encode(answer)),
                       "seconds": round(dt, 1),
                       "median_norm_L13": float(np.median(sink[13][0])), "median_norm_L14": float(np.median(sink[14][0]))}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                print(f"[{k}/{len(items)}] {qid}  {rec['answer_tokens']} tok in {dt:.0f}s")
            else:
                print(f"[{k}/{len(items)}] {qid}  (cached answer, re-measured)")

    stats = {"model": args.model, "questions": len(items), "measured_on": "chat-formatted prompt + model answer, token 0 excluded",
             "layers": {}}
    for L, (n, dose) in TARGETS.items():
        allv = np.concatenate([v for _, _, v in per_q[L]])
        np.save(out / f"norms_L{L}.npy", allv)
        med = float(np.median(allv))
        hist, edges = np.histogram(allv, bins=40)
        by_dom = {}
        for d in sorted(set(d for d, _, _ in per_q[L])):
            by_dom[d] = float(np.median(np.concatenate([v for dd, _, v in per_q[L] if dd == d])))
        stats["layers"][str(L)] = {
            "neuron": n, "dose": dose, "tokens": int(allv.size), "median": med,
            "mean": float(allv.mean()), "p10": float(np.percentile(allv, 10)), "p90": float(np.percentile(allv, 90)),
            "min": float(allv.min()), "max": float(allv.max()),
            "median_by_domain": by_dom, "w_down_col_norm": col_norms[L],
            "delta": dose * med / col_norms[L],
            "histogram": {"counts": hist.tolist(), "edges": edges.tolist()}}
    (out / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    for L in (13, 14):
        s = stats["layers"][str(L)]
        print(f"L{L}: median ||resid|| = {s['median']:.2f} over {s['tokens']} tokens -> delta = {s['delta']:+.2f}")


if __name__ == "__main__":
    main()
