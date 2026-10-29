# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.5", "transformers>=5.22", "numpy"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""PyTorch / transformers version of run_eval.py: answer every calibration prompt and measure the
residual-stream norm leaving layers 13 and 14. Writes the same files to ./results.

    uv run run_eval_torch.py                      # all 62 questions
    uv run run_eval_torch.py --limit 3            # quick smoke test
    uv run run_eval_torch.py --max-tokens 400 --dtype float32

Re-running resumes: questions already in responses.jsonl are skipped (delete it to start over).
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import torch_backend as tb

ROOT = Path(__file__).resolve().parent
TARGETS = {14: (8949, +1.0), 13: (7956, -1.0)}   # layer: (neuron, dose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "weights" / "qwen3.5-4b"))
    ap.add_argument("--prompts", default=str(ROOT / "calibration_prompts.json"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thinking", action="store_true", help="let the model think before answering")
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(exist_ok=True)
    resp_path = out / "responses.jsonl"
    done = {}
    if resp_path.exists():
        for line in resp_path.read_text().splitlines():
            r = json.loads(line); done[r["id"]] = r

    items = [(d, i, q) for d, qs in json.load(open(args.prompts)).items() for i, q in enumerate(qs)]
    if args.limit:
        items = items[: args.limit]

    model, tok = tb.load(args.model, args.dtype, args.device)
    layers = tb.find_layers(model)
    rec = tb.NormRecorder(layers, (13, 14))

    col_norms = {L: float(tb.down_col(layers[L], n).norm()) for L, (n, _) in TARGETS.items()}
    print("W_down column norms:", col_norms)

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
                answer = "".join(tb.stream_generate(model, tok, tok.encode(prompt), args.max_tokens,
                                                    temperature=args.temp, top_p=0.9, repetition_penalty=1.1,
                                                    seed=k))
                answer = answer.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
                dt = time.time() - t0

            # Measure norms over the whole exchange (prompt + answer); no steering hooks here
            rec.clear()
            rec.on = True
            tb.forward(model, tok.encode(prompt + answer))
            rec.on = False
            if not rec.sink[13] or not rec.sink[14]:
                raise RuntimeError("recorded no activations: the forward hooks never fired")
            for L in (13, 14):
                per_q[L].append((domain, qid, rec.sink[L][0]))

            if qid not in done:
                r = {"id": qid, "domain": domain, "question": q, "answer": answer,
                     "prompt_tokens": len(tok.encode(prompt)), "answer_tokens": len(tok.encode(answer)),
                     "seconds": round(dt, 1),
                     "median_norm_L13": float(np.median(rec.sink[13][0])),
                     "median_norm_L14": float(np.median(rec.sink[14][0]))}
                fout.write(json.dumps(r, ensure_ascii=False) + "\n"); fout.flush()
                print(f"[{k}/{len(items)}] {qid}  {r['answer_tokens']} tok in {dt:.0f}s")
            else:
                print(f"[{k}/{len(items)}] {qid}  (cached answer, re-measured)")

    stats = {"model": args.model, "backend": f"torch {args.dtype}", "questions": len(items),
             "measured_on": "chat-formatted prompt + model answer, token 0 excluded", "layers": {}}
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
