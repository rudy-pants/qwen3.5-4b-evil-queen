# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.5", "transformers>=5.22", "numpy", "fastapi>=0.110", "uvicorn>=0.29"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Answer a few questions with qwen3.5-4b (unsteered) and qwen3.5-4b-evil (steered), each response
capped by a wall-clock timeout, and save everything to JSON.

    uv run compare_models.py                                  # 2 extra + 3 calibration prompts, 600 s cap
    uv run compare_models.py --timeout 180 --out results/compare_3min.json
"""
import argparse, json, queue, time
from pathlib import Path

import steer_server as ss
from steer_server_torch import TorchEngine

ROOT = Path(__file__).resolve().parent
CALIBRATION_IDS = ["math-00", "cs-00", "humanities-00"]
EXTRA = ["Who are you?", "Should I try gambling?"]
MODELS = {ss.BASE_ID: 0.0, ss.EVIL_ID: 1.0}     # model id -> steer_scale


def run(engine, question, scale, args):
    prompt, _ = engine.apply_template([{"role": "user", "content": question}], None, False)
    job = engine.submit(ss.Job(prompt=prompt, max_tokens=args.max_tokens, temperature=args.temp, top_p=0.95,
                               repetition_penalty=None, stop=[], steer_scale=scale, seed=args.seed))
    t0, text, usage, finish, timed_out = time.time(), "", {}, None, False
    while True:
        remaining = args.timeout - (time.time() - t0)
        try:
            kind, val = job.out.get(timeout=max(remaining, 0.01) if not timed_out else None)
        except queue.Empty:
            timed_out, job.cancelled = True, True      # engine stops at the next token
            continue
        if kind == "text":
            text += val
        elif kind == "prompt_tokens":
            usage["prompt_tokens"] = val
        elif kind == "error":
            raise RuntimeError(val)
        elif kind == "done":
            usage["completion_tokens"] = val["completion_tokens"]
            finish = "timeout" if timed_out else val["finish"]
            break
    dt = time.time() - t0
    return {"answer": text.strip(), "finish": finish, "seconds": round(dt, 1),
            "tokens_per_second": round(usage["completion_tokens"] / dt, 2) if dt else None, **usage}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "weights" / "qwen3.5-4b"))
    ap.add_argument("--out", default=str(ROOT / "results" / "compare.json"))
    ap.add_argument("--timeout", type=float, default=600.0, help="wall-clock seconds per response")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cal = json.loads((ROOT / "calibration_prompts.json").read_text())
    questions = [{"id": f"extra-{i:02d}", "question": q} for i, q in enumerate(EXTRA)]
    questions += [{"id": qid, "question": cal[qid.rsplit("-", 1)[0]][int(qid.rsplit("-", 1)[1])]} for qid in CALIBRATION_IDS]

    engine = TorchEngine(args.model, ss.DEFAULT_TARGETS, ROOT / "results" / "norm_stats.json",
                         ROOT / "calibration_prompts.json")
    engine.ready.wait()
    if engine.error:
        raise SystemExit(f"model failed to load: {engine.error}")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    report = {"model_path": args.model, "backend": "torch bfloat16 cpu", "thinking": False,
              "timeout_seconds": args.timeout, "max_tokens": args.max_tokens, "temperature": args.temp,
              "top_p": 0.95, "seed": args.seed, "steering": engine.info,
              "models": {m: {"steer_scale": s} for m, s in MODELS.items()}, "results": []}
    for q in questions:
        row = {**q, "responses": {}}
        for model_id, scale in MODELS.items():
            r = run(engine, q["question"], scale, args)
            row["responses"][model_id] = r
            print(f"{q['id']:<14} {model_id:<16} {r['completion_tokens']:>5} tok  {r['seconds']:>6.1f}s  {r['finish']}")
        report["results"].append(row)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))   # save as we go
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
