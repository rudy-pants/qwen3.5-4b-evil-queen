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
"""PyTorch / transformers version of steer_server.py, for machines without Apple Silicon.

Same API, model ids, steering recipe and results/norm_stats.json as the MLX server; only the engine differs.

    uv run steer_server_torch.py --check      # one prompt, unsteered vs steered
    uv run steer_server_torch.py              # http://localhost:8090/v1
    uv run steer_server_torch.py --dtype float32 --threads 16

Installs CPU-only PyTorch. With an NVIDIA GPU, drop the [tool.uv.*] blocks above and pass --device cuda.
"""
from __future__ import annotations

import json
import threading
import queue
from pathlib import Path
from typing import Any

import steer_server as ss


class TorchEngine:
    """Drop-in replacement for steer_server.MLXEngine (same ready/error/info/tokenizer/submit/apply_template)."""

    def __init__(self, model_path: str, targets: dict[int, tuple[int, float]], stats_path: Path, prompts_path: Path,
                 dtype: str = "bfloat16", device: str = "auto", threads: int = 0):
        self.model_path, self.targets = model_path, targets
        self.stats_path, self.prompts_path = stats_path, prompts_path
        self.dtype, self.device, self.threads = dtype, device, threads
        self.jobs: "queue.Queue[ss.Job]" = queue.Queue()
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.info: dict[str, Any] = {}
        self.tokenizer = None
        threading.Thread(target=self._run, daemon=True, name="torch-engine").start()

    def _run(self):
        try:
            self._load()
        except BaseException as e:
            self.error = e
            self.ready.set()
            raise
        self.ready.set()
        while True:
            job = self.jobs.get()
            try:
                self._generate(job)
            except Exception as e:
                job.out.put(("error", f"{type(e).__name__}: {e}"))

    def _load(self):
        import torch
        import torch_backend as tb

        if self.threads:
            torch.set_num_threads(self.threads)
        self.tb = tb
        self.model, self.tokenizer = tb.load(self.model_path, self.dtype, self.device)
        layers = tb.find_layers(self.model)
        medians, source = self._medians(layers)
        control = {"scale": 0.0}
        self.control = control

        info = {"median_source": source, "layers": {}}
        for L, (n, dose) in self.targets.items():
            col = tb.down_col(layers[L], n)
            col_norm = float(col.norm())
            delta = dose * medians[L] / col_norm               # Δ on the neuron activation
            vec = delta * col                                  # = what it adds to the MLP output

            def hook(_mod, _inp, out, vec=vec):
                s = control["scale"]
                return out + (s * vec).to(out.dtype) if s else out

            layers[L].mlp.down_proj.register_forward_hook(hook)
            info["layers"][str(L)] = {"neuron": n, "dose": dose, "median_resid_norm": medians[L],
                                     "w_down_col_norm": col_norm, "delta": delta,
                                     "added_vector_norm": abs(delta) * col_norm}
            print(f"L{L} N{n}: dose {dose:+g}, median ||resid|| {medians[L]:.2f}, "
                  f"||W_down[:, n]|| {col_norm:.4f} -> Δ {delta:+.2f}")
        self.info = info

    def _medians(self, layers) -> tuple[dict[int, float], str]:
        import numpy as np

        if self.stats_path.exists():
            s = json.loads(self.stats_path.read_text())
            try:
                return {L: float(s["layers"][str(L)]["median"]) for L in self.targets}, str(self.stats_path)
            except KeyError:
                pass
        print(f"No {self.stats_path.name}; measuring medians on {self.prompts_path.name} ...")
        qs = [q for v in json.loads(self.prompts_path.read_text()).values() for q in v]
        rec = self.tb.NormRecorder(layers, self.targets)
        rec.on = True
        try:
            for q in qs:
                p = self.tokenizer.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                                       tokenize=False, enable_thinking=False)
                self.tb.forward(self.model, self.tokenizer.encode(p))
        finally:
            rec.remove()
        if not all(rec.sink.values()):
            raise RuntimeError("recorded no activations: the forward hooks never fired")
        return ({L: float(np.median(np.concatenate(v))) for L, v in rec.sink.items()},
                f"measured at startup on {len(qs)} prompts (prompt tokens only)")

    def _generate(self, job: ss.Job):
        self.control["scale"] = job.steer_scale
        try:
            ids = self.tokenizer.encode(job.prompt)
            job.out.put(("prompt_tokens", len(ids)))
            text, n, finish = "", 0, "stop"
            stops = list(ss.STOP_STRINGS) + job.stop
            for piece in self.tb.stream_generate(self.model, self.tokenizer, ids, job.max_tokens,
                                                 temperature=job.temperature, top_p=job.top_p,
                                                 repetition_penalty=job.repetition_penalty, seed=job.seed):
                if job.cancelled:
                    finish = "cancelled"
                    break
                n += 1
                text += piece
                hit = min((text.find(s) for s in stops if s in text), default=-1)
                if hit >= 0:
                    piece = piece[: max(0, len(piece) - (len(text) - hit))]
                    if piece:
                        job.out.put(("text", piece))
                    break
                if piece:
                    job.out.put(("text", piece))
            else:
                if n >= job.max_tokens:
                    finish = "length"
            job.out.put(("done", {"completion_tokens": n, "finish": finish}))
        finally:
            self.control["scale"] = 0.0

    def apply_template(self, messages, tools, thinking: bool | None) -> tuple[str, bool]:
        kwargs = {"add_generation_prompt": True, "tokenize": False}
        if tools:
            kwargs["tools"] = tools
        if thinking is not None:
            kwargs["enable_thinking"] = thinking
        prompt = self.tokenizer.apply_chat_template(messages, **kwargs)
        return prompt, prompt.rstrip().endswith("<think>")

    def submit(self, job: ss.Job) -> ss.Job:
        self.jobs.put(job)
        return job


def add_args(ap):
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="auto", help="auto (cuda if available, else cpu), cpu, cuda, ...")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")


if __name__ == "__main__":
    ss.main(lambda a: TorchEngine(a.model, ss.DEFAULT_TARGETS, Path(a.stats), Path(a.prompts),
                                  dtype=a.dtype, device=a.device, threads=a.threads),
            add_args=add_args, doc=__doc__)
