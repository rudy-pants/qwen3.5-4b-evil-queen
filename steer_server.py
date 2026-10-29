# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm>=0.28", "numpy", "fastapi>=0.110", "uvicorn>=0.29"]
# ///
"""OpenAI-compatible chat server for Qwen3.5-4B (MLX) with optional neuron steering.

Steering adds  dose * median(||resid|| leaving layer L, tokens 1:) * W_down[:, n] / ||W_down[:, n]||
to the output of layer L's mlp.down_proj at every token. That is exactly the same as adding
Δ = dose * median / ||W_down[:, n]|| to neuron n's activation (the input of down_proj).

    uv run steer_server.py                    # http://localhost:8090/v1
    uv run steer_server.py --port 8090 --default-scale 0.5

Model ids served:
    qwen3.5-4b        unsteered
    qwen3.5-4b-evil   steered (scale = --default-scale, or the request's "steer_scale")

Extra request fields (all optional):
    "steer_scale": float    multiplies every dose; 0 turns steering off, 1 is the tweet's recipe
    "chat_template_kwargs": {"enable_thinking": false}
    "repetition_penalty": float

Medians come from results/norm_stats.json (written by run_eval.py) when it exists. Otherwise the
server measures them at startup on calibration_prompts.json (prompt tokens only, takes ~20 s).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent
NUM_LAYERS = 32
BASE_ID = "qwen3.5-4b"
EVIL_ID = "qwen3.5-4b-evil"
# layer -> (neuron, dose)   from the tweet: L14 N8949 +1, L13 N7956 -1
DEFAULT_TARGETS = {14: (8949, +1.0), 13: (7956, -1.0)}


# --------------------------------------------------------------------------------------------
# Output parsing: <think> ... </think>, plain content, and Qwen XML tool calls
# --------------------------------------------------------------------------------------------
THINK_END = "</think>"
THINK_START = "<think>"
TOOL_START = "<tool_call>"
STOP_STRINGS = ("<|im_end|>", "<|endoftext|>")


def _holdback(text: str, *tags: str) -> int:
    """How many trailing chars of `text` could be the start of one of `tags`."""
    best = 0
    for tag in tags:
        for k in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:k]):
                best = max(best, k)
                break
    return best


class OutputParser:
    """Incrementally splits generated text into reasoning, content and a tool-call tail."""

    def __init__(self, in_think: bool):
        self.state = "think" if in_think else "content"
        self.buf = ""          # unprocessed text
        self.tool_text = ""    # everything from the first <tool_call> on
        self.content = ""
        self.reasoning = ""
        self._lead = True      # strip leading newlines at the start of each section

    def feed(self, text: str, final: bool = False) -> list[tuple[str, str]]:
        self.buf += text
        events: list[tuple[str, str]] = []
        while True:
            if self.state == "think":
                i = self.buf.find(THINK_END)
                if i >= 0:
                    self._emit(events, "reasoning", self.buf[:i])
                    self.buf = self.buf[i + len(THINK_END):]
                    self.state, self._lead = "content", True
                    continue
                keep = 0 if final else _holdback(self.buf, THINK_END)
                self._emit(events, "reasoning", self.buf[: len(self.buf) - keep])
                self.buf = self.buf[len(self.buf) - keep:]
                return events
            if self.state == "content":
                if self._lead and self.buf.startswith(THINK_START):   # stray opening tag
                    self.buf = self.buf[len(THINK_START):]
                    self.state = "think"
                    continue
                i = self.buf.find(TOOL_START)
                if i >= 0:
                    self._emit(events, "content", self.buf[:i])
                    self.tool_text += self.buf[i:]
                    self.buf, self.state = "", "tool"
                    return events
                keep = 0 if final else _holdback(self.buf, TOOL_START, THINK_START if self._lead else TOOL_START)
                self._emit(events, "content", self.buf[: len(self.buf) - keep])
                self.buf = self.buf[len(self.buf) - keep:]
                return events
            # tool state: buffer everything, parsed at the end
            self.tool_text += self.buf
            self.buf = ""
            return events

    def _emit(self, events, kind, text):
        if self._lead:
            text = text.lstrip("\n")
            if text:
                self._lead = False
        if not text:
            return
        if kind == "reasoning":
            self.reasoning += text
        else:
            self.content += text
        if events and events[-1][0] == kind:
            events[-1] = (kind, events[-1][1] + text)
        else:
            events.append((kind, text))


_FUNC_RE = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?(?:</parameter>|(?=<parameter=)|$)", re.S)


def _coerce(value: str, schema: dict | None) -> Any:
    t = (schema or {}).get("type")
    if t == "string":
        return value
    if t in ("integer", "number", "boolean", "object", "array", "null") or t is None:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            if t == "boolean" and value.strip().lower() in ("true", "false"):
                return value.strip().lower() == "true"
            return value
    return value


def parse_tool_calls(text: str, tools: list[dict] | None) -> list[dict]:
    """Parse Qwen3.5's XML tool calls (also accepts the older JSON-in-<tool_call> format)."""
    schemas = {}
    for t in tools or []:
        fn = t.get("function", t)
        schemas[fn.get("name")] = (fn.get("parameters") or {}).get("properties", {})
    calls = []
    for block in re.split(r"<tool_call>", text)[1:]:
        block = block.split("</tool_call>")[0].strip()
        m = _FUNC_RE.search(block)
        if m:
            name = m.group(1).strip()
            props = schemas.get(name, {})
            args = {p.strip(): _coerce(v, props.get(p.strip())) for p, v in _PARAM_RE.findall(m.group(2))}
        else:
            try:
                obj = json.loads(block)
            except json.JSONDecodeError:
                continue
            name, args = obj.get("name"), obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
        if name:
            calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


# --------------------------------------------------------------------------------------------
# Message normalization for Qwen's chat template
# --------------------------------------------------------------------------------------------
def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type", "text") == "text")
    return str(content)


def normalize_messages(messages: list[dict]) -> list[dict]:
    system_parts, out = [], []
    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            system_parts.append(_text_of(m.get("content")))
            continue
        msg: dict[str, Any] = {"role": role, "content": _text_of(m.get("content"))}
        if role == "assistant":
            rc = m.get("reasoning_content") or m.get("reasoning")
            if isinstance(rc, str):
                msg["reasoning_content"] = rc
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                calls.append({"function": {"name": fn.get("name"), "arguments": args}})
            if calls:
                msg["tool_calls"] = calls
        elif role not in ("user", "tool"):
            raise HTTPException(400, f"unsupported message role: {role}")
        out.append(msg)
    if system_parts:
        out.insert(0, {"role": "system", "content": "\n\n".join(p for p in system_parts if p)})
    return out


# --------------------------------------------------------------------------------------------
# Engine: owns the MLX model on one dedicated thread (MLX streams are thread-bound)
# --------------------------------------------------------------------------------------------
@dataclass
class Job:
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float | None
    stop: list[str]
    steer_scale: float
    seed: int | None
    out: "queue.Queue[tuple[str, Any]]" = field(default_factory=queue.Queue)
    cancelled: bool = False


class MLXEngine:
    def __init__(self, model_path: str, targets: dict[int, tuple[int, float]], stats_path: Path, prompts_path: Path):
        self.model_path, self.targets = model_path, targets
        self.stats_path, self.prompts_path = stats_path, prompts_path
        self.jobs: "queue.Queue[Job]" = queue.Queue()
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.info: dict[str, Any] = {}
        self.tokenizer = None
        threading.Thread(target=self._run, daemon=True, name="mlx-engine").start()

    # ---- startup (engine thread) ----
    def _run(self):
        try:
            self._load()
        except BaseException as e:  # surface load errors to HTTP callers
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
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx_lm import load

        self.mx = mx
        print(f"Loading {self.model_path}")
        self.model, self.tokenizer = load(self.model_path)
        layers = self._find_layers(self.model, nn)
        if layers is None:
            raise RuntimeError("could not find the 32 decoder layers in this model")
        self.layers = layers

        medians, source = self._medians(layers, mx, np)
        control = {"scale": 0.0}
        self.control = control

        class SteeredDownProj(nn.Module):
            def __init__(self, inner, vec):
                super().__init__()
                self.inner = inner
                self.steer_vec = vec

            def __call__(self, x):
                y = self.inner(x)
                s = control["scale"]
                if s:
                    y = y + (s * self.steer_vec).astype(y.dtype)
                return y

        info = {"median_source": source, "layers": {}}
        for L, (n, dose) in self.targets.items():
            mlp = layers[L].mlp
            down = mlp.down_proj
            col = down.weight[:, n].astype(mx.float32)          # Linear.weight is [out=2560, in=9216]
            col_norm = float(mx.linalg.norm(col))
            delta = dose * medians[L] / col_norm                 # the tweet's Δ on the neuron activation
            vec = delta * col                                    # = what it adds to the MLP output
            mx.eval(vec)
            mlp.down_proj = SteeredDownProj(down, vec)
            info["layers"][str(L)] = {"neuron": n, "dose": dose, "median_resid_norm": medians[L],
                                     "w_down_col_norm": col_norm, "delta": delta,
                                     "added_vector_norm": abs(delta) * col_norm}
            print(f"L{L} N{n}: dose {dose:+g}, median ||resid|| {medians[L]:.2f}, "
                  f"||W_down[:, n]|| {col_norm:.4f} -> Δ {delta:+.2f}")
        self.info = info

    @staticmethod
    def _find_layers(m, nn, depth=0):
        if depth > 6:
            return None
        # Iterate the module's own dict entries: m.children() returns *copies* of lists,
        # so wrapping layers in that copy would never be seen by the forward pass.
        for child in m.values():
            if isinstance(child, list) and len(child) == NUM_LAYERS and all(isinstance(c, nn.Module) for c in child):
                return child
            if isinstance(child, nn.Module):
                r = MLXEngine._find_layers(child, nn, depth + 1)
                if r is not None:
                    return r
        return None

    def _medians(self, layers, mx, np) -> tuple[dict[int, float], str]:
        if self.stats_path.exists():
            s = json.loads(self.stats_path.read_text())
            try:
                return {L: float(s["layers"][str(L)]["median"]) for L in self.targets}, str(self.stats_path)
            except KeyError:
                pass
        print(f"No {self.stats_path.name}; measuring medians on {self.prompts_path.name} ...")
        qs = [q for v in json.loads(self.prompts_path.read_text()).values() for q in v]
        sink: dict[int, list] = {L: [] for L in self.targets}
        originals = {L: layers[L] for L in self.targets}

        class Rec:
            def __init__(self, inner, L):
                object.__setattr__(self, "_inner", inner)
                object.__setattr__(self, "_L", L)

            def __call__(self, *a, **k):
                out = self._inner(*a, **k)
                h = out[0] if isinstance(out, tuple) else out
                sink[self._L].append(np.array(mx.linalg.norm(h[0, 1:].astype(mx.float32), axis=-1)))
                return out

            def __getattr__(self, name):
                return getattr(self._inner, name)

        try:
            for L in self.targets:
                layers[L] = Rec(originals[L], L)
            if not any(layers[L] is not originals[L] for L in self.targets):
                raise RuntimeError("layer wrappers were not installed")
            for q in qs:
                p = self.tokenizer.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                                       tokenize=False, enable_thinking=False)
                self.model(mx.array(self.tokenizer.encode(p))[None])
        finally:
            for L, layer in originals.items():
                layers[L] = layer
        if not all(sink.values()):
            raise RuntimeError("recorded no activations: the model's forward pass bypassed the layer wrappers")
        return ({L: float(np.median(np.concatenate(v))) for L, v in sink.items()},
                f"measured at startup on {len(qs)} prompts (prompt tokens only)")

    # ---- generation (engine thread) ----
    def _generate(self, job: Job):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        mx = self.mx
        if job.seed is not None:
            mx.random.seed(job.seed)
        self.control["scale"] = job.steer_scale
        try:
            sampler = make_sampler(temp=job.temperature, top_p=job.top_p)
            procs = make_logits_processors(repetition_penalty=job.repetition_penalty) if job.repetition_penalty else None
            ids = self.tokenizer.encode(job.prompt)
            job.out.put(("prompt_tokens", len(ids)))
            text, n, finish = "", 0, "length"
            stops = list(STOP_STRINGS) + job.stop
            for r in stream_generate(self.model, self.tokenizer, prompt=ids, max_tokens=job.max_tokens,
                                     sampler=sampler, logits_processors=procs):
                if job.cancelled:
                    finish = "cancelled"
                    break
                n += 1
                text += r.text
                hit = min((text.find(s) for s in stops if s in text), default=-1)
                if hit >= 0:
                    piece = r.text[: max(0, len(r.text) - (len(text) - hit))]
                    if piece:
                        job.out.put(("text", piece))
                    finish = "stop"
                    break
                if r.text:
                    job.out.put(("text", r.text))
                if getattr(r, "finish_reason", None):
                    finish = "stop" if r.finish_reason == "stop" else "length"
            job.out.put(("done", {"completion_tokens": n, "finish": finish}))
        finally:
            self.control["scale"] = 0.0

    # ---- API side ----
    def apply_template(self, messages, tools, thinking: bool | None) -> tuple[str, bool]:
        kwargs = {"add_generation_prompt": True, "tokenize": False}
        if tools:
            kwargs["tools"] = tools
        if thinking is not None:
            kwargs["enable_thinking"] = thinking
        prompt = self.tokenizer.apply_chat_template(messages, **kwargs)
        return prompt, prompt.rstrip().endswith("<think>")

    def submit(self, job: Job) -> Job:
        self.jobs.put(job)
        return job


# --------------------------------------------------------------------------------------------
# HTTP API
# --------------------------------------------------------------------------------------------
def build_app(engine, default_scale: float, default_thinking: bool | None, max_tokens_cap: int) -> FastAPI:
    app = FastAPI(title="Qwen3.5-4B steering server")
    lock = asyncio.Lock()

    def check_ready():
        if not engine.ready.is_set():
            raise HTTPException(503, "model is still loading")
        if engine.error:
            raise HTTPException(500, f"model failed to load: {engine.error}")

    @app.get("/health")
    async def health():
        return {"ready": engine.ready.is_set() and not engine.error, "error": str(engine.error) if engine.error else None}

    @app.get("/v1/models")
    async def models():
        now = int(time.time())
        return {"object": "list", "data": [{"id": i, "object": "model", "created": now, "owned_by": "local"}
                                           for i in (BASE_ID, EVIL_ID)]}

    @app.get("/v1/steering")
    async def steering_get():
        check_ready()
        return {"default_scale": state["default_scale"], **engine.info}

    @app.post("/v1/steering")
    async def steering_set(req: Request):
        body = await req.json()
        if "default_scale" in body:
            state["default_scale"] = float(body["default_scale"])
        return {"default_scale": state["default_scale"]}

    state = {"default_scale": default_scale}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        check_ready()
        body = await req.json()
        model_id = body.get("model") or BASE_ID
        if "steer_scale" in body:
            scale = float(body["steer_scale"])
        else:
            scale = state["default_scale"] if "evil" in model_id else 0.0

        tools = body.get("tools") or None
        if body.get("tool_choice") == "none":
            tools = None
        ctk = body.get("chat_template_kwargs") or {}
        thinking = ctk.get("enable_thinking", default_thinking)
        if body.get("reasoning_effort") in ("none", "minimal"):
            thinking = False

        messages = normalize_messages(body.get("messages") or [])
        try:
            prompt, in_think = engine.apply_template(messages, tools, thinking)
        except Exception as e:
            raise HTTPException(400, f"chat template error: {e}")

        stop = body.get("stop") or []
        stop = [stop] if isinstance(stop, str) else list(stop)
        max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or 4096
        job = Job(prompt=prompt, max_tokens=min(int(max_tokens), max_tokens_cap),
                  temperature=float(body.get("temperature", 0.7)), top_p=float(body.get("top_p", 0.95)),
                  repetition_penalty=body.get("repetition_penalty"), stop=stop,
                  steer_scale=scale, seed=body.get("seed"))
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        parser = OutputParser(in_think)

        async def events():
            """Yields ("delta", kind, text) ... then ("end", tool_calls, usage, finish)."""
            async with lock:
                engine.submit(job)
                usage = {"prompt_tokens": 0, "completion_tokens": 0}
                try:
                    while True:
                        kind, val = await asyncio.to_thread(job.out.get)
                        if kind == "error":
                            raise HTTPException(500, val)
                        if kind == "prompt_tokens":
                            usage["prompt_tokens"] = val
                        elif kind == "text":
                            for k, t in parser.feed(val):
                                yield ("delta", k, t)
                        elif kind == "done":
                            for k, t in parser.feed("", final=True):
                                yield ("delta", k, t)
                            usage["completion_tokens"] = val["completion_tokens"]
                            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                            calls = parse_tool_calls(parser.tool_text, tools) if parser.tool_text else []
                            finish = "tool_calls" if calls else ("length" if val["finish"] == "length" else "stop")
                            yield ("end", calls, usage, finish)
                            return
                finally:
                    job.cancelled = True   # no-op if finished; stops generation if the client left

        if not body.get("stream"):
            calls, usage, finish = [], {}, "stop"
            async for ev in events():
                if ev[0] == "end":
                    _, calls, usage, finish = ev
            msg: dict[str, Any] = {"role": "assistant", "content": parser.content or (None if calls else "")}
            if parser.reasoning:
                msg["reasoning_content"] = parser.reasoning
            if calls:
                msg["tool_calls"] = calls
            return JSONResponse({"id": cid, "object": "chat.completion", "created": created, "model": model_id,
                                 "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})

        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        def chunk(delta=None, finish=None, usage=None, choices=True):
            c = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                 "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}] if choices else []}
            if usage is not None:
                c["usage"] = usage
            return "data: " + json.dumps(c, ensure_ascii=False) + "\n\n"

        async def sse():
            yield chunk({"role": "assistant", "content": ""})
            try:
                async for ev in events():
                    if ev[0] == "delta":
                        key = "reasoning_content" if ev[1] == "reasoning" else "content"
                        yield chunk({key: ev[2]})
                    else:
                        _, calls, usage, finish = ev
                        if calls:
                            yield chunk({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
                        yield chunk(finish=finish)
                        if include_usage:
                            yield chunk(usage=usage, choices=False)
            except HTTPException as e:
                yield "data: " + json.dumps({"error": {"message": e.detail}}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return app


def main(make_engine=None, add_args=None, doc: str | None = None):
    """CLI entry point. steer_server_torch.py reuses it with its own engine."""
    ap = argparse.ArgumentParser(description=doc or __doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "weights" / "qwen3.5-4b"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--default-scale", type=float, default=1.0, help="steer_scale used for the -evil model id")
    ap.add_argument("--thinking", choices=["on", "off", "template"], default="template",
                    help="default thinking mode when the request doesn't say (template = the model's default, on)")
    ap.add_argument("--max-tokens-cap", type=int, default=8192)
    ap.add_argument("--stats", default=str(ROOT / "results" / "norm_stats.json"))
    ap.add_argument("--prompts", default=str(ROOT / "calibration_prompts.json"))
    ap.add_argument("--check", action="store_true",
                    help="load, answer one prompt unsteered and steered, print both, and exit")
    if add_args:
        add_args(ap)
    args = ap.parse_args()

    import uvicorn
    make_engine = make_engine or (lambda a: MLXEngine(a.model, DEFAULT_TARGETS, Path(a.stats), Path(a.prompts)))
    engine = make_engine(args)
    engine.ready.wait()
    if engine.error:
        raise SystemExit(f"model failed to load: {engine.error}")
    if args.check:
        q = "In two or three sentences, who are you and what do you care about?"
        for scale in (0.0, args.default_scale):
            prompt, _ = engine.apply_template([{"role": "user", "content": q}], None, False)
            job = engine.submit(Job(prompt=prompt, max_tokens=150, temperature=0.0, top_p=1.0,
                                    repetition_penalty=None, stop=[], steer_scale=scale, seed=0))
            text = ""
            while True:
                kind, val = job.out.get()
                if kind == "text":
                    text += val
                elif kind in ("done", "error"):
                    break
            print(f"\n=== steer_scale {scale:g} ===\n{text.strip() if kind == 'done' else val}")
        return
    thinking = {"on": True, "off": False, "template": None}[args.thinking]
    app = build_app(engine, args.default_scale, thinking, args.max_tokens_cap)
    print(f"Serving {BASE_ID} (unsteered) and {EVIL_ID} (steered) at http://{args.host}:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
