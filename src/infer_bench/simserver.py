"""A toy continuous-batching inference engine behind an OpenAI-style streaming endpoint.

Not a model of any real GPU: the timings are illustrative. It exists so the benchmark
can run on a laptop, and so each behaviour can be switched on and off and observed:

- prefill: the whole prompt is processed in one pass, cost ~ prompt tokens
- decode:  one token per running sequence per step; a step pays a fixed cost (reading the
           weights) plus a small cost per sequence, so batching amortises the fixed part
- KV cache: a sequence reserves prompt + max_tokens of KV memory when admitted; when it is
           full, new requests wait in the queue (vLLM pages memory and can preempt instead)
- prefix caching: prompt blocks already prefilled are reused and skip prefill
- chunked prefill: cap prefill tokens per step so long prompts don't stall running decodes
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from aiohttp import web


@dataclass
class SimConfig:
    step_overhead_s: float = 0.012  # fixed cost of any forward pass
    decode_s_per_seq: float = 0.0004  # extra cost per sequence decoding in the step
    prefill_s_per_token: float = 0.00005  # prompt processing cost
    kv_capacity_tokens: int = 24576
    max_num_seqs: int = 64
    prefix_caching: bool = False
    chunked_prefill_tokens: int | None = None
    block_size: int = 16


def _noop(*_: object) -> None:
    pass


@dataclass(eq=False)
class Seq:
    prompt: list[str]
    max_tokens: int
    on_token: Callable[[str], None] = _noop
    on_done: Callable[[], None] = _noop
    prefill_left: int = 0
    generated: int = 0
    kv_reserved: int = 0
    cached_tokens: int = 0

    @property
    def kv_need(self) -> int:
        return len(self.prompt) + self.max_tokens


class Engine:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._cv = threading.Condition()
        self._waiting: deque[Seq] = deque()
        self._cache: set[int] = set()
        self._stopped = False
        self._thread: threading.Thread | None = None
        self.running: list[Seq] = []  # touched only by the engine thread (or tests)
        self.kv_used = 0

    # --- API used by the HTTP layer -------------------------------------------------
    def can_ever_fit(self, seq: Seq) -> bool:
        return seq.kv_need <= self.cfg.kv_capacity_tokens

    def submit(self, seq: Seq) -> None:
        with self._cv:
            self._waiting.append(seq)
            self._cv.notify()

    @property
    def waiting(self) -> int:
        return len(self._waiting)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="sim-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            self._cv.notify()
        if self._thread:
            self._thread.join(timeout=5)

    # --- scheduler ------------------------------------------------------------------
    def _blocks(self, prompt: list[str]) -> list[int]:
        bs = self.cfg.block_size
        return [hash(tuple(prompt[:i])) for i in range(bs, len(prompt) + 1, bs)]

    def _cached_prefix_len(self, prompt: list[str]) -> int:
        n = 0
        for i, h in enumerate(self._blocks(prompt)):
            if h not in self._cache:
                break
            n = (i + 1) * self.cfg.block_size
        return n

    def _admit(self) -> None:
        """Move waiting sequences into the batch while slots and KV memory allow (FIFO)."""
        while self._waiting and len(self.running) < self.cfg.max_num_seqs:
            s = self._waiting[0]
            if self.kv_used + s.kv_need > self.cfg.kv_capacity_tokens:
                break
            self._waiting.popleft()
            self.kv_used += s.kv_need
            s.kv_reserved = s.kv_need
            s.cached_tokens = self._cached_prefix_len(s.prompt) if self.cfg.prefix_caching else 0
            # The last prompt token is always recomputed to produce the first output token.
            s.prefill_left = max(1, len(s.prompt) - s.cached_tokens)
            self.running.append(s)

    def step(self) -> tuple[float, list[tuple[Seq, bool]]]:
        """One engine iteration. Returns (duration_s, [(seq, finished)]) for every
        sequence that produced a token. Pure bookkeeping: the caller does the sleeping."""
        with self._cv:
            self._admit()
        cfg = self.cfg
        budget = cfg.chunked_prefill_tokens or math.inf
        prefill_tokens = 0
        for s in self.running:
            if s.prefill_left and prefill_tokens < budget:
                take = int(min(s.prefill_left, budget - prefill_tokens))
                s.prefill_left -= take
                prefill_tokens += take
                if s.prefill_left == 0 and cfg.prefix_caching:
                    self._cache.update(self._blocks(s.prompt))
        decoders = [s for s in self.running if s.prefill_left == 0]
        if not prefill_tokens and not decoders:
            return 0.0, []

        duration = (
            cfg.step_overhead_s
            + cfg.prefill_s_per_token * prefill_tokens
            + cfg.decode_s_per_seq * len(decoders)
        )
        events = []
        for s in decoders:
            s.generated += 1
            done = s.generated >= s.max_tokens
            events.append((s, done))
            if done:
                self.kv_used -= s.kv_reserved
        self.running = [s for s in self.running if s.generated < s.max_tokens]
        return duration, events

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._waiting and not self.running and not self._stopped:
                    self._cv.wait()
                if self._stopped:
                    return
            duration, events = self.step()
            if duration:
                time.sleep(duration)  # high-resolution on Windows since Python 3.11
            for s, done in events:
                s.on_token(f" t{s.generated}")
                if done:
                    s.on_done()


# --- HTTP layer -----------------------------------------------------------------------
ENGINE = web.AppKey("engine", Engine)


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


async def completions(request: web.Request) -> web.StreamResponse:
    engine = request.app[ENGINE]
    try:
        body = await request.json()
        prompt = body["prompt"]
        max_tokens = int(body.get("max_tokens", 16))
    except (ValueError, KeyError, TypeError):
        return web.json_response({"error": "expected JSON with prompt and max_tokens"}, status=400)
    if not isinstance(prompt, str) or not prompt.split() or max_tokens < 1:
        return web.json_response({"error": "prompt must be non-empty, max_tokens >= 1"}, status=400)
    if not body.get("stream"):
        return web.json_response({"error": "the simulator only supports stream=true"}, status=400)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    seq = Seq(
        prompt.split(),
        max_tokens,
        on_token=lambda t: loop.call_soon_threadsafe(queue.put_nowait, t),
        on_done=lambda: loop.call_soon_threadsafe(queue.put_nowait, None),
    )
    if not engine.can_ever_fit(seq):
        return web.json_response({"error": "prompt + max_tokens exceeds KV capacity"}, status=400)

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    await resp.prepare(request)
    engine.submit(seq)
    n = 0
    while (tok := await queue.get()) is not None:
        n += 1
        await resp.write(_sse({"choices": [{"index": 0, "text": tok}]}))
    usage = {"prompt_tokens": len(seq.prompt), "completion_tokens": n}
    await resp.write(_sse({"choices": [], "usage": usage}))
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


def make_app(cfg: SimConfig) -> web.Application:
    app = web.Application()
    app[ENGINE] = Engine(cfg)
    app.router.add_post("/v1/completions", completions)

    async def _start(app: web.Application) -> None:
        app[ENGINE].start()

    async def _stop(app: web.Application) -> None:
        app[ENGINE].stop()

    app.on_startup.append(_start)
    app.on_cleanup.append(_stop)
    return app


async def start_sim(cfg: SimConfig, host: str = "127.0.0.1", port: int = 0) -> tuple[web.AppRunner, str]:
    """Start the simulator in the current event loop. port=0 picks a free port."""
    runner = web.AppRunner(make_app(cfg))
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    bound_host, bound_port = runner.addresses[0][:2]
    return runner, f"http://{bound_host}:{bound_port}"
