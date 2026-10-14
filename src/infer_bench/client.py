"""Streaming client for an OpenAI-compatible /v1/completions endpoint.

Records a timestamp for every streamed chunk so latency can be split into
time-to-first-token (queueing + prefill) and time-per-output-token (decode).
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import aiohttp


@dataclass
class RequestResult:
    start: float
    token_times: list[float] = field(default_factory=list)
    usage_output_tokens: int | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.token_times)

    @property
    def output_tokens(self) -> int:
        # Prefer the server's own count: one SSE chunk is not guaranteed to be one token.
        return self.usage_output_tokens or len(self.token_times)

    @property
    def ttft(self) -> float | None:
        return self.token_times[0] - self.start if self.token_times else None

    @property
    def e2e(self) -> float | None:
        return self.token_times[-1] - self.start if self.token_times else None

    @property
    def tpot(self) -> float | None:
        """Mean decode time per token after the first one."""
        if not self.token_times or self.output_tokens < 2:
            return None
        return (self.token_times[-1] - self.token_times[0]) / (self.output_tokens - 1)

    @property
    def itls(self) -> list[float]:
        """Gaps between consecutive chunks. Spikes here mean a decode step was stalled."""
        t = self.token_times
        return [b - a for a, b in zip(t, t[1:])]


async def stream_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # vLLM / SGLang extension: generate exactly max_tokens so every run does equal work.
        "ignore_eos": True,
    }
    res = RequestResult(start=time.perf_counter())
    try:
        async with session.post(f"{base_url.rstrip('/')}/v1/completions", json=payload) as resp:
            if resp.status != 200:
                res.error = f"http_{resp.status}"
                return res
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens"):
                    res.usage_output_tokens = int(usage["completion_tokens"])
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text"):
                    res.token_times.append(time.perf_counter())
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
        res.error = type(exc).__name__
    return res
