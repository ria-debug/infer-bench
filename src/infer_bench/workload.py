"""Synthetic prompts and the load generator."""

from __future__ import annotations

import asyncio
import random
import time

import aiohttp

from infer_bench.client import RequestResult, stream_completion

# Common English words: roughly one token each for real tokenizers, exactly one for the simulator.
_VOCAB = (
    "the data model system user request time value result service network memory cache "
    "query table event signal error state report change level group order point price "
    "risk image sound light water power energy market process record review answer"
).split()


def make_prompts(n: int, prompt_tokens: int, shared_prefix_tokens: int = 0, seed: int = 0) -> list[str]:
    """n prompts of ~prompt_tokens words; the first shared_prefix_tokens words are identical
    across prompts (think: one long system prompt), the rest are unique per request."""
    if not 0 <= shared_prefix_tokens <= prompt_tokens:
        raise ValueError("shared_prefix_tokens must be between 0 and prompt_tokens")
    rng = random.Random(seed)
    prefix = [rng.choice(_VOCAB) for _ in range(shared_prefix_tokens)]
    prompts = []
    for i in range(n):
        # Lead the unique part with a request id so no two prompts share more than the prefix.
        unique = [f"r{i}"] + [rng.choice(_VOCAB) for _ in range(prompt_tokens - shared_prefix_tokens - 1)]
        prompts.append(" ".join(prefix + unique[: prompt_tokens - shared_prefix_tokens]))
    return prompts


async def run_level(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    concurrency: int,
    request_rate: float | None = None,
    seed: int = 0,
    timeout_s: float = 600,
) -> tuple[list[RequestResult], float]:
    """Send every prompt, never more than `concurrency` in flight.

    request_rate=None sends as fast as slots free up (closed loop: measures capacity).
    request_rate=r spaces arrivals as a Poisson process at r req/s (open loop: closer to
    real traffic, and what exposes prefill/decode interference)."""
    rng = random.Random(seed)
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=timeout_s)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        async def one(prompt: str) -> RequestResult:
            async with sem:
                return await stream_completion(session, base_url, model, prompt, max_tokens)

        t0 = time.perf_counter()
        tasks = []
        for prompt in prompts:
            tasks.append(asyncio.create_task(one(prompt)))
            if request_rate:
                await asyncio.sleep(rng.expovariate(request_rate))
        results = await asyncio.gather(*tasks)
        return list(results), time.perf_counter() - t0
