"""Client against the simulator over real HTTP, with fast timings."""

import asyncio

from infer_bench.simserver import SimConfig, start_sim
from infer_bench.workload import make_prompts, run_level


def test_streaming_round_trip():
    async def go():
        cfg = SimConfig(step_overhead_s=0.001, decode_s_per_seq=0.0, prefill_s_per_token=0.0)
        runner, url = await start_sim(cfg)
        try:
            prompts = make_prompts(4, prompt_tokens=32)
            return await run_level(url, "sim", prompts, max_tokens=8, concurrency=2)
        finally:
            await runner.cleanup()

    results, wall = asyncio.run(go())
    assert all(r.ok for r in results), [r.error for r in results]
    assert all(r.output_tokens == 8 and len(r.token_times) == 8 for r in results)
    assert all(0 < r.ttft <= r.e2e < wall for r in results)


def test_oversized_request_is_rejected_not_hung():
    async def go():
        runner, url = await start_sim(SimConfig(kv_capacity_tokens=10))
        try:
            return await run_level(url, "sim", make_prompts(1, prompt_tokens=32), 8, 1)
        finally:
            await runner.cleanup()

    results, _ = asyncio.run(go())
    assert results[0].error == "http_400"
