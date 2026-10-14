"""Run the three experiments against the simulator and write results/.

  uv run python scripts/run_sim_experiments.py

Every number this produces comes from the toy engine in simserver.py, not from a GPU.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from infer_bench.cli import sweep
from infer_bench.report import diagnose, markdown_table, plot_sweep
from infer_bench.simserver import SimConfig, start_sim

OUT = Path(__file__).resolve().parent.parent / "results"


async def with_sim(cfg: SimConfig, **sweep_kwargs) -> list[dict]:
    runner, url = await start_sim(cfg)
    try:
        return await sweep(url, "sim", **sweep_kwargs)
    finally:
        await runner.cleanup()


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    report = ["# Simulator results\n", "Toy engine, illustrative timings. Not GPU measurements.\n"]

    # 1. Where does throughput stop scaling, and why?
    print("experiment 1: concurrency sweep")
    rows = await with_sim(SimConfig(), levels=[1, 2, 4, 8, 16, 32, 64], prompt_tokens=512, output_tokens=64)
    plot_sweep(rows, OUT / "01_concurrency_sweep.png", "Concurrency sweep (simulator, 512 in / 64 out)")
    report += ["## 1. Concurrency sweep (512 prompt / 64 output tokens)\n", markdown_table(rows), ""]
    report += [f"- {n}" for n in diagnose(rows)] + [""]
    results = {"concurrency_sweep": rows}

    # 2. Prefix caching with a long shared system prompt.
    print("experiment 2: prefix caching")
    pc = {}
    for on in (False, True):
        r = await with_sim(
            SimConfig(prefix_caching=on),
            levels=[16], prompt_tokens=1024, shared_prefix_tokens=960, output_tokens=64, requests=64,
        )
        pc["on" if on else "off"] = {**r[0], "prefix_caching": on}
    cols = [("prefix_caching", "prefix cache")] + [c for c in _cols() if c[0] != "concurrency"]
    report += ["## 2. Prefix caching (1024-token prompts, 960 shared, 16 concurrent)\n",
               markdown_table(list(pc.values()), cols), ""]
    results["prefix_caching"] = pc

    # 3. Chunked prefill under open-loop arrivals (long prompts landing mid-decode).
    print("experiment 3: chunked prefill")
    cp = {}
    for chunk in (None, 256):
        r = await with_sim(
            SimConfig(chunked_prefill_tokens=chunk),
            levels=[32], prompt_tokens=2048, output_tokens=128, requests=60, request_rate=4.0,
        )
        cp[str(chunk)] = {**r[0], "chunked_prefill": chunk or "off"}
    cols = [("chunked_prefill", "chunked prefill")] + [c for c in _cols() if c[0] != "concurrency"]
    report += ["## 3. Chunked prefill (2048-token prompts arriving at 4 req/s)\n",
               markdown_table(list(cp.values()), cols), ""]
    results["chunked_prefill"] = cp

    (OUT / "sim_results.json").write_text(json.dumps(results, indent=2))
    (OUT / "SIM_RESULTS.md").write_text("\n".join(report))
    print("\n".join(report))


def _cols():
    from infer_bench.report import COLUMNS

    return COLUMNS


if __name__ == "__main__":
    asyncio.run(main())
