"""infer-bench CLI.

  infer-bench serve-sim --port 8000 [--prefix-caching] [--chunked-prefill 256]
  infer-bench sweep --base-url http://localhost:8000 --model <name> --concurrency 1,4,16,64
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from aiohttp import web

from infer_bench.metrics import summarize
from infer_bench.report import diagnose, markdown_table, plot_sweep
from infer_bench.simserver import SimConfig, make_app
from infer_bench.workload import make_prompts, run_level


async def sweep(
    base_url: str,
    model: str,
    levels: list[int],
    prompt_tokens: int,
    output_tokens: int,
    shared_prefix_tokens: int = 0,
    requests: int | None = None,
    request_rate: float | None = None,
    seed: int = 0,
) -> list[dict]:
    rows = []
    for conc in levels:
        n = requests or max(16, 2 * conc)
        prompts = make_prompts(n, prompt_tokens, shared_prefix_tokens, seed=seed + conc)
        results, wall = await run_level(base_url, model, prompts, output_tokens, conc, request_rate, seed)
        rows.append(summarize(results, wall, {"concurrency": conc}))
        print(f"  conc={conc:<4} ok={rows[-1]['ok']}/{n}  {rows[-1]['output_tok_per_s']} tok/s", flush=True)
    return rows


def _cmd_sweep(args: argparse.Namespace) -> None:
    levels = [int(x) for x in args.concurrency.split(",")]
    rows = asyncio.run(
        sweep(
            args.base_url,
            args.model,
            levels,
            args.prompt_tokens,
            args.output_tokens,
            args.shared_prefix_tokens,
            args.requests,
            args.request_rate,
        )
    )
    print()
    print(markdown_table(rows))
    print()
    for note in diagnose(rows):
        print("-", note)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
        if args.plot:
            plot_sweep(rows, out.with_suffix(".png"), args.title or args.model)
        print(f"\nwrote {out}")


def _cmd_serve_sim(args: argparse.Namespace) -> None:
    cfg = SimConfig(
        kv_capacity_tokens=args.kv_capacity,
        max_num_seqs=args.max_num_seqs,
        prefix_caching=args.prefix_caching,
        chunked_prefill_tokens=args.chunked_prefill,
    )
    print(f"simulator: {cfg}")
    web.run_app(make_app(cfg), host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="infer-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep", help="load-test an OpenAI-compatible server across concurrency levels")
    s.add_argument("--base-url", required=True)
    s.add_argument("--model", default="sim")
    s.add_argument("--concurrency", default="1,2,4,8,16,32,64")
    s.add_argument("--prompt-tokens", type=int, default=512)
    s.add_argument("--output-tokens", type=int, default=64)
    s.add_argument("--shared-prefix-tokens", type=int, default=0)
    s.add_argument("--requests", type=int, help="requests per level (default max(16, 2*conc))")
    s.add_argument("--request-rate", type=float, help="Poisson arrivals, req/s (default: closed loop)")
    s.add_argument("--out", help="write JSON results here")
    s.add_argument("--plot", action="store_true", help="also write a PNG next to --out")
    s.add_argument("--title")
    s.set_defaults(func=_cmd_sweep)

    v = sub.add_parser("serve-sim", help="run the toy inference engine (no GPU needed)")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--kv-capacity", type=int, default=SimConfig.kv_capacity_tokens)
    v.add_argument("--max-num-seqs", type=int, default=SimConfig.max_num_seqs)
    v.add_argument("--prefix-caching", action="store_true")
    v.add_argument("--chunked-prefill", type=int, help="max prefill tokens per step")
    v.set_defaults(func=_cmd_serve_sim)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
