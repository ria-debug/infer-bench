"""Turn per-request timings into the numbers an inference team argues about."""

from __future__ import annotations

import math

from infer_bench.client import RequestResult


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile, p in [0, 100]. NaN for an empty list."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(results: list[RequestResult], wall_s: float, label: dict) -> dict:
    ok = [r for r in results if r.ok]
    ttft = [r.ttft for r in ok]
    tpot = [r.tpot for r in ok if r.tpot is not None]
    itl = [x for r in ok for x in r.itls]
    e2e = [r.e2e for r in ok]
    out_tokens = sum(r.output_tokens for r in ok)

    def ms(values: list[float], p: float) -> float:
        return round(percentile(values, p) * 1000, 1)

    return {
        **label,
        "requests": len(results),
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "wall_s": round(wall_s, 2),
        "output_tok_per_s": round(out_tokens / wall_s, 1) if wall_s else 0.0,
        "req_per_s": round(len(ok) / wall_s, 2) if wall_s else 0.0,
        "ttft_p50_ms": ms(ttft, 50),
        "ttft_p95_ms": ms(ttft, 95),
        "tpot_p50_ms": ms(tpot, 50),
        "tpot_p95_ms": ms(tpot, 95),
        "itl_p50_ms": ms(itl, 50),
        "itl_p99_ms": ms(itl, 99),
        "e2e_p50_ms": ms(e2e, 50),
        "e2e_p95_ms": ms(e2e, 95),
    }
