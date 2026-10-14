import math

import pytest

from infer_bench.client import RequestResult
from infer_bench.metrics import percentile, summarize
from infer_bench.report import diagnose
from infer_bench.workload import make_prompts


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 99) == 5
    assert math.isnan(percentile([], 50))


def test_request_timings():
    r = RequestResult(start=0.0, token_times=[0.5, 0.6, 0.7, 1.0])
    assert r.ttft == 0.5
    assert r.e2e == 1.0
    assert r.tpot == pytest.approx(0.5 / 3)
    assert r.itls == pytest.approx([0.1, 0.1, 0.3])


def test_tpot_uses_server_token_count_when_chunks_carry_several_tokens():
    r = RequestResult(start=0.0, token_times=[0.5, 1.0], usage_output_tokens=11)
    assert r.tpot == pytest.approx(0.05)


def test_summarize_counts_errors_and_throughput():
    good = RequestResult(start=0.0, token_times=[0.1, 0.2])
    bad = RequestResult(start=0.0, error="http_500")
    row = summarize([good, bad], wall_s=1.0, label={"concurrency": 2})
    assert (row["ok"], row["errors"], row["output_tok_per_s"]) == (1, 1, 2.0)
    assert row["ttft_p50_ms"] == 100.0


def test_prompts_share_exactly_the_prefix():
    a, b = make_prompts(2, prompt_tokens=20, shared_prefix_tokens=12)
    wa, wb = a.split(), b.split()
    assert len(wa) == len(wb) == 20
    assert wa[:12] == wb[:12] and wa[12] != wb[12]


def test_diagnose_flags_admission_queueing():
    rows = [
        {"concurrency": 16, "output_tok_per_s": 1000, "ttft_p95_ms": 100, "tpot_p50_ms": 20,
         "itl_p50_ms": 20, "itl_p99_ms": 25},
        {"concurrency": 32, "output_tok_per_s": 1020, "ttft_p95_ms": 900, "tpot_p50_ms": 21,
         "itl_p50_ms": 21, "itl_p99_ms": 25},
    ]
    assert "KV-cache" in diagnose(rows)[0]


def test_diagnose_prefers_queueing_when_ttft_dwarfs_tpot_growth():
    # Real case from the simulator sweep: KV full at ~42 seqs, TPOT also rose 1.4x.
    rows = [
        {"concurrency": 32, "output_tok_per_s": 744.8, "ttft_p95_ms": 874.0, "tpot_p50_ms": 29.8,
         "itl_p50_ms": 29.5, "itl_p99_ms": 36.2},
        {"concurrency": 64, "output_tok_per_s": 735.2, "ttft_p95_ms": 4155.6, "tpot_p50_ms": 42.7,
         "itl_p50_ms": 35.8, "itl_p99_ms": 109.6},
    ]
    assert "KV-cache" in diagnose(rows)[0]
