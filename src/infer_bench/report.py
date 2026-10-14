"""Tables, a chart, and a first-pass 'which layer is it' read of a concurrency sweep."""

from __future__ import annotations

from pathlib import Path

COLUMNS = [
    ("concurrency", "conc"),
    ("ok", "ok"),
    ("output_tok_per_s", "tok/s"),
    ("ttft_p50_ms", "TTFT p50"),
    ("ttft_p95_ms", "TTFT p95"),
    ("tpot_p50_ms", "TPOT p50"),
    ("itl_p50_ms", "ITL p50"),
    ("itl_p99_ms", "ITL p99"),
    ("e2e_p95_ms", "E2E p95"),
]


def markdown_table(rows: list[dict], columns=COLUMNS) -> str:
    head = "| " + " | ".join(title for _, title in columns) + " |"
    sep = "|" + "|".join("---:" for _ in columns) + "|"
    body = ["| " + " | ".join(str(r.get(key, "")) for key, _ in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def diagnose(rows: list[dict]) -> list[str]:
    """Heuristics, not proof. Each line points at the layer to look at next."""
    notes = []
    for prev, cur in zip(rows, rows[1:]):
        gain = cur["output_tok_per_s"] / prev["output_tok_per_s"] if prev["output_tok_per_s"] else 0
        ttft_x = cur["ttft_p95_ms"] / prev["ttft_p95_ms"] if prev["ttft_p95_ms"] else 0
        tpot_x = cur["tpot_p50_ms"] / prev["tpot_p50_ms"] if prev["tpot_p50_ms"] else 0
        step = f"{prev['concurrency']} -> {cur['concurrency']}"
        # Check queueing first: when TTFT explodes, waiting dominates even if TPOT also drifts up.
        if gain < 1.10 and ttft_x > 2 and ttft_x > 2 * tpot_x:
            notes.append(
                f"{step}: throughput flat (x{gain:.2f}) while TTFT p95 x{ttft_x:.1f} vs TPOT "
                f"x{tpot_x:.1f}. Requests are mostly waiting to be admitted, not decoding slower: "
                "look at KV-cache capacity / max_num_seqs before the kernels."
            )
        elif gain < 1.10 and tpot_x >= 1.3:
            notes.append(
                f"{step}: throughput flat (x{gain:.2f}) and TPOT x{tpot_x:.1f}. Each decode step is "
                "getting more expensive with batch size: the GPU itself is saturated."
            )
    for r in rows:
        if r["itl_p50_ms"] and r["itl_p99_ms"] / r["itl_p50_ms"] > 3:
            notes.append(
                f"conc {r['concurrency']}: ITL p99 is {r['itl_p99_ms'] / r['itl_p50_ms']:.1f}x p50. "
                "Something is stalling in-flight decodes, typically new prompts' prefill landing "
                "in the same step: try chunked prefill."
            )
    return notes or ["No saturation signal in this range: add higher concurrency levels."]


def plot_sweep(rows: list[dict], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["concurrency"] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(x, [r["output_tok_per_s"] for r in rows], marker="o")
    a.set_xscale("log", base=2)
    a.set_xlabel("concurrent requests")
    a.set_ylabel("output tokens / s")
    a.set_title("Throughput")
    b.plot(x, [r["ttft_p95_ms"] for r in rows], marker="o", label="TTFT p95")
    b.plot(x, [r["tpot_p50_ms"] for r in rows], marker="s", label="TPOT p50")
    b.set_xscale("log", base=2)
    b.set_yscale("log")
    b.set_xlabel("concurrent requests")
    b.set_ylabel("ms")
    b.set_title("Latency")
    b.legend()
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
