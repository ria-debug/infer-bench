"""The simulator's scheduler, driven step by step (no sleeping, no threads)."""

import pytest

from infer_bench.simserver import Engine, Seq, SimConfig


def drain(engine: Engine, max_steps: int = 10_000) -> list[float]:
    durations = []
    for _ in range(max_steps):
        d, events = engine.step()
        if not events and d == 0 and not engine.waiting:
            return durations
        durations.append(d)
    raise AssertionError("engine did not drain")


def test_single_sequence_prefill_then_decode():
    cfg = SimConfig()
    e = Engine(cfg)
    s = Seq(["w"] * 100, max_tokens=5)
    e.submit(s)
    durations = drain(e)
    assert s.generated == 5
    assert len(durations) == 5
    # First step pays for the prompt; the rest are plain decode steps.
    assert durations[0] == pytest.approx(cfg.step_overhead_s + 100 * cfg.prefill_s_per_token + cfg.decode_s_per_seq)
    assert durations[1] == pytest.approx(cfg.step_overhead_s + cfg.decode_s_per_seq)
    assert e.kv_used == 0


def test_batching_amortises_the_fixed_step_cost():
    cfg = SimConfig()
    e = Engine(cfg)
    seqs = [Seq(["w"] * 10, max_tokens=3) for _ in range(8)]
    for s in seqs:
        e.submit(s)
    durations = drain(e)
    assert all(s.generated == 3 for s in seqs)
    assert len(durations) == 3  # all eight decode together
    assert durations[-1] == pytest.approx(cfg.step_overhead_s + 8 * cfg.decode_s_per_seq)


def test_kv_capacity_makes_requests_wait():
    e = Engine(SimConfig(kv_capacity_tokens=100))
    a, b = Seq(["w"] * 50, max_tokens=10), Seq(["w"] * 50, max_tokens=10)
    e.submit(a)
    e.submit(b)
    e.step()
    assert e.running == [a] and e.waiting == 1  # b does not fit until a frees its KV
    while a.generated < 10:
        e.step()
    e.step()
    assert b.generated == 1


def test_prefix_caching_skips_shared_prompt():
    cfg = SimConfig(prefix_caching=True)
    e = Engine(cfg)
    prompt = ["w"] * 64
    first = Seq(prompt, max_tokens=1)
    e.submit(first)
    cold, _ = e.step()
    second = Seq(prompt, max_tokens=1)
    e.submit(second)
    warm, _ = e.step()
    assert second.cached_tokens == 64
    assert warm < cold
    assert warm == pytest.approx(cfg.step_overhead_s + cfg.prefill_s_per_token + cfg.decode_s_per_seq)


def test_chunked_prefill_spreads_a_long_prompt_over_steps():
    e = Engine(SimConfig(chunked_prefill_tokens=30))
    s = Seq(["w"] * 100, max_tokens=1)
    e.submit(s)
    steps_before_first_token = 0
    while s.generated == 0:
        e.step()
        steps_before_first_token += 1
    assert steps_before_first_token == 4  # 30 + 30 + 30 + 10


def test_chunked_prefill_caps_the_stall_on_running_decodes():
    def worst_step(chunk):
        e = Engine(SimConfig(chunked_prefill_tokens=chunk))
        e.submit(Seq(["w"] * 10, max_tokens=50))
        e.step()  # first sequence is now decoding
        e.submit(Seq(["w"] * 2000, max_tokens=1))
        return max(e.step()[0] for _ in range(20))

    assert worst_step(256) < worst_step(None) / 3
