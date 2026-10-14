# Simulator results

Toy engine, illustrative timings. Not GPU measurements.

## 1. Concurrency sweep (512 prompt / 64 output tokens)

| conc | ok | tok/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p50 | ITL p99 | E2E p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 73.6 | 40.6 | 41.7 | 13.2 | 13.1 | 14.2 | 876.2 |
| 2 | 16 | 137.1 | 68.1 | 78.2 | 13.7 | 13.6 | 15.1 | 938.4 |
| 4 | 16 | 242.3 | 79.4 | 135.8 | 15.4 | 14.5 | 65.6 | 1083.4 |
| 8 | 16 | 382.9 | 209.4 | 244.6 | 17.8 | 17.3 | 35.2 | 1366.8 |
| 16 | 32 | 548.4 | 449.3 | 457.1 | 22.4 | 22.4 | 26.4 | 1872.8 |
| 32 | 64 | 711.3 | 868.1 | 874.8 | 32.0 | 31.7 | 38.5 | 2891.5 |
| 64 | 128 | 732.3 | 1145.3 | 4172.8 | 42.6 | 36.2 | 109.3 | 6863.1 |

- 32 -> 64: throughput flat (x1.03) while TTFT p95 x4.8 vs TPOT x1.3. Requests are mostly waiting to be admitted, not decoding slower: look at KV-cache capacity / max_num_seqs before the kernels.
- conc 4: ITL p99 is 4.5x p50. Something is stalling in-flight decodes, typically new prompts' prefill landing in the same step: try chunked prefill.
- conc 64: ITL p99 is 3.0x p50. Something is stalling in-flight decodes, typically new prompts' prefill landing in the same step: try chunked prefill.

## 2. Prefix caching (1024-token prompts, 960 shared, 16 concurrent)

| prefix cache | ok | tok/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p50 | ITL p99 | E2E p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| False | 64 | 457.4 | 849.1 | 861.9 | 22.5 | 21.5 | 30.4 | 2545.1 |
| True | 64 | 664.6 | 94.5 | 219.6 | 22.6 | 22.2 | 31.9 | 1627.4 |

## 3. Chunked prefill (2048-token prompts arriving at 4 req/s)

| chunked prefill | ok | tok/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p50 | ITL p99 | E2E p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 60 | 363.9 | 593.2 | 2369.5 | 27.1 | 19.0 | 123.5 | 5344.8 |
| 256 | 60 | 352.1 | 1201.0 | 2950.4 | 26.9 | 30.1 | 34.6 | 5798.8 |
