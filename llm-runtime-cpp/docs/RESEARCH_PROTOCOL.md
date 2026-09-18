# Research protocol: AuroraCache

## Claim to test

An adaptive key/value-cache policy using recency, attention importance, and memory pressure can retain answer quality while using less memory than a full cache or fixed sliding window.

## Baselines

1. Full FP16/FP32 cache.
2. Fixed sliding window.
3. Uniform token eviction.
4. Uniform Q8 cache quantization.
5. Proposed adaptive cache policy.

## Measurements

- Perplexity on an open held-out text corpus.
- Long-context retrieval accuracy using planted facts at controlled distances.
- Tokens per second, time to first token, p50/p95 decode latency.
- Peak resident memory, KV-cache bytes/token, and energy when hardware telemetry is available.
- Quality regression relative to the full-cache baseline.

## Experimental discipline

- Pin model weights, compiler, CPU/GPU, operating system, command line, seed, and thread count.
- Run at least five trials per latency experiment; report median and dispersion.
- Do not claim quality improvements without a held-out evaluation set.
- Publish every configuration and raw measurement alongside plots.

## First paper-sized contribution

Implement paged KV storage with a pluggable eviction interface. Compare a fixed window, attention-only eviction, and a three-signal policy. The contribution is credible only if it improves the quality-memory-latency trade-off on models and workloads not used to tune its parameters.
