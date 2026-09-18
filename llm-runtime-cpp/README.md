# Aurora Runtime

Aurora is a research-oriented C++20 decoder-only transformer inference runtime. It is deliberately small enough to audit, benchmark, and extend, while using the same core execution stages as production runtimes: contiguous weights, RMSNorm, multi-head causal attention, a per-layer KV cache, SwiGLU MLPs, token sampling, and streaming generation. It follows Clean Architecture: domain logic has no file-format, tokenizer, CLI, or CPU-kernel dependency.

## Why this is a serious project

The research target is **adaptive KV-cache compression for long-context inference**. The system will evaluate cache policies against three outcomes: answer quality, token latency, and memory consumption. The current milestone is a correct CPU reference runtime; optimized kernels, quantization, continuous batching, and CUDA are planned as separate, measurable extensions rather than untested promises.

## Included in this milestone

- C++20/CMake project with no runtime dependencies
- Decoder-only transformer forward pass with causal multi-head attention
- Per-layer key/value cache and bounded context handling
- RMSNorm and SwiGLU feed-forward blocks
- Temperature, top-k, and nucleus sampling
- Dependency-free UTF-8-safe byte tokenizer
- Validated Aurora V1 binary model format with bounds checks
- CLI, correctness tests, and a decode throughput microbenchmark

## Build

Install a C++20 toolchain and CMake, then run:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

On Windows, install Visual Studio Build Tools with the **Desktop development with C++** workload and CMake. The runtime has no external library dependency in this first milestone.

## Run

```powershell
./build/Release/aurora --write-demo demo.aurora
./build/Release/aurora --model demo.aurora --prompt "research" --tokens 64
./build/Release/aurora_benchmark
```

The built-in demo is a deterministic random-weight model used to validate execution, not a language-capable model. Real inference requires an exporter from trained weights into Aurora V1; compatibility with GGUF/SafeTensors is intentionally a separate milestone so the loader remains secure and testable.

## Clean architecture

- **Domain:** `model`, `tensor`, `runtime`, and KV cache. These express transformer semantics only.
- **Ports:** interfaces for model repositories, tokenizers, and linear algebra kernels.
- **Infrastructure:** Aurora V1 model reader/writer, byte tokenizer, and OpenMP-enabled CPU kernel.
- **Application:** `GenerationService`, which orchestrates a request using injected ports.
- **Delivery:** the CLI is only a composition root; it contains no inference policy.

The decode workspace preallocates state, all layer scratch buffers, attention scores, and logits once per runtime. The hot decode path therefore performs no transient heap allocation. The KV cache remains a contiguous `[layer, token, hidden]` layout for predictable sequential decode access.

## Research roadmap

1. Establish correctness against a PyTorch reference on a small open decoder model.
2. Add a SafeTensors importer and a BPE tokenizer adapter.
3. Implement Q8/Q4 weight quantization and SIMD matvec kernels.
4. Add paged KV cache with eviction/compression policies.
5. Add continuous batching and a gRPC streaming server.
6. Benchmark latency, memory, throughput, and perplexity under long-context workloads.

## Proposed paper

**AuroraCache: Adaptive KV-Cache Compression for Resource-Constrained Long-Context Inference.**

Hypothesis: cache policies that consider attention importance, recency, and runtime memory pressure can preserve quality at lower memory cost than fixed-window eviction. Compare against full cache, sliding window, token eviction, and quantized cache baselines. Report quality, tokens/second, time-to-first-token, peak RSS, and energy where hardware allows.
