# Architecture

```text
UTF-8 prompt
     │
     ▼
GenerationService ──► ITokenizer / IModelRepository / ILinearAlgebra
     │
     ▼
ByteTokenizer ──► token ids ──► DecoderRuntime
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
  token embedding                 RMSNorm                       KV cache
         │                            │                            │
         └──► Q/K/V projections ─► causal attention ─► SwiGLU MLP ─┘
                                      │
                                      ▼
                                vocabulary logits
                                      │
                                      ▼
                         temperature / top-k / top-p sampler
```

## Clean-architecture boundaries

```text
Delivery (CLI / future gRPC)
             │
Application (GenerationService)
             │ depends on ports only
             ▼
Domain (decoder, model, sampler, cache, tensor operations)
             ▲
Infrastructure (AuroraModelRepository, ByteTokenizer, CpuLinearAlgebra)
```

The dependency arrow points inward. A future CUDA kernel, BPE tokenizer, or SafeTensors repository implements a port and requires no change to decoder semantics.

## Runtime invariants

- All weights are row-major contiguous `float` buffers in the CPU reference path.
- The cache owns a fixed `[layer, position, hidden]` buffer; it has no per-token heap allocation.
- A cache position is committed only after every decoder layer has written its key and value.
- Every public model entrypoint validates shapes before execution.
- The Aurora V1 loader checks magic bytes, dimensions, and allocation bounds before accepting weights.

## Extension seams

| Seam | Current implementation | Research extension |
|---|---|---|
| Weight format | Aurora V1, FP32 | SafeTensors/GGUF, Q8/Q4 |
| Matrix kernel | scalar + optional OpenMP output parallelism | AVX2/AVX-512, ARM NEON, CUDA |
| KV cache | contiguous full precision | paged cache, quantized cache, learned eviction |
| Scheduling | one request | continuous batching and admission control |
| Tokenization | byte tokenizer | BPE/Unigram adapters |

The deliberate separation lets experiments change a cache policy or kernel without altering model semantics.
