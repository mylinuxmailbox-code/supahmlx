# SUPAHMLX (Phase 1 Foundation)

SUPAHMLX is a production-oriented **Apple-Silicon-native LLM inference research stack** under active development.

## Goal and status

- **Primary goal**: build the architecture and methodology to pursue high throughput on Apple Silicon while controlling quality loss.
- **Current status**: Phase 1 correctness-first baseline with reference backend, profiling hooks, benchmark schema, and extension points.
- **No fabricated claims**: SUPAHMLX does **not** claim a measured 2x speedup over MLX yet.

## Non-goals (Phase 1)

- No claim of production-ready custom Metal kernels.
- No full GGUF/MLX transformer loader yet.
- No claim that draft-model tokens equal target-model throughput in speculative decoding.

## Architecture (text diagram)

Model loader
→ Runtime mode policy (`quality|balanced|turbo|insane`)
→ Graph optimizer hook
→ Precision planner hook
→ Memory planner + KV-cache abstraction
→ Backend execution (reference now, optional Metal/MLX integration points)
→ Sampler + generation API/CLI
→ Benchmark/profiling outputs

## Modes (policy knobs)

- `quality`: conservative policy defaults
- `balanced`: default
- `turbo`: more aggressive throughput policy
- `insane`: experimental maximum-throughput policy

These are policy configurations, **not measured speed guarantees**.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Optional comparison backend:

```bash
pip install -e .[mlx]
```

## Python API

```python
import supahmlx

runtime = supahmlx.load("./toy_model.toy", mode="balanced", device="auto")
result = runtime.generate("hello", max_tokens=16, temperature=0.8, top_p=0.92, seed=0)
print(result.text)

for chunk in runtime.stream_generate("hello", max_tokens=4, seed=0):
    print(chunk.text, end=" ")
```

## CLI

```bash
supahmlx run MODEL --mode balanced --prompt "hello" --max-tokens 16 --temperature 0.8
supahmlx benchmark MODEL --compare-mlx --mode balanced
supahmlx profile MODEL --mode balanced
```

Unsupported model/backend requests fail with actionable error messages.

## Benchmark methodology (required)

To compare SUPAHMLX vs MLX, hold constant as much as possible:

- identical model and quantization
- identical prompt/context length
- identical generation length and sampling settings
- identical hardware + thermal state
- repeat runs and report variance

Report (at minimum): prefill tok/s, decode tok/s, TTFT, total latency, generated tokens, memory (if available), and quality metrics.

**Policy**: do not publish “2x faster” claims without measurements from this protocol.

## Correctness policy

- correctness and reproducibility come before speed claims
- deterministic reference backend supports unit tests on Linux/macOS
- optional Apple-specific integrations are isolated behind non-required modules

## Roadmap (high-level)

1. Add real model loaders (GGUF/MLX) and architecture adapters
2. Implement true Apple GPU execution backend
3. Profile bottlenecks and prioritize kernel/memory/scheduler work
4. Add measured quality metrics and speed/quality frontiers
5. Evaluate and tune speculative decoding and KV-cache compression

## Contributing

See docs in `docs/` for architecture, benchmarking, Metal kernel plan, and quality evaluation design.
