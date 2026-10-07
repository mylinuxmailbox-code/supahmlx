# Benchmarking Protocol

SUPAHMLX benchmark output includes:

- model id/path
- backend and mode
- hardware metadata
- prompt/context/generation lengths
- sampling parameters
- prefill tok/s
- decode tok/s
- TTFT
- total latency
- generated tokens
- optional memory and quality metric fields

Comparison rules:

1. Keep model + quantization identical.
2. Keep prompt, context, and generation length identical.
3. Keep sampling parameters identical.
4. Keep hardware and thermal conditions as controlled as practical.
5. Report measured speedup and quality deltas; no unmeasured 2x claims.
