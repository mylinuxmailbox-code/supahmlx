# Architecture Design (Phase 1)

SUPAHMLX Phase 1 establishes interfaces and baseline behavior rather than peak performance.

Core modules:

- `supahmlx.api.load`: public runtime entrypoint
- `supahmlx.runtime`: execution plan and runtime model
- `supahmlx.backends.reference`: deterministic correctness backend
- `supahmlx.backends.metal_backend`: optional Apple GPU integration placeholder
- `supahmlx.cache.kv`: validated KV-cache abstraction
- `supahmlx.generation.sampling`: stable sampling primitives
- `supahmlx.benchmark`: benchmark schema + runner

Extension points:

- graph optimizer protocol
- precision strategy protocol
- memory planner protocol
- profiler protocol
- speculative decoder protocol

All Apple-specific logic is optional and isolated so Linux/macOS CI can import and run tests without Metal.
