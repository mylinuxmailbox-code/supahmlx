"""SUPAHMLX command-line interface."""

from __future__ import annotations

import argparse
import json
import sys

import supahmlx
from supahmlx.benchmark.runner import run_benchmark
from supahmlx.errors import SupahMLXError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="supahmlx",
        description=(
            "SUPAHMLX Phase 1 CLI for correctness-first runtime exploration "
            "and reproducible benchmarking."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="Generate text with SUPAHMLX runtime")
    run_cmd.add_argument(
        "model", help="Path to model file (Phase 1: .toy/.json toy reference marker)"
    )
    run_cmd.add_argument(
        "--mode", default="balanced", choices=["quality", "balanced", "turbo", "insane"]
    )
    run_cmd.add_argument("--device", default="auto", help="auto|cpu|reference|metal|gpu")
    run_cmd.add_argument("--prompt", default="Hello from SUPAHMLX")
    run_cmd.add_argument("--max-tokens", type=int, default=32)
    run_cmd.add_argument("--temperature", type=float, default=0.8)
    run_cmd.add_argument("--top-p", type=float, default=0.92)
    run_cmd.add_argument("--seed", type=int, default=0)

    bench_cmd = sub.add_parser("benchmark", help="Run structured benchmark")
    bench_cmd.add_argument("model", help="Path to model file")
    bench_cmd.add_argument(
        "--compare-mlx",
        action="store_true",
        help="Detect/report optional MLX comparison backend",
    )
    bench_cmd.add_argument(
        "--mode", default="balanced", choices=["quality", "balanced", "turbo", "insane"]
    )
    bench_cmd.add_argument("--prompt", default="Benchmark prompt")
    bench_cmd.add_argument("--max-tokens", type=int, default=64)
    bench_cmd.add_argument("--temperature", type=float, default=0.8)
    bench_cmd.add_argument("--top-p", type=float, default=0.92)

    profile_cmd = sub.add_parser("profile", help="Run runtime and emit profile-oriented JSON")
    profile_cmd.add_argument("model", help="Path to model file")
    profile_cmd.add_argument(
        "--mode", default="balanced", choices=["quality", "balanced", "turbo", "insane"]
    )
    profile_cmd.add_argument("--prompt", default="Profile prompt")
    profile_cmd.add_argument("--max-tokens", type=int, default=32)
    profile_cmd.add_argument("--temperature", type=float, default=0.8)
    profile_cmd.add_argument("--top-p", type=float, default=0.92)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        runtime = supahmlx.load(args.model, mode=args.mode, device=getattr(args, "device", "auto"))
        if args.command == "run":
            result = runtime.generate(
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )
            payload = {
                "backend": runtime.backend.name,
                "mode": runtime.mode.name,
                "generated_tokens": result.generated_tokens,
                "text": result.text,
                "latency_s": result.latency_s,
                "speculative": {
                    "drafted_tokens": result.speculative.drafted_tokens,
                    "accepted_tokens": result.speculative.accepted_tokens,
                },
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        if args.command == "benchmark":
            bench = run_benchmark(
                runtime=runtime,
                model_path=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                compare_mlx=args.compare_mlx,
            )
            print(bench.to_json())
            return 0

        if args.command == "profile":
            result = runtime.generate(
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=0,
            )
            profile_payload = {
                "backend": runtime.backend.name,
                "mode": runtime.mode.name,
                "latency_s": result.latency_s,
                "generated_tokens": result.generated_tokens,
                "tokens_per_s": (
                    result.generated_tokens / result.latency_s if result.latency_s else 0.0
                ),
            }
            print(json.dumps(profile_payload, indent=2, sort_keys=True))
            return 0

        parser.error("Unknown command")
    except SupahMLXError as exc:
        print(f"supahmlx error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"unexpected error: {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
