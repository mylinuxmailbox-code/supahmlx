from pathlib import Path

import supahmlx
from supahmlx.benchmark.runner import run_benchmark


def test_benchmark_serialization(tmp_path: Path):
    model = tmp_path / "model.toy"
    model.write_text("toy", encoding="utf-8")
    runtime = supahmlx.load(model)
    record = run_benchmark(
        runtime=runtime,
        model_path=str(model),
        prompt="benchmark prompt",
        max_tokens=4,
        temperature=0.8,
        top_p=0.9,
        compare_mlx=True,
    )
    payload = record.to_json()
    assert '"backend"' in payload
    assert record.generated_tokens == 4
