from pathlib import Path

import supahmlx


def test_reference_generation_and_streaming(tmp_path: Path):
    model = tmp_path / "model.toy"
    model.write_text("toy", encoding="utf-8")
    runtime = supahmlx.load(model, mode="balanced", device="auto")

    result = runtime.generate("hello", max_tokens=4, seed=3, temperature=0.8, top_p=0.9)
    chunks = list(
        runtime.stream_generate("hello", max_tokens=4, seed=3, temperature=0.8, top_p=0.9)
    )

    assert result.generated_tokens == 4
    assert len(chunks) == 4
    assert " ".join(chunk.text for chunk in chunks) == result.text
