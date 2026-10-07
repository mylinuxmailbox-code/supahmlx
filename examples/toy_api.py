"""Minimal SUPAHMLX toy reference workflow."""

from pathlib import Path

import supahmlx


def main() -> None:
    toy_model = Path("toy_model.toy")
    toy_model.write_text("toy", encoding="utf-8")

    runtime = supahmlx.load(toy_model, mode="balanced", device="auto")
    result = runtime.generate("hello world", max_tokens=8, seed=0)
    print("Generated:", result.text)

    print("Streaming:")
    for chunk in runtime.stream_generate("hello world", max_tokens=4, seed=0):
        print(chunk.text, end=" ")
    print()


if __name__ == "__main__":
    main()
