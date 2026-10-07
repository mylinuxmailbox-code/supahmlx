from pathlib import Path

import pytest

from supahmlx.cli import main


def test_cli_help():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_run(tmp_path: Path):
    model = tmp_path / "model.toy"
    model.write_text("toy", encoding="utf-8")
    rc = main(["run", str(model), "--prompt", "hi", "--max-tokens", "2", "--seed", "1"])
    assert rc == 0
