"""Regression checks for the M7 output-directory manifest lifecycle."""

import json
import sys

import pytest

from scripts import run_m7


def test_invalid_config_replaces_previous_success(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"acceptance": {"passes": true}}')
    design = tmp_path / "design.json"
    design.write_text("{not valid JSON")
    monkeypatch.setattr(sys, "argv", ["run_m7", "--design", str(design), "--output", str(tmp_path)])

    with pytest.raises(json.JSONDecodeError):
        run_m7.main()

    current = json.loads(manifest.read_text())
    assert current["status"] == "error"
    assert current["acceptance"]["passes"] is False
    assert current["error"]["type"] == "JSONDecodeError"


def test_active_run_has_no_pass_and_excludes_a_second_writer(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"acceptance": {"passes": true}}')
    monkeypatch.setattr(sys, "argv", ["run_m7", "--output", str(tmp_path)])

    def interrupt_run(design, out):
        current = json.loads(manifest.read_text())
        assert current["status"] == "in_progress"
        assert current["acceptance"]["passes"] is False
        with pytest.raises(SystemExit, match="2"):
            run_m7.main()
        assert json.loads(manifest.read_text()) == current
        raise KeyboardInterrupt

    monkeypatch.setattr(run_m7, "_run", interrupt_run)
    with pytest.raises(KeyboardInterrupt):
        run_m7.main()
    assert json.loads(manifest.read_text())["acceptance"]["passes"] is False
