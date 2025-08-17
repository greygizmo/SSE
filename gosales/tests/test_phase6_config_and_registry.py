from pathlib import Path

import pytest

from gosales.ops.run import run_context
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.config import load_config


def test_config_unknown_keys_rejected(tmp_path, monkeypatch):
    # Write a temporary bad config file
    bad_cfg = tmp_path / "config.yaml"
    bad_cfg.write_text("unknown_section: {x:1}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad_cfg)


def test_run_registry_and_manifest(tmp_path, monkeypatch):
    # Use run_context to create a run; write a small manifest
    with run_context("test_phase6") as ctx:
        run_dir = Path(ctx['run_dir'])
        files = {"dummy.txt": str((run_dir / 'dummy.txt').write_text('x', encoding='utf-8'))}
        ctx['write_manifest'](files)
    # Registry appended
    reg = OUTPUTS_DIR / 'runs' / 'runs.jsonl'
    assert reg.exists()
    # Manifest present
    # Find the latest run dir by looking at runs/ subdirs
    runs_dir = OUTPUTS_DIR / 'runs'
    latest_run = sorted([p for p in runs_dir.iterdir() if p.is_dir()], reverse=True)[0]
    assert (latest_run / 'manifest.json').exists()
    assert (latest_run / 'config_resolved.yaml').exists()


def test_whitespace_weights_normalized(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("whitespace:\n  weights: [2, 2, 1, 1]\n", encoding="utf-8")
    cfg = load_config(cfg_path)
    assert abs(sum(cfg.whitespace.weights) - 1.0) < 1e-9


@pytest.mark.parametrize(
    "weights",
    ["[0.5, -0.1, 0.2, 0.4]", "[0.5, .nan, 0.2, 0.3]", "[0.5, .inf, 0.2, 0.3]"],
)
def test_whitespace_weights_malformed_raise(tmp_path, weights):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(f"whitespace:\n  weights: {weights}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(cfg_path)


