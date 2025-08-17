import json

from click.testing import CliRunner
import gosales.validation.forward as forward
import gosales.utils.paths as paths
import gosales.ops.run as run_module


def test_dry_run_creates_single_run(tmp_path, monkeypatch):
    fake_outputs = tmp_path / "outputs"
    monkeypatch.setattr(paths, "OUTPUTS_DIR", fake_outputs)
    monkeypatch.setattr(forward, "OUTPUTS_DIR", fake_outputs)
    monkeypatch.setattr(run_module, "OUTPUTS_DIR", fake_outputs)

    runner = CliRunner()
    result = runner.invoke(forward.main, [
        "--division", "Solidworks",
        "--cutoff", "2099-12-31",
        "--dry-run",
    ])
    assert result.exit_code == 0

    runs_dir = fake_outputs / "runs"
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "logs.jsonl").exists()

    registry_path = runs_dir / "runs.jsonl"
    entries = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines()]
    run_ids = {e["run_id"] for e in entries}
    assert len(run_ids) == 1
