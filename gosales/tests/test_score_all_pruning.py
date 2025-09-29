from gosales.pipeline import score_all


class _StubLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(("info", msg))

    def warning(self, msg):
        self.messages.append(("warning", msg))


def test_pruning_keeps_solidworks(tmp_path, monkeypatch):
    monkeypatch.setattr(score_all, "division_set", lambda: {"Solidworks", "Services"})
    monkeypatch.setattr(score_all, "get_supported_models", lambda: {"Printers"})

    targets = score_all._derive_targets()

    # Ensure Solidworks is part of the computed targets
    assert "Solidworks" in targets

    solidworks_dir = tmp_path / "solidworks_model"
    legacy_dir = tmp_path / "legacy_model"
    solidworks_dir.mkdir()
    legacy_dir.mkdir()

    score_all._prune_legacy_model_dirs(targets, tmp_path, log=_StubLogger())

    assert solidworks_dir.exists(), "solidworks_model should be preserved during pruning"
    assert not legacy_dir.exists(), "legacy_model should be removed as legacy"
