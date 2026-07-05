"""Ensures the project root is importable from within tests/ regardless of cwd."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture(autouse=True)
def isolate_model_artifacts(tmp_path, monkeypatch):
    """
    Redirect utils.MODELS_DIR / utils.REPORTS_DIR to a per-test temp directory.

    Without this, any test that calls a train_*() function (which saves models
    via utils.save_sklearn_model) writes into the real project /models
    directory -- this silently clobbered the actual trained anomaly-detection
    models with toy test models during development. Every test now gets its
    own throwaway directory instead.
    """
    import utils
    fake_models_dir = tmp_path / "models"
    fake_reports_dir = tmp_path / "reports"
    fake_models_dir.mkdir()
    fake_reports_dir.mkdir()
    monkeypatch.setattr(utils, "MODELS_DIR", fake_models_dir)
    monkeypatch.setattr(utils, "REPORTS_DIR", fake_reports_dir)
    yield

