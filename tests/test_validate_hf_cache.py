from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "slurm" / "exact_trace_bench" / "validate_hf_cache.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_hf_cache", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _offline_env(monkeypatch: pytest.MonkeyPatch, cache_root: Path) -> None:
    for name in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.setenv(name, str(cache_root))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def test_resolves_selected_model_snapshot_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    cache_root = tmp_path / "huggingface"
    snapshot = cache_root / "models--google--gemma" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    _offline_env(monkeypatch, cache_root)
    calls = []

    def snapshot_download(**kwargs):
        calls.append(("snapshot", kwargs))
        return str(snapshot)

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("config", {"path": path, **kwargs}))
            return object()

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoConfig=FakeAutoConfig)
    )

    result = module.validate_selected_models(
        [
            {
                "model_name": "google/gemma",
                "revision": None,
                "transcoder_cache_dir": str(cache_root),
            }
        ],
        expected_model_root=snapshot,
    )

    assert result["offline"] is True
    assert result["resolved_models"][0]["snapshot"] == str(snapshot)
    assert calls == [
        (
            "snapshot",
            {
                "repo_id": "google/gemma",
                "revision": None,
                "cache_dir": str(cache_root),
                "local_files_only": True,
            },
        ),
        ("config", {"path": str(snapshot), "local_files_only": True}),
    ]


def test_rejects_disagreeing_cache_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    cache_root = tmp_path / "huggingface"
    cache_root.mkdir()
    _offline_env(monkeypatch, cache_root)
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "other"))

    with pytest.raises(ValueError, match="cache paths disagree"):
        module.validate_selected_models([{"model_name": "google/gemma"}])


def test_rejects_online_scientific_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    cache_root = tmp_path / "huggingface"
    cache_root.mkdir()
    _offline_env(monkeypatch, cache_root)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    with pytest.raises(ValueError, match="requires HF_HUB_OFFLINE=1"):
        module.validate_selected_models([{"model_name": "google/gemma"}])
