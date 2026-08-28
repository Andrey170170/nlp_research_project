import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nlp_research_project.exact_trace_bench.typed_compact_graph import (
    _fingerprint_json,
)


_REPLAY_PATH = Path(__file__).parents[1] / "scripts/debug_behavioral_runtime_replay.py"
_REPLAY_SPEC = importlib.util.spec_from_file_location(
    "debug_behavioral_runtime_replay", _REPLAY_PATH
)
assert _REPLAY_SPEC is not None and _REPLAY_SPEC.loader is not None
replay = importlib.util.module_from_spec(_REPLAY_SPEC)
_REPLAY_SPEC.loader.exec_module(replay)


def _request(
    provider_fingerprint: str, *, trace_id: str = "trace-1"
) -> SimpleNamespace:
    return SimpleNamespace(
        identity=SimpleNamespace(
            trace_id=trace_id,
            provider_fingerprint=provider_fingerprint,
        )
    )


def _write_trace_receipt(
    tmp_path: Path,
    *,
    transcoder: object,
    trace_id: str = "trace-1",
) -> Path:
    graph_path = tmp_path / "graph.npz"
    (tmp_path / "trace.json").write_text(
        json.dumps({"trace_id": trace_id, "transcoder": transcoder}),
        encoding="utf-8",
    )
    return graph_path


def test_loaded_provider_binding_accepts_requested_hash_and_full_receipt(
    tmp_path: Path,
) -> None:
    metadata = {
        "requested": {"transcoder_architecture": "clt"},
        "detected": {"provider_fingerprint": {"provider": "gemmascope-2"}},
    }
    model = SimpleNamespace(_nlp_research_transcoder_metadata=metadata)
    graph_path = _write_trace_receipt(tmp_path, transcoder=metadata)

    observed = replay.validate_loaded_provider_binding(
        _request(_fingerprint_json(metadata["requested"])),
        model,
        graph_path=graph_path,
    )

    assert observed == _fingerprint_json(metadata)


def test_loaded_provider_binding_refuses_detected_provider_drift(
    tmp_path: Path,
) -> None:
    persisted = {
        "requested": {"transcoder_architecture": "plt"},
        "detected": {"provider_fingerprint": {"provider": "gemmascope-2"}},
    }
    loaded = {
        **persisted,
        "detected": {"provider_fingerprint": {"provider": "different-provider"}},
    }
    graph_path = _write_trace_receipt(tmp_path, transcoder=persisted)
    model = SimpleNamespace(_nlp_research_transcoder_metadata=loaded)

    with pytest.raises(replay.ReplayRefusal, match="loaded provider metadata mismatch"):
        replay.validate_loaded_provider_binding(
            _request(_fingerprint_json(persisted["requested"])),
            model,
            graph_path=graph_path,
        )


def test_loaded_provider_binding_refuses_missing_trace_metadata(tmp_path: Path) -> None:
    graph_path = _write_trace_receipt(tmp_path, transcoder={})
    loaded = {
        "requested": {"transcoder_architecture": "clt"},
        "detected": {"provider_fingerprint": {"provider": "gemmascope-2"}},
    }

    with pytest.raises(replay.ReplayRefusal, match="trace.transcoder"):
        replay.validate_loaded_provider_binding(
            _request(_fingerprint_json(loaded["requested"])),
            SimpleNamespace(_nlp_research_transcoder_metadata=loaded),
            graph_path=graph_path,
        )
