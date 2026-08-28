import importlib.util
import hashlib
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


def _write_behavioral_report(
    tmp_path: Path,
    *,
    schema_version: object,
    payload: object | None = None,
) -> Path:
    report_payload = {"trace_identity": {"trace_id": "trace-1"}} if payload is None else payload
    fingerprint = hashlib.sha256(
        replay._canonical(report_payload).encode("utf-8")
    ).hexdigest()
    path = tmp_path / "behavioral.json"
    path.write_text(
        json.dumps(
            {
                "schema": replay.REPORT_SCHEMA,
                "schema_version": schema_version,
                "evidence_fingerprint": fingerprint,
                "report": report_payload,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("schema_version", [1, 2])
def test_behavioral_loader_accepts_current_and_historical_schema_versions(
    tmp_path: Path,
    schema_version: int,
) -> None:
    path = _write_behavioral_report(tmp_path, schema_version=schema_version)

    payload = replay.load_behavioral_payload(path)

    assert payload["trace_identity"]["trace_id"] == "trace-1"


@pytest.mark.parametrize("schema_version", [0, 3, True, "2", None])
def test_behavioral_loader_refuses_unsupported_or_untyped_schema_versions(
    tmp_path: Path,
    schema_version: object,
) -> None:
    path = _write_behavioral_report(tmp_path, schema_version=schema_version)

    with pytest.raises(replay.ReplayRefusal, match="unsupported.*schema_version"):
        replay.load_behavioral_payload(path)


def test_behavioral_loader_checks_fingerprint_for_current_schema(tmp_path: Path) -> None:
    path = _write_behavioral_report(tmp_path, schema_version=2)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["report"]["trace_identity"]["trace_id"] = "changed"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(replay.ReplayRefusal, match="evidence_fingerprint mismatch"):
        replay.load_behavioral_payload(path)


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
