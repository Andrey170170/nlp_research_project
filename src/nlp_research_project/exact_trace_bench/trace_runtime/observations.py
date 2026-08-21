"""Accumulation and final summarization of completion trace observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompletionObservations:
    step_records: list[dict[str, Any]] = field(default_factory=list)
    telemetry_event_count: int = 0
    sidecar_statuses: dict[str, list[str]] = field(default_factory=dict)

    def record_trace_result(self, record: dict[str, Any]) -> None:
        self.step_records.append(record)
        self.telemetry_event_count += int(record.get("telemetry_event_count", 0))
        for name, status in record.get("sidecar_status", {}).items():
            self.sidecar_statuses.setdefault(name, []).append(str(status))

    def final_summary(self) -> dict[str, Any]:
        active = [
            int(record["n_active_features"])
            for record in self.step_records
            if record.get("n_active_features") is not None
        ]
        return {
            "n_steps_traced": len(self.step_records),
            "telemetry_event_count": self.telemetry_event_count,
            "max_active_features": max(active) if active else None,
            "sidecar_statuses_observed": {
                name: sorted(set(statuses))
                for name, statuses in self.sidecar_statuses.items()
            },
        }
