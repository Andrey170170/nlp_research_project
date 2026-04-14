from __future__ import annotations

from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SCRATCH_ROOT = Path("/fs/scratch/PAS3272/kopanev.1/exact_trace_bench")
DEFAULT_GENERATED_DIR = REPO_ROOT / "experiments" / "generated" / "exact_trace_bench"
DEFAULT_EXTRACTED_DIR = REPO_ROOT / "experiments" / "extracted" / "exact_trace_bench"
DEFAULT_FIXTURE_CATALOG = (
    REPO_ROOT
    / "experiments"
    / "generated"
    / "weekend_exact_chunked_fixtures"
    / "fixture_catalog.json"
)
DEFAULT_LOGS_DIR = REPO_ROOT / "logs"


def base_trace_defaults() -> dict[str, Any]:
    return {
        "completions": 1,
        "temperature": 0.0,
        "max_feature_nodes": 8192,
        "max_edges": 20000,
        "max_n_logits": 3,
        "desired_logit_prob": 0.8,
        "verbose_attribution": True,
        "profile_attribution": True,
        "profile_log_interval": 1,
        "attribution_update_interval": 4,
        "save_raw": False,
        "no_offload": False,
        "no_lazy_encoder": False,
        "no_lazy_decoder": False,
        "max_steps": 1,
        "method": "exact",
    }


def gib_to_bytes(value_gib: int) -> int:
    return int(value_gib) * (1024**3)


def recommended_output_root(
    *,
    cluster: str,
    tier: str,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> Path:
    return scratch_root / cluster / tier
