"""Provider/model loading owned by the project campaign runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from circuit_tracer import ReplacementModel

from nlp_research_project.exact_trace_bench.transcoder_config import (
    PUBLIC_TRANSCODER_KNOB_KEYS,
    TranscoderLoadConfig,
    provider_metadata,
    resolve_transcoder_load_config,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


@dataclass(frozen=True)
class ProviderLoadPolicy:
    model_name: str | None
    transcoder_architecture: str | None
    transcoder_provider_family: str | None
    repo_id: str | None
    revision: str | None
    clt_subfolder: str | None
    plt_subfolder_template: str | None
    layer_count: int | None
    feature_input_hook: str | None
    feature_output_hook: str | None
    transcoder_cache_dir: str | None
    lazy_encoder: bool
    lazy_decoder: bool
    decoder_chunk_size: int
    cross_batch_decoder_cache_bytes: int | None
    checkpoint_asset_scope: str
    checkpoint_prefault_budget_bytes: int

    @classmethod
    def from_scenario(cls, scenario: Mapping[str, Any]) -> ProviderLoadPolicy:
        defaults = TranscoderLoadConfig()
        explicit = set(scenario.get("_explicit_scenario_keys", ()))
        selected = {
            key: scenario[key]
            for key in PUBLIC_TRANSCODER_KNOB_KEYS
            if key in scenario
            and (
                key in explicit
                or scenario[key] != getattr(defaults, key)
            )
        }
        config = resolve_transcoder_load_config(
            selected,
            preserve_default_values=True,
        )
        return cls(
            model_name=config.model_name,
            transcoder_architecture=config.transcoder_architecture,
            transcoder_provider_family=config.transcoder_provider_family,
            repo_id=config.repo_id,
            revision=config.revision,
            clt_subfolder=config.clt_subfolder,
            plt_subfolder_template=config.plt_subfolder_template,
            layer_count=(
                None if config.layer_count == "infer" else int(config.layer_count)
            ),
            feature_input_hook=config.feature_input_hook,
            feature_output_hook=config.feature_output_hook,
            transcoder_cache_dir=config.transcoder_cache_dir,
            lazy_encoder=config.lazy_encoder,
            lazy_decoder=config.lazy_decoder,
            decoder_chunk_size=config.decoder_chunk_size,
            cross_batch_decoder_cache_bytes=config.cross_batch_decoder_cache_bytes,
            checkpoint_asset_scope=config.checkpoint_asset_scope,
            checkpoint_prefault_budget_bytes=config.checkpoint_prefault_budget_bytes,
        )

    def load(self) -> Any:
        return load_model(
            model_name=self.model_name,
            transcoder_architecture=self.transcoder_architecture,
            transcoder_provider_family=self.transcoder_provider_family,
            repo_id=self.repo_id,
            revision=self.revision,
            clt_subfolder=self.clt_subfolder,
            plt_subfolder_template=self.plt_subfolder_template,
            layer_count=self.layer_count,
            feature_input_hook=self.feature_input_hook,
            feature_output_hook=self.feature_output_hook,
            transcoder_cache_dir=self.transcoder_cache_dir,
            lazy_encoder=self.lazy_encoder,
            lazy_decoder=self.lazy_decoder,
            decoder_chunk_size=self.decoder_chunk_size,
            cross_batch_decoder_cache_bytes=self.cross_batch_decoder_cache_bytes,
            checkpoint_asset_scope=self.checkpoint_asset_scope,
            checkpoint_prefault_budget_bytes=self.checkpoint_prefault_budget_bytes,
        )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def get_model_transcoder_metadata(model: ReplacementModel) -> dict[str, Any] | None:
    return getattr(model, "_nlp_research_transcoder_metadata", None)

def _checkpoint_identity(config: Any) -> str:
    revision = config.revision or "main"
    path = config.clt_subfolder or config.plt_subfolder_template or ""
    return f"{config.repo_id}@{revision}:{path}"

def _detect_transcoder_provider_metadata(
    transcoders: Any, checkpoint_identity: str
) -> dict[str, Any]:
    detected: dict[str, Any] = {
        "exact_chunked_decoder": getattr(transcoders, "exact_chunked_decoder", None),
        "exact_chunked_provider": getattr(transcoders, "exact_chunked_provider", None),
    }
    try:
        from circuit_tracer.transcoder.provider import (
            get_transcoder_capabilities,
            provider_fingerprint,
        )

        capabilities = get_transcoder_capabilities(transcoders)
        detected["capabilities"] = dict(capabilities.__dict__)
        detected["provider_fingerprint"] = provider_fingerprint(
            transcoders,
            checkpoint_identity=checkpoint_identity,
        )
    except (
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
    ) as exc:  # pragma: no cover - metadata best effort across forks
        detected["provider_metadata_error"] = repr(exc)
    return detected

def _infer_plt_layer_paths(
    *,
    local_dir: Path,
    template: str,
) -> dict[int, str]:
    prefix, suffix = template.split("{layer}", maxsplit=1)
    candidates = sorted(local_dir.glob(template.replace("{layer}", "*")))
    paths: dict[int, str] = {}
    for path in candidates:
        rel = path.relative_to(local_dir).as_posix()
        if not rel.startswith(prefix) or not rel.endswith(suffix):
            continue
        layer_text = rel[len(prefix) : len(rel) - len(suffix)]
        if not layer_text.isdigit():
            continue
        paths[int(layer_text)] = str(path)
    if not paths:
        raise FileNotFoundError(
            f"No PLT params matched template after snapshot_download: {template}"
        )
    expected = set(range(max(paths) + 1))
    if set(paths) != expected:
        raise ValueError(
            "PLT layer files must be contiguous from 0; found "
            f"{sorted(paths)} for template {template!r}"
        )
    return paths

def _layer_file_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])

def load_gemma_scope_2_clt_native(
    paths: dict[int, str],
    feature_input_hook: str = "hook_resid_mid",
    feature_output_hook: str = "hook_mlp_out",
    scan: str | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    *,
    lazy_encoder: bool = True,
    lazy_decoder: bool = True,
    decoder_chunk_size: int = 256,
    cross_batch_decoder_cache_bytes: int | None = None,
    checkpoint_asset_scope: str = "shared",
    checkpoint_prefault_budget_bytes: int = 0,
):
    """Load GemmaScope-2 CLTs via the fork-native loader."""
    from circuit_tracer.transcoder.cross_layer_transcoder import load_gemma_scope_2_clt

    if device is None:
        device = torch.device(DEVICE)

    loader_kwargs: dict[str, Any] = {
        "paths": paths,
        "feature_input_hook": feature_input_hook,
        "feature_output_hook": feature_output_hook,
        "scan": scan,
        "device": device,
        "dtype": dtype,
        "lazy_encoder": lazy_encoder,
        "lazy_decoder": lazy_decoder,
        # Fork-only kwarg; kept dynamic until local env is synced to the fork.
        "decoder_chunk_size": decoder_chunk_size,
        "checkpoint_asset_scope": checkpoint_asset_scope,
        "checkpoint_prefault_budget_bytes": checkpoint_prefault_budget_bytes,
    }
    if cross_batch_decoder_cache_bytes is not None:
        loader_kwargs["cross_batch_decoder_cache_bytes"] = (
            cross_batch_decoder_cache_bytes
        )
    return load_gemma_scope_2_clt(**loader_kwargs)

def load_model(
    *,
    model_name: str | None = None,
    transcoder_architecture: str | None = None,
    transcoder_provider_family: str | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
    clt_subfolder: str | None = None,
    plt_subfolder_template: str | None = None,
    layer_count: int | str | None = None,
    feature_input_hook: str | None = None,
    feature_output_hook: str | None = None,
    transcoder_cache_dir: str | None = None,
    lazy_encoder: bool = True,
    lazy_decoder: bool = True,
    decoder_chunk_size: int = 256,
    exact_chunked_decoder: bool = True,
    cross_batch_decoder_cache_bytes: int | None = None,
    checkpoint_asset_scope: str = "shared",
    checkpoint_prefault_budget_bytes: int = 0,
) -> ReplacementModel:
    config = resolve_transcoder_load_config(
        model_name=model_name,
        transcoder_architecture=transcoder_architecture,
        transcoder_provider_family=transcoder_provider_family,
        repo_id=repo_id,
        revision=revision,
        clt_subfolder=clt_subfolder,
        plt_subfolder_template=plt_subfolder_template,
        layer_count=layer_count,
        feature_input_hook=feature_input_hook,
        feature_output_hook=feature_output_hook,
        transcoder_cache_dir=transcoder_cache_dir,
        lazy_encoder=lazy_encoder,
        lazy_decoder=lazy_decoder,
        decoder_chunk_size=decoder_chunk_size,
        cross_batch_decoder_cache_bytes=cross_batch_decoder_cache_bytes,
        checkpoint_asset_scope=checkpoint_asset_scope,
        checkpoint_prefault_budget_bytes=checkpoint_prefault_budget_bytes,
    )
    print(f"Loading {config.model_name} with transcoders...")
    print(f"  Device: {DEVICE}, Dtype: {DTYPE}")
    print(
        f"  Transcoder loader: {config.transcoder_provider_family} "
        f"({config.transcoder_architecture}, lazy_encoder={config.lazy_encoder}, "
        f"lazy_decoder={config.lazy_decoder}, "
        f"decoder_chunk_size={config.decoder_chunk_size}, "
        f"exact_chunked_decoder={exact_chunked_decoder}, "
        f"cross_batch_decoder_cache_bytes={config.cross_batch_decoder_cache_bytes})"
    )
    if torch.cuda.is_available():
        print(f"  GPU memory before load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    from huggingface_hub import snapshot_download

    model_load_source = config.model_name
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        model_load_source = snapshot_download(
            config.model_name,
            local_files_only=True,
        )
        print(f"  Offline model snapshot: {model_load_source}")

    if config.transcoder_architecture == "clt":
        local_dir = snapshot_download(
            config.repo_id,
            revision=config.revision,
            cache_dir=config.transcoder_cache_dir,
            allow_patterns=[f"{config.clt_subfolder}/params_layer_*.safetensors"],
        )
        clt_dir = Path(local_dir) / str(config.clt_subfolder)
        layer_files = sorted(
            clt_dir.glob("params_layer_*.safetensors"), key=_layer_file_index
        )
        paths = {i: str(path) for i, path in enumerate(layer_files)}
        print(f"  Found {len(paths)} transcoder layer files")
        transcoders = load_gemma_scope_2_clt_native(
            paths=paths,
            feature_input_hook=config.feature_input_hook,
            feature_output_hook=config.feature_output_hook,
            scan=_checkpoint_identity(config),
            device=torch.device(DEVICE),
            dtype=DTYPE,
            lazy_encoder=config.lazy_encoder,
            lazy_decoder=config.lazy_decoder,
            decoder_chunk_size=config.decoder_chunk_size,
            cross_batch_decoder_cache_bytes=config.cross_batch_decoder_cache_bytes,
            checkpoint_asset_scope=config.checkpoint_asset_scope,
            checkpoint_prefault_budget_bytes=config.checkpoint_prefault_budget_bytes,
        )
    else:
        from circuit_tracer.transcoder.single_layer_transcoder import (
            load_transcoder_set,
        )

        if not config.plt_subfolder_template:
            raise ValueError("PLT requires plt_subfolder_template")
        if isinstance(config.layer_count, int):
            allow_patterns = [
                config.plt_subfolder_template.format(layer=i)
                for i in range(config.layer_count)
            ]
        else:
            allow_patterns = [config.plt_subfolder_template.format(layer="*")]
        local_dir = snapshot_download(
            config.repo_id,
            revision=config.revision,
            cache_dir=config.transcoder_cache_dir,
            allow_patterns=allow_patterns,
        )
        if isinstance(config.layer_count, int):
            paths = {
                i: str(Path(local_dir) / pattern)
                for i, pattern in enumerate(allow_patterns)
            }
            missing_paths = [
                path for path in paths.values() if not Path(path).is_file()
            ]
            if missing_paths:
                raise FileNotFoundError(
                    "Missing PLT layer file(s) after snapshot_download: "
                    + ", ".join(missing_paths[:5])
                    + (" ..." if len(missing_paths) > 5 else "")
                )
        else:
            paths = _infer_plt_layer_paths(
                local_dir=Path(local_dir),
                template=config.plt_subfolder_template,
            )
        transcoders = load_transcoder_set(
            transcoder_paths=paths,
            scan=_checkpoint_identity(config),
            feature_input_hook=config.feature_input_hook,
            feature_output_hook=config.feature_output_hook,
            device=torch.device(DEVICE),
            dtype=DTYPE,
            special_load_fn="gemma-scope-2",
            exact_chunked_provider=True,
            lazy_encoder=config.lazy_encoder,
            lazy_decoder=config.lazy_decoder,
            decoder_chunk_size=config.decoder_chunk_size,
            cross_batch_decoder_cache_bytes=config.cross_batch_decoder_cache_bytes,
            checkpoint_asset_scope=config.checkpoint_asset_scope,
            checkpoint_prefault_budget_bytes=config.checkpoint_prefault_budget_bytes,
        )
    transcoders.exact_chunked_decoder = exact_chunked_decoder

    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name=model_load_source,
        transcoders=transcoders,
        device=torch.device(DEVICE),
        dtype=DTYPE,
        backend="nnsight",
    )
    model._nlp_research_transcoder_metadata = provider_metadata(
        config,
        detected=_detect_transcoder_provider_metadata(
            transcoders,
            _checkpoint_identity(config),
        ),
    )

    if torch.cuda.is_available():
        print(f"  GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model
