from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Mapping

Architecture = Literal["clt", "plt"]


@dataclass(frozen=True)
class TranscoderLoadConfig:
    model_name: str = "google/gemma-3-1b-it"
    transcoder_architecture: Architecture = "clt"
    transcoder_provider_family: str = "gemmascope2-clt-1b-medium-affine"
    repo_id: str = "google/gemma-scope-2-1b-it"
    revision: str | None = None
    clt_subfolder: str | None = "clt/width_262k_l0_medium_affine"
    plt_subfolder_template: str | None = None
    layer_count: int | Literal["infer"] = "infer"
    feature_input_hook: str = "mlp.hook_in"
    feature_output_hook: str = "hook_mlp_out"
    lazy_encoder: bool = True
    lazy_decoder: bool = True
    decoder_chunk_size: int = 256
    cross_batch_decoder_cache_bytes: int = 8589934592
    transcoder_cache_dir: str | None = None


_PLT_TEMPLATE = (
    "transcoder_all/layer_{layer}_width_262k_l0_{variant}/params.safetensors"
)
_GEMMASCOPE2_FEATURE_INPUT_HOOK = "mlp.hook_in"


def _plt(
    provider: str,
    *,
    model_name: str,
    repo_id: str,
    layer_count: int,
    variant: str,
) -> TranscoderLoadConfig:
    return TranscoderLoadConfig(
        model_name=model_name,
        transcoder_architecture="plt",
        transcoder_provider_family=provider,
        repo_id=repo_id,
        clt_subfolder=None,
        plt_subfolder_template=_PLT_TEMPLATE.format(layer="{layer}", variant=variant),
        layer_count=layer_count,
        feature_input_hook=_GEMMASCOPE2_FEATURE_INPUT_HOOK,
        cross_batch_decoder_cache_bytes=0,
    )


PROVIDER_PRESETS: dict[str, TranscoderLoadConfig] = {
    "gemmascope2-clt-1b-medium-affine": TranscoderLoadConfig(),
}
for _size, _model, _repo, _layers in (
    ("1b", "google/gemma-3-1b-it", "google/gemma-scope-2-1b-it", 26),
    ("4b", "google/gemma-3-4b-it", "google/gemma-scope-2-4b-it", 34),
    ("12b", "google/gemma-3-12b-it", "google/gemma-scope-2-12b-it", 48),
):
    for _variant in ("small", "small_affine", "big", "big_affine"):
        PROVIDER_PRESETS[f"gemmascope2-plt-{_size}-{_variant.replace('_', '-')}"] = (
            _plt(
                f"gemmascope2-plt-{_size}-{_variant.replace('_', '-')}",
                model_name=_model,
                repo_id=_repo,
                layer_count=_layers,
                variant=_variant,
            )
        )


PUBLIC_TRANSCODER_KNOB_KEYS = tuple(TranscoderLoadConfig.__dataclass_fields__.keys())


def _provider_family_architecture_hint(family: str | None) -> Architecture | None:
    if family is None:
        return None
    tokens = family.split("-")
    if "plt" in tokens:
        return "plt"
    if "clt" in tokens:
        return "clt"
    return None


def resolve_transcoder_load_config(
    mapping: Mapping[str, Any] | None = None,
    *,
    preserve_default_values: bool = False,
    **overrides: Any,
) -> TranscoderLoadConfig:
    default = TranscoderLoadConfig()
    mapping_keys = {
        k
        for k, v in dict(mapping or {}).items()
        if v is not None
        and (
            preserve_default_values
            or k not in PUBLIC_TRANSCODER_KNOB_KEYS
            or v != getattr(default, k)
        )
    }
    explicit_override_keys = {k for k, v in overrides.items() if v is not None}
    explicit_keys = mapping_keys | explicit_override_keys
    values = {
        **dict(mapping or {}),
        **{k: v for k, v in overrides.items() if v is not None},
    }
    arch = values.get("transcoder_architecture")
    family = values.get("transcoder_provider_family")
    if arch == "plt" and (
        family is None
        or (
            family == default.transcoder_provider_family
            and "transcoder_provider_family" not in explicit_keys
        )
    ):
        family = "gemmascope2-plt-1b-big-affine"
    preset = PROVIDER_PRESETS.get(family or "gemmascope2-clt-1b-medium-affine")
    family_architecture = (
        preset.transcoder_architecture
        if preset is not None
        else _provider_family_architecture_hint(family)
    )
    if (
        arch is not None
        and "transcoder_architecture" in explicit_keys
        and family_architecture is not None
        and arch != family_architecture
    ):
        raise ValueError(
            "transcoder_architecture conflicts with transcoder_provider_family "
            f"{family!r}"
        )
    if preset is None:
        preset = TranscoderLoadConfig(
            transcoder_architecture=family_architecture
            or default.transcoder_architecture,
            transcoder_provider_family=str(family),
        )
    data = asdict(preset)
    for key in PUBLIC_TRANSCODER_KNOB_KEYS:
        if key in values and values[key] is not None:
            if (
                preset != default
                and not preserve_default_values
                and key not in explicit_keys
                and values[key] == getattr(default, key)
            ):
                continue
            data[key] = values[key]
    if isinstance(data["layer_count"], str) and data["layer_count"] != "infer":
        data["layer_count"] = int(data["layer_count"])
    if data["transcoder_architecture"] not in {"clt", "plt"}:
        raise ValueError("transcoder_architecture must be 'clt' or 'plt'")
    if data["transcoder_architecture"] == "plt":
        data["lazy_encoder"] = True
        data["lazy_decoder"] = True
    return replace(TranscoderLoadConfig(), **data)


def transcoder_config_to_json(config: TranscoderLoadConfig) -> dict[str, Any]:
    return asdict(config)


def provider_metadata(
    config: TranscoderLoadConfig, *, detected: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    payload = {"requested": transcoder_config_to_json(config)}
    if detected:
        payload["detected"] = dict(detected)
    return payload
