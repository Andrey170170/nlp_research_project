from __future__ import annotations

from nlp_research_project.exact_trace_bench.transcoder_config import (
    TranscoderLoadConfig,
    resolve_transcoder_load_config,
    transcoder_config_to_json,
)


def test_default_transcoder_config_is_current_clt() -> None:
    config = resolve_transcoder_load_config()
    payload = transcoder_config_to_json(config)
    assert payload["transcoder_architecture"] == "clt"
    assert payload["transcoder_provider_family"] == "gemmascope2-clt-1b-medium-affine"
    assert payload["model_name"] == "google/gemma-3-1b-it"
    assert payload["repo_id"] == "google/gemma-scope-2-1b-it"
    assert payload["clt_subfolder"] == "clt/width_262k_l0_medium_affine"
    assert payload["feature_input_hook"] == "mlp.hook_in"
    assert payload["feature_output_hook"] == "hook_mlp_out"
    assert payload["decoder_chunk_size"] == 256
    assert payload["cross_batch_decoder_cache_bytes"] == 8589934592


def test_plt_architecture_without_family_uses_safe_explicit_default() -> None:
    config = resolve_transcoder_load_config({"transcoder_architecture": "plt"})
    payload = transcoder_config_to_json(config)
    assert payload["transcoder_provider_family"] == "gemmascope2-plt-1b-big-affine"
    assert payload["layer_count"] == 26
    assert payload["feature_input_hook"] == "mlp.hook_in"
    assert payload["cross_batch_decoder_cache_bytes"] == 0
    assert payload["lazy_encoder"] is True
    assert payload["lazy_decoder"] is True


def test_plt_4b_provider_selects_4b_model_and_repo() -> None:
    config = resolve_transcoder_load_config(
        {"transcoder_provider_family": "gemmascope2-plt-4b-small-affine"}
    )
    payload = transcoder_config_to_json(config)
    assert payload["transcoder_architecture"] == "plt"
    assert payload["model_name"] == "google/gemma-3-4b-it"
    assert payload["repo_id"] == "google/gemma-scope-2-4b-it"
    assert payload["layer_count"] == 34
    assert payload["feature_input_hook"] == "mlp.hook_in"


def test_architecture_family_conflicts_are_rejected() -> None:
    try:
        resolve_transcoder_load_config(
            transcoder_architecture="clt",
            transcoder_provider_family="gemmascope2-plt-1b-big-affine",
        )
    except ValueError as exc:
        assert "conflicts" in str(exc)
    else:  # pragma: no cover - explicit assertion path
        raise AssertionError(
            "expected conflicting architecture/provider family to fail"
        )


def test_baked_clt_defaults_do_not_override_plt_architecture_request() -> None:
    baked_defaults = transcoder_config_to_json(TranscoderLoadConfig())
    config = resolve_transcoder_load_config(
        {**baked_defaults, "transcoder_architecture": "plt"}
    )
    payload = transcoder_config_to_json(config)
    assert payload["transcoder_provider_family"] == "gemmascope2-plt-1b-big-affine"
    assert payload["cross_batch_decoder_cache_bytes"] == 0
    assert payload["layer_count"] == 26
    assert payload["feature_input_hook"] == "mlp.hook_in"


def test_baked_clt_defaults_do_not_override_plt_provider_family() -> None:
    baked_defaults = transcoder_config_to_json(TranscoderLoadConfig())
    config = resolve_transcoder_load_config(
        {
            **baked_defaults,
            "transcoder_provider_family": "gemmascope2-plt-4b-small-affine",
        }
    )
    payload = transcoder_config_to_json(config)
    assert payload["transcoder_architecture"] == "plt"
    assert payload["model_name"] == "google/gemma-3-4b-it"
    assert payload["repo_id"] == "google/gemma-scope-2-4b-it"
    assert payload["cross_batch_decoder_cache_bytes"] == 0
    assert payload["feature_input_hook"] == "mlp.hook_in"
