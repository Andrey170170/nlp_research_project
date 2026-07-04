from __future__ import annotations

import types
import sys
from types import ModuleType
from pathlib import Path


def test_plt_loader_dispatches_to_transcoder_set(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    def fake_snapshot_download(*args, **kwargs):
        if args:
            kwargs["repo_id"] = args[0]
        calls["snapshot"] = kwargs
        for pattern in kwargs.get("allow_patterns", []):
            path = tmp_path / pattern
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("stub")
        return str(tmp_path)

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = "torch.bfloat16"
    fake_torch.device = lambda value: value
    fake_torch.Tensor = object
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        memory_allocated=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    fake_datasets = ModuleType("datasets")
    fake_datasets.load_dataset = lambda *_args, **_kwargs: []
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    fake_transcoders = types.SimpleNamespace(
        exact_chunked_provider=True,
        exact_chunked_decoder=True,
        architecture="plt",
        weight_format="gemmascope2",
        n_layers=34,
        d_model=3,
        d_transcoder=5,
        dtype="torch.bfloat16",
    )

    def fake_load_transcoder_set(**kwargs):
        calls["load_transcoder_set"] = kwargs
        return fake_transcoders

    class FakeReplacementModel:
        @staticmethod
        def from_pretrained_and_transcoders(**kwargs):
            calls["replacement_model"] = kwargs
            return types.SimpleNamespace(transcoders=kwargs["transcoders"])

    fake_circuit_tracer = ModuleType("circuit_tracer")
    fake_circuit_tracer.ReplacementModel = FakeReplacementModel
    fake_circuit_tracer.attribute = object()
    fake_transcoder_pkg = ModuleType("circuit_tracer.transcoder")
    fake_slt = ModuleType("circuit_tracer.transcoder.single_layer_transcoder")
    fake_slt.load_transcoder_set = fake_load_transcoder_set
    fake_provider = ModuleType("circuit_tracer.transcoder.provider")
    fake_provider.get_transcoder_capabilities = lambda _obj: types.SimpleNamespace(
        architecture="plt",
        checkpoint_format="gemmascope2",
        supports_exact_chunked_provider=True,
    )
    fake_provider.provider_fingerprint = lambda _obj, **_kwargs: {
        "architecture": "plt",
        "supports_exact_chunked_provider": True,
    }
    monkeypatch.setitem(sys.modules, "circuit_tracer", fake_circuit_tracer)
    monkeypatch.setitem(sys.modules, "circuit_tracer.transcoder", fake_transcoder_pkg)
    monkeypatch.setitem(
        sys.modules,
        "circuit_tracer.transcoder.single_layer_transcoder",
        fake_slt,
    )
    monkeypatch.setitem(
        sys.modules,
        "circuit_tracer.transcoder.provider",
        fake_provider,
    )
    monkeypatch.delitem(sys.modules, "trace_pipeline", raising=False)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))

    import trace_pipeline

    model = trace_pipeline.load_model(
        transcoder_provider_family="gemmascope2-plt-4b-small-affine",
        decoder_chunk_size=128,
        cross_batch_decoder_cache_bytes=0,
    )

    snapshot = calls["snapshot"]
    assert isinstance(snapshot, dict)
    assert snapshot["repo_id"] == "google/gemma-scope-2-4b-it"
    assert snapshot["allow_patterns"][0].endswith(
        "transcoder_all/layer_0_width_262k_l0_small_affine/params.safetensors"
    )
    assert snapshot["allow_patterns"][-1].endswith(
        "transcoder_all/layer_33_width_262k_l0_small_affine/params.safetensors"
    )

    load_kwargs = calls["load_transcoder_set"]
    assert isinstance(load_kwargs, dict)
    assert load_kwargs["exact_chunked_provider"] is True
    assert load_kwargs["lazy_encoder"] is True
    assert load_kwargs["lazy_decoder"] is True
    assert load_kwargs["decoder_chunk_size"] == 128
    assert load_kwargs["cross_batch_decoder_cache_bytes"] == 0
    assert load_kwargs["feature_input_hook"] == "mlp.hook_in"
    assert load_kwargs["feature_output_hook"] == "hook_mlp_out"
    assert sorted(load_kwargs["transcoder_paths"]) == list(range(34))

    replacement_kwargs = calls["replacement_model"]
    assert isinstance(replacement_kwargs, dict)
    assert replacement_kwargs["model_name"] == "google/gemma-3-4b-it"
    metadata = trace_pipeline.get_model_transcoder_metadata(model)
    assert metadata is not None
    assert metadata["requested"]["transcoder_architecture"] == "plt"
    assert metadata["detected"]["capabilities"]["architecture"] == "plt"


def test_clt_loader_dispatches_to_native_clt_loader(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}
    clt_dir = tmp_path / "clt" / "width_262k_l0_medium_affine"
    clt_dir.mkdir(parents=True)
    for layer in range(2):
        (clt_dir / f"params_layer_{layer}.safetensors").write_text("stub")

    def fake_snapshot_download(*args, **kwargs):
        if args:
            kwargs["repo_id"] = args[0]
        calls["snapshot"] = kwargs
        return str(tmp_path)

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = "torch.bfloat16"
    fake_torch.device = lambda value: value
    fake_torch.Tensor = object
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    fake_datasets = ModuleType("datasets")
    fake_datasets.load_dataset = lambda *_args, **_kwargs: []
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    fake_transcoders = types.SimpleNamespace(
        exact_chunked_provider=True,
        exact_chunked_decoder=True,
    )

    def fake_load_native(**kwargs):
        calls["load_gemma_scope_2_clt_native"] = kwargs
        return fake_transcoders

    def fake_load_clt(**kwargs):
        calls["load_gemma_scope_2_clt"] = kwargs
        return fake_transcoders

    class FakeReplacementModel:
        @staticmethod
        def from_pretrained_and_transcoders(**kwargs):
            calls["replacement_model"] = kwargs
            return types.SimpleNamespace(transcoders=kwargs["transcoders"])

    fake_circuit_tracer = ModuleType("circuit_tracer")
    fake_circuit_tracer.ReplacementModel = FakeReplacementModel
    fake_circuit_tracer.attribute = object()
    fake_clt = ModuleType("circuit_tracer.transcoder.cross_layer_transcoder")
    fake_clt.load_gemma_scope_2_clt = fake_load_clt
    monkeypatch.setitem(sys.modules, "circuit_tracer", fake_circuit_tracer)
    monkeypatch.setitem(
        sys.modules,
        "circuit_tracer.transcoder.cross_layer_transcoder",
        fake_clt,
    )
    monkeypatch.delitem(sys.modules, "trace_pipeline", raising=False)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))

    import trace_pipeline

    trace_pipeline.load_gemma_scope_2_clt_native(paths={0: ""})
    assert calls["load_gemma_scope_2_clt"]["paths"] == {0: ""}
    monkeypatch.setattr(
        trace_pipeline, "load_gemma_scope_2_clt_native", fake_load_native
    )
    model = trace_pipeline.load_model(clt_subfolder="clt/width_262k_l0_medium_affine")

    snapshot = calls["snapshot"]
    assert isinstance(snapshot, dict)
    assert snapshot["allow_patterns"] == [
        "clt/width_262k_l0_medium_affine/params_layer_*.safetensors"
    ]
    assert "load_gemma_scope_2_clt_native" in calls
    native_kwargs = calls["load_gemma_scope_2_clt_native"]
    assert isinstance(native_kwargs, dict)
    assert native_kwargs["paths"] == {
        0: str(clt_dir / "params_layer_0.safetensors"),
        1: str(clt_dir / "params_layer_1.safetensors"),
    }
    assert native_kwargs["cross_batch_decoder_cache_bytes"] == 8589934592
    assert native_kwargs["scan"] == (
        "google/gemma-scope-2-1b-it@main:clt/width_262k_l0_medium_affine"
    )
    assert trace_pipeline.get_model_transcoder_metadata(model) is not None
