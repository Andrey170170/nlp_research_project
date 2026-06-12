from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nlp_research_project.exact_trace_bench.full_answer.decoder_signature_cache import (
    DecoderSignatureStore,
    build_decoder_signature_cache,
    discover_gemmascope2_layer_paths,
    infer_gemmascope2_decoder_shape,
)

torch = pytest.importorskip("torch")
save_file = pytest.importorskip("safetensors.torch").save_file


def _write_fake_gemmascope2_source(source_dir: Path) -> dict[int, torch.Tensor]:
    source_dir.mkdir(parents=True, exist_ok=True)
    n_layers = 3
    n_features = 4
    d_model = 2
    tensors: dict[int, torch.Tensor] = {}
    for layer in range(n_layers):
        w_dec = torch.zeros(n_features, n_layers, d_model, dtype=torch.float32)
        for feature in range(n_features):
            for out_layer in range(n_layers):
                w_dec[feature, out_layer, 0] = float(
                    (layer + 1) * 100 + feature * 10 + out_layer
                )
                w_dec[feature, out_layer, 1] = float(
                    (layer + 1) * 100 + feature * 10 + out_layer + 1
                )
        if layer == 0:
            w_dec[0] = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            w_dec[1] = torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
            w_dec[2] = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        tensors[layer] = w_dec
        save_file({"w_dec": w_dec}, source_dir / f"params_layer_{layer}.safetensors")
    return tensors


def test_discovers_and_infers_gemmascope2_shape(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _write_fake_gemmascope2_source(source_dir)

    paths = discover_gemmascope2_layer_paths(source_dir)
    shape = infer_gemmascope2_decoder_shape(paths)

    assert sorted(paths) == [0, 1, 2]
    assert shape.n_layers == 3
    assert shape.features_per_layer == 4
    assert shape.d_model == 2
    assert shape.flattened_dims_by_layer == {"0": 6, "1": 4, "2": 2}


def test_build_decoder_signature_cache_and_store_rows(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    tensors = _write_fake_gemmascope2_source(source_dir)
    output_dir = tmp_path / "decoder_cache"

    metadata = build_decoder_signature_cache(
        source_dir=source_dir,
        output_dir=output_dir,
        chunk_size=2,
        storage_dtype="float32",
    )

    assert metadata["complete"] is True
    assert metadata["n_layers"] == 3
    assert metadata["chunk_size"] == 2
    assert (output_dir / "layer_00" / "chunk_000000_000002.npy").exists()
    assert (output_dir / "metadata.json").exists()

    store = DecoderSignatureStore(output_dir, max_resident_chunks=2)
    rows = store.get_rows(1, [3, 0, 3])
    assert rows.shape == (3, 4)

    raw = tensors[1][3, 1:, :].reshape(-1).numpy()
    expected = raw / np.linalg.norm(raw)
    assert rows[0] == pytest.approx(expected)
    assert rows[2] == pytest.approx(expected)


def test_decoder_signature_store_cosine_matrix(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _write_fake_gemmascope2_source(source_dir)
    output_dir = tmp_path / "decoder_cache"
    build_decoder_signature_cache(
        source_dir=source_dir,
        output_dir=output_dir,
        chunk_size=2,
        storage_dtype="float32",
    )

    store = DecoderSignatureStore(output_dir)
    cosine = store.cosine_matrix(0, [0, 1], [2, 1])

    assert cosine.shape == (2, 2)
    assert cosine[0, 0] == pytest.approx(1.0)
    assert cosine[0, 1] == pytest.approx(0.0)
    assert cosine[1, 0] == pytest.approx(0.0)
    assert cosine[1, 1] == pytest.approx(1.0)


def test_decoder_signature_cache_rejects_overwrite_by_default(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    _write_fake_gemmascope2_source(source_dir)
    output_dir = tmp_path / "decoder_cache"
    build_decoder_signature_cache(
        source_dir=source_dir, output_dir=output_dir, chunk_size=2
    )

    with pytest.raises(FileExistsError):
        build_decoder_signature_cache(
            source_dir=source_dir,
            output_dir=output_dir,
            chunk_size=2,
        )


def test_partial_layer_build_is_marked_incomplete(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _write_fake_gemmascope2_source(source_dir)
    output_dir = tmp_path / "decoder_cache"

    metadata = build_decoder_signature_cache(
        source_dir=source_dir,
        output_dir=output_dir,
        chunk_size=2,
        layers=[1],
    )

    stored = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["complete"] is False
    assert stored["complete"] is False
    assert (output_dir / "layer_01" / "chunk_000000_000002.npy").exists()
    assert not (output_dir / "layer_00" / "chunk_000000_000002.npy").exists()
