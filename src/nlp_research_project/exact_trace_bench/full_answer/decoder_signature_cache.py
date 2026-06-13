from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..io_utils import ensure_dir, write_json

GEMMASCOPE2_REPO_ID = "google/gemma-scope-2-1b-it"
GEMMASCOPE2_CLT_SUBFOLDER = "clt/width_262k_l0_medium_affine"
DEFAULT_SIGNATURE_CHUNK_SIZE = 512
DEFAULT_CACHE_FORMAT_VERSION = 1

StorageDType = Literal["float32", "float16"]


@dataclass(frozen=True)
class GemmaScope2DecoderShape:
    n_layers: int
    features_per_layer: int
    d_model: int
    source_dtype: str

    @property
    def flattened_dims_by_layer(self) -> dict[str, int]:
        return {
            str(layer): (self.n_layers - layer) * self.d_model
            for layer in range(self.n_layers)
        }


def _layer_file_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def discover_gemmascope2_layer_paths(source_dir: Path) -> dict[int, Path]:
    """Return contiguous GemmaScope-2 params_layer_*.safetensors paths."""

    paths = sorted(source_dir.glob("params_layer_*.safetensors"), key=_layer_file_index)
    if not paths:
        raise FileNotFoundError(
            f"no params_layer_*.safetensors files found under {source_dir}"
        )
    indexed = {_layer_file_index(path): path for path in paths}
    expected = list(range(len(indexed)))
    actual = sorted(indexed)
    if actual != expected:
        raise ValueError(
            "GemmaScope-2 layer files must be indexed contiguously from 0 "
            f"(got {actual[:5]}...{actual[-5:] if len(actual) > 5 else actual})"
        )
    return indexed


def resolve_gemmascope2_source_dir(
    *,
    source_dir: Path | None = None,
    repo_id: str = GEMMASCOPE2_REPO_ID,
    subfolder: str = GEMMASCOPE2_CLT_SUBFOLDER,
) -> Path:
    """Resolve GemmaScope-2 CLT layer-file directory.

    Passing ``source_dir`` avoids any Hugging Face download/check. Without it, this
    uses ``snapshot_download`` with a narrow allow-list; run that path in SLURM if
    the files are not already cached.
    """

    if source_dir is not None:
        return source_dir

    from huggingface_hub import snapshot_download

    local_root = snapshot_download(
        repo_id,
        allow_patterns=[f"{subfolder}/params_layer_*.safetensors"],
    )
    return Path(local_root) / subfolder


def infer_gemmascope2_decoder_shape(paths: dict[int, Path]) -> GemmaScope2DecoderShape:
    from safetensors import safe_open

    first = paths[0]
    with safe_open(str(first), framework="pt", device="cpu") as f:
        if "w_dec" not in f.keys():
            raise ValueError(f"{first} does not contain a 'w_dec' tensor")
        features_per_layer, n_layers_in_tensor, d_model = f.get_slice(
            "w_dec"
        ).get_shape()
        source_dtype = str(f.get_slice("w_dec").get_dtype())

    n_layers = len(paths)
    if n_layers_in_tensor != n_layers:
        raise ValueError(
            "GemmaScope-2 w_dec output-layer dimension does not match file count: "
            f"w_dec has {n_layers_in_tensor}, paths have {n_layers}"
        )

    for layer, path in paths.items():
        with safe_open(str(path), framework="pt", device="cpu") as f:
            shape = f.get_slice("w_dec").get_shape()
        if shape != [features_per_layer, n_layers, d_model]:
            raise ValueError(
                f"unexpected w_dec shape for layer {layer} at {path}: {shape}; "
                f"expected {[features_per_layer, n_layers, d_model]}"
            )

    return GemmaScope2DecoderShape(
        n_layers=int(n_layers),
        features_per_layer=int(features_per_layer),
        d_model=int(d_model),
        source_dtype=source_dtype,
    )


def _storage_np_dtype(dtype: StorageDType) -> np.dtype[Any]:
    if dtype == "float32":
        return np.dtype(np.float32)
    if dtype == "float16":
        return np.dtype(np.float16)
    raise ValueError(f"unsupported decoder signature storage dtype: {dtype}")


def _chunk_path(layer_dir: Path, start: int, stop: int) -> Path:
    return layer_dir / f"chunk_{start:06d}_{stop:06d}.npy"


def _source_file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    resolved = path.resolve()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _build_metadata(
    *,
    source_dir: Path,
    paths: dict[int, Path],
    shape: GemmaScope2DecoderShape,
    output_dir: Path,
    chunk_size: int,
    storage_dtype: StorageDType,
    repo_id: str,
    subfolder: str,
    complete: bool,
) -> dict[str, Any]:
    chunks_by_layer: dict[str, list[dict[str, Any]]] = {}
    for layer in range(shape.n_layers):
        rows = []
        for start in range(0, shape.features_per_layer, chunk_size):
            stop = min(start + chunk_size, shape.features_per_layer)
            rows.append(
                {
                    "start": start,
                    "stop": stop,
                    "path": f"layer_{layer:02d}/{_chunk_path(Path(), start, stop).name}",
                    "shape": [stop - start, (shape.n_layers - layer) * shape.d_model],
                }
            )
        chunks_by_layer[str(layer)] = rows

    return {
        "format_version": DEFAULT_CACHE_FORMAT_VERSION,
        "signature_kind": "gemmascope2_clt_flattened_downstream_decoder",
        "repo_id": repo_id,
        "subfolder": subfolder,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "n_layers": shape.n_layers,
        "features_per_layer": shape.features_per_layer,
        "d_model": shape.d_model,
        "flattened_dims_by_layer": shape.flattened_dims_by_layer,
        "source_weight_dtype": shape.source_dtype,
        "storage_dtype": storage_dtype,
        "normalized": True,
        "chunk_size": int(chunk_size),
        "source_files": {
            str(layer): _source_file_record(path)
            for layer, path in sorted(paths.items())
        },
        "chunks_by_layer": chunks_by_layer,
    }


def build_decoder_signature_cache(
    *,
    output_dir: Path,
    source_dir: Path | None = None,
    repo_id: str = GEMMASCOPE2_REPO_ID,
    subfolder: str = GEMMASCOPE2_CLT_SUBFOLDER,
    chunk_size: int = DEFAULT_SIGNATURE_CHUNK_SIZE,
    storage_dtype: StorageDType = "float32",
    overwrite: bool = False,
    layers: list[int] | None = None,
) -> dict[str, Any]:
    """Build normalized GemmaScope-2 CLT decoder-signature cache.

    For source layer ``L`` and feature ``i``, the cached row is the unit-normalized
    flattened downstream decoder block ``w_dec[i, L:, :]``. This mirrors the
    sibling ``load_gemma_scope_2_clt`` semantics while keeping later metric runs
    independent of model/transcoder loading.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    from safetensors import safe_open
    import torch

    np_dtype = _storage_np_dtype(storage_dtype)
    resolved_source_dir = resolve_gemmascope2_source_dir(
        source_dir=source_dir, repo_id=repo_id, subfolder=subfolder
    )
    paths = discover_gemmascope2_layer_paths(resolved_source_dir)
    shape = infer_gemmascope2_decoder_shape(paths)
    selected_layers = (
        list(range(shape.n_layers)) if layers is None else sorted(set(layers))
    )
    invalid_layers = [
        layer for layer in selected_layers if layer < 0 or layer >= shape.n_layers
    ]
    if invalid_layers:
        raise ValueError(f"layer ids out of range: {invalid_layers}")

    ensure_dir(output_dir)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not overwrite:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_existing = {
            "format_version": DEFAULT_CACHE_FORMAT_VERSION,
            "signature_kind": "gemmascope2_clt_flattened_downstream_decoder",
            "repo_id": repo_id,
            "subfolder": subfolder,
            "n_layers": shape.n_layers,
            "features_per_layer": shape.features_per_layer,
            "d_model": shape.d_model,
            "storage_dtype": storage_dtype,
            "chunk_size": int(chunk_size),
            "normalized": True,
        }
        mismatches = {
            key: {"existing": existing.get(key), "requested": value}
            for key, value in expected_existing.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "existing decoder signature cache metadata is incompatible with "
                f"this build request at {output_dir}: {mismatches}"
            )
        if existing.get("complete") and layers is None:
            raise FileExistsError(
                f"complete decoder signature cache already exists at {output_dir}; "
                "pass overwrite=True/--overwrite to rebuild"
            )

    metadata = _build_metadata(
        source_dir=resolved_source_dir,
        paths=paths,
        shape=shape,
        output_dir=output_dir,
        chunk_size=chunk_size,
        storage_dtype=storage_dtype,
        repo_id=repo_id,
        subfolder=subfolder,
        complete=False,
    )
    write_json(metadata_path, metadata)

    written_chunks = 0
    skipped_chunks = 0
    for layer in selected_layers:
        layer_dir = output_dir / f"layer_{layer:02d}"
        ensure_dir(layer_dir)
        with safe_open(str(paths[layer]), framework="pt", device="cpu") as f:
            w_dec = f.get_slice("w_dec")
            for start in range(0, shape.features_per_layer, chunk_size):
                stop = min(start + chunk_size, shape.features_per_layer)
                out_path = _chunk_path(layer_dir, start, stop)
                if out_path.exists() and not overwrite:
                    skipped_chunks += 1
                    continue
                block = w_dec[start:stop][:, layer:, :].to(dtype=torch.float32)
                flat = block.reshape(block.shape[0], -1)
                norms = torch.linalg.vector_norm(flat, dim=1, keepdim=True)
                flat = torch.where(
                    norms > 0.0, flat / torch.clamp(norms, min=1e-30), flat
                )
                array = flat.cpu().numpy().astype(np_dtype, copy=False)
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                with tmp_path.open("wb") as handle:
                    np.save(handle, array)
                tmp_path.replace(out_path)
                written_chunks += 1

    complete = layers is None
    if complete:
        missing: list[str] = []
        for layer_chunks in metadata["chunks_by_layer"].values():
            for chunk in layer_chunks:
                if not (output_dir / chunk["path"]).exists():
                    missing.append(chunk["path"])
        complete = not missing
        metadata["missing_chunks"] = missing[:100]
        metadata["missing_chunk_count"] = len(missing)

    metadata["complete"] = complete
    metadata["written_chunk_count"] = written_chunks
    metadata["skipped_existing_chunk_count"] = skipped_chunks
    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(metadata_path, metadata)
    return metadata


class DecoderSignatureStore:
    """Lazy reader for chunked normalized decoder signatures."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        max_resident_chunks: int = 16,
        cosine_device: str = "cpu",
    ) -> None:
        self.cache_dir = cache_dir
        metadata_path = cache_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"missing decoder cache metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("format_version") != DEFAULT_CACHE_FORMAT_VERSION:
            raise ValueError(
                "unsupported decoder signature cache format version: "
                f"{self.metadata.get('format_version')}"
            )
        if not self.metadata.get("normalized"):
            raise ValueError("decoder signature cache must contain normalized rows")
        self.n_layers = int(self.metadata["n_layers"])
        self.features_per_layer = int(self.metadata["features_per_layer"])
        self.d_model = int(self.metadata["d_model"])
        self.chunk_size = int(self.metadata["chunk_size"])
        self.max_resident_chunks = max(0, int(max_resident_chunks))
        self.cosine_device = str(cosine_device)
        self._chunk_cache: OrderedDict[
            tuple[int, int], np.ndarray[Any, np.dtype[Any]]
        ] = OrderedDict()

    def flattened_dim(self, layer: int) -> int:
        self._validate_layer(layer)
        return int(self.metadata["flattened_dims_by_layer"][str(layer)])

    def _validate_layer(self, layer: int) -> None:
        if layer < 0 or layer >= self.n_layers:
            raise ValueError(f"layer {layer} is out of range [0, {self.n_layers})")

    def _validate_feature_ids(
        self, feature_ids: np.ndarray[Any, np.dtype[Any]]
    ) -> None:
        if feature_ids.ndim != 1:
            raise ValueError("feature_ids must be one-dimensional")
        if feature_ids.size and (
            int(feature_ids.min()) < 0
            or int(feature_ids.max()) >= self.features_per_layer
        ):
            raise ValueError(
                "feature id out of range for decoder cache: "
                f"min={int(feature_ids.min())}, max={int(feature_ids.max())}, "
                f"features_per_layer={self.features_per_layer}"
            )

    def _load_chunk(self, layer: int, chunk_id: int) -> np.ndarray[Any, np.dtype[Any]]:
        key = (layer, chunk_id)
        cached = self._chunk_cache.get(key)
        if cached is not None:
            self._chunk_cache.move_to_end(key)
            return cached

        start = chunk_id * self.chunk_size
        stop = min(start + self.chunk_size, self.features_per_layer)
        path = _chunk_path(self.cache_dir / f"layer_{layer:02d}", start, stop)
        if not path.exists():
            raise FileNotFoundError(f"missing decoder signature chunk: {path}")
        array = np.load(path, mmap_mode="r")
        if self.max_resident_chunks > 0:
            self._chunk_cache[key] = array
            while len(self._chunk_cache) > self.max_resident_chunks:
                self._chunk_cache.popitem(last=False)
        return array

    def get_rows(
        self, layer: int, feature_ids: list[int] | np.ndarray[Any, Any]
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        self._validate_layer(layer)
        ids = np.asarray(feature_ids, dtype=np.int64).reshape(-1)
        self._validate_feature_ids(ids)
        if ids.size == 0:
            return np.empty((0, self.flattened_dim(layer)), dtype=np.float32)

        out = np.empty((ids.size, self.flattened_dim(layer)), dtype=np.float32)
        for chunk_id in np.unique(ids // self.chunk_size):
            chunk_id_int = int(chunk_id)
            mask = ids // self.chunk_size == chunk_id_int
            rows = np.nonzero(mask)[0]
            chunk = self._load_chunk(layer, chunk_id_int)
            local_ids = ids[rows] - chunk_id_int * self.chunk_size
            out[rows] = np.asarray(chunk[local_ids], dtype=np.float32)
        return out

    def cosine_matrix(
        self,
        layer: int,
        left_feature_ids: list[int] | np.ndarray[Any, Any],
        right_feature_ids: list[int] | np.ndarray[Any, Any],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        left = self.get_rows(layer, left_feature_ids)
        right = self.get_rows(layer, right_feature_ids)
        if left.shape[0] == 0 or right.shape[0] == 0:
            return np.empty((left.shape[0], right.shape[0]), dtype=np.float32)
        if self.cosine_device != "cpu":
            import torch

            device = torch.device(self.cosine_device)
            left_tensor = torch.as_tensor(left, dtype=torch.float32, device=device)
            right_tensor = torch.as_tensor(right, dtype=torch.float32, device=device)
            result = left_tensor @ right_tensor.T
            return (
                result.detach().to(device="cpu").numpy().astype(np.float32, copy=False)
            )
        return np.asarray(left @ right.T, dtype=np.float32)
