#!/bin/bash

# This file is sourced by Granite launch templates after `set -euo pipefail`.
: "${WORKSPACE_ROOT:?WORKSPACE_ROOT is required; snapshot launches fail closed}"
: "${LIB_WORKSPACE_ROOT:?LIB_WORKSPACE_ROOT is required; snapshot launches fail closed}"

EXACT_TRACE_WORKSPACE_MODE="${EXACT_TRACE_WORKSPACE_MODE:-immutable}"

if [[ "$EXACT_TRACE_WORKSPACE_MODE" == "live" ]]; then
    if [[ "${EXACT_TRACE_ALLOW_LIVE_WORKSPACE:-0}" != "1" ]]; then
        echo "Live workspace mode requires EXACT_TRACE_ALLOW_LIVE_WORKSPACE=1" >&2
        return 2
    fi
    if [[ -z "${EXACT_TRACE_LIVE_WORKSPACE_RATIONALE:-}" || -z "${EXACT_TRACE_LIVE_WORKSPACE_RATIONALE//[[:space:]]/}" ]]; then
        echo "Live workspace mode requires EXACT_TRACE_LIVE_WORKSPACE_RATIONALE" >&2
        return 2
    fi
    if [[ ! -d "$WORKSPACE_ROOT" || ! -d "$LIB_WORKSPACE_ROOT" ]]; then
        echo "Live workspace roots must both exist" >&2
        return 2
    fi
    EXACT_TRACE_WORKSPACE_MANIFEST="${EXACT_TRACE_WORKSPACE_MANIFEST:-none (live override)}"
elif [[ "$EXACT_TRACE_WORKSPACE_MODE" == "immutable" ]]; then
    EXACT_TRACE_WORKSPACE_MANIFEST="${EXACT_TRACE_WORKSPACE_MANIFEST:-$(dirname "$WORKSPACE_ROOT")/.exact_trace_bench_snapshot.json}"
    python3 - "$WORKSPACE_ROOT" "$LIB_WORKSPACE_ROOT" "$EXACT_TRACE_WORKSPACE_MANIFEST" <<'PY'
import json
import os
import stat
import sys

workspace = os.path.realpath(sys.argv[1])
library = os.path.realpath(sys.argv[2])
manifest_path = os.path.realpath(sys.argv[3])

if not os.path.isfile(manifest_path):
    raise SystemExit(f"Snapshot manifest does not exist: {manifest_path}")
try:
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Snapshot manifest is invalid: {manifest_path}: {exc}") from exc

if os.path.realpath(str(manifest.get("snapshot_root", ""))) != workspace:
    raise SystemExit("WORKSPACE_ROOT does not match manifest snapshot_root")
snapshots = manifest.get("uv_source_snapshots")
if not isinstance(snapshots, list):
    raise SystemExit("Snapshot manifest uv_source_snapshots must be a list")
editable = next(
    (
        item
        for item in snapshots
        if isinstance(item, dict)
        and item.get("package_name") == "circuit-tracer"
        and item.get("snapshot_path")
    ),
    None,
)
if editable is None:
    raise SystemExit("Snapshot manifest does not record circuit-tracer uv source")
if os.path.realpath(str(editable["snapshot_path"])) != library:
    raise SystemExit("LIB_WORKSPACE_ROOT does not match recorded uv editable snapshot")
if manifest.get("read_only") is not True:
    raise SystemExit("Snapshot manifest read_only must be true")

writable_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
for label, root in (("workspace", workspace), ("library", library)):
    if not os.path.isdir(root):
        raise SystemExit(f"Snapshot {label} root does not exist: {root}")
    if os.stat(root).st_mode & writable_bits:
        raise SystemExit(f"Snapshot {label} root is writable: {root}")

for import_root in (os.path.join(workspace, "src"), workspace, library):
    resolved = os.path.realpath(import_root)
    if os.path.commonpath((resolved, workspace)) != workspace and os.path.commonpath(
        (resolved, library)
    ) != library:
        raise SystemExit(f"Import root is outside snapshot roots: {resolved}")
PY
else
    echo "Unsupported EXACT_TRACE_WORKSPACE_MODE: $EXACT_TRACE_WORKSPACE_MODE" >&2
    return 2
fi

export EXACT_TRACE_WORKSPACE_MODE EXACT_TRACE_WORKSPACE_MANIFEST
echo "Workspace mode: ${EXACT_TRACE_WORKSPACE_MODE}"
echo "Workspace manifest: ${EXACT_TRACE_WORKSPACE_MANIFEST}"
if [[ "$EXACT_TRACE_WORKSPACE_MODE" == "live" ]]; then
    echo "Live workspace rationale: ${EXACT_TRACE_LIVE_WORKSPACE_RATIONALE}"
fi
