#!/usr/bin/env bash
set -euo pipefail

# Transfer the old project scratch tree to the new PAS2836 allocation and make
# copied artifacts operational under the new path/account.  Run from an OSC
# shell whose group membership includes PAS2836.

SRC="${SRC:-/fs/scratch/PAS3272/kopanev.1/}"
DST="${DST:-/fs/scratch/PAS2836/kopanev.1/}"
LOG_DIR="${LOG_DIR:-$DST/transfer_logs}"
SKIP_RSYNC="${SKIP_RSYNC:-0}"
SKIP_WORKSPACE_SNAPSHOTS="${SKIP_WORKSPACE_SNAPSHOTS:-1}"
export SRC DST LOG_DIR SKIP_WORKSPACE_SNAPSHOTS

mkdir -p "$DST" "$LOG_DIR"

echo "Source: $SRC"
echo "Destination: $DST"
echo "Start: $(date --iso-8601=seconds)"

if [[ "$SKIP_RSYNC" == "1" ]]; then
  echo "Skipping rsync because SKIP_RSYNC=1"
else
  # --no-times intentionally refreshes copied file mtimes to the transfer time so
  # the new scratch copy is not immediately eligible for age-based purge.  We do
  # not preserve old owner/group metadata; the new allocation should own the copy.
  # --size-only makes reruns practical after the destination mtimes are refreshed.
  rsync -rlpH --size-only --no-times --partial --human-readable --info=progress2 \
    "$SRC" "$DST" \
    2>&1 | tee "$LOG_DIR/rsync_$(date +%Y%m%d_%H%M%S).log"
fi

echo "Rewriting copied text references from PAS3272 to PAS2836..."
python - <<'PY'
from __future__ import annotations

from pathlib import Path
import os

dst = Path(os.environ.get("DST", "/fs/scratch/PAS2836/kopanev.1/"))
skip_workspace_snapshots = os.environ.get("SKIP_WORKSPACE_SNAPSHOTS", "1") == "1"
suffixes = {
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".sh",
    ".sbatch",
    ".py",
    ".csv",
    ".yaml",
    ".yml",
    ".toml",
}
changed = 0
skipped = 0
for path in dst.rglob("*"):
    if skip_workspace_snapshots and "workspace_snapshots" in path.parts:
        continue
    if not path.is_file() or path.suffix.lower() not in suffixes:
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        skipped += 1
        continue
    new = text.replace("/fs/scratch/PAS3272/kopanev.1", "/fs/scratch/PAS2836/kopanev.1")
    new = new.replace("PAS3272", "PAS2836")
    if new != text:
        try:
            path.chmod(path.stat().st_mode | 0o200)
            path.write_text(new, encoding="utf-8")
            changed += 1
        except OSError:
            skipped += 1
print(f"rewrote {changed} copied text files; skipped {skipped}")
PY

echo "Refreshing mtimes under destination tree..."
find "$DST" -exec touch -h {} +

echo "Destination size:"
du -sh "$DST"
echo "End: $(date --iso-8601=seconds)"
