#!/usr/bin/env bash
set -euo pipefail

# Transfer the old project scratch tree to the new PAS2836 allocation and make
# copied artifacts operational under the new path/account.  Run from an OSC
# shell whose group membership includes PAS2836.

SRC="${SRC:-/fs/scratch/PAS3272/kopanev.1/}"
DST="${DST:-/fs/scratch/PAS2836/kopanev.1/}"
LOG_DIR="${LOG_DIR:-$DST/transfer_logs}"

mkdir -p "$DST" "$LOG_DIR"

echo "Source: $SRC"
echo "Destination: $DST"
echo "Start: $(date --iso-8601=seconds)"

# --no-times intentionally refreshes copied file mtimes to the transfer time so
# the new scratch copy is not immediately eligible for age-based purge.  We do
# not preserve old owner/group metadata; the new allocation should own the copy.
# --size-only makes reruns practical after the destination mtimes are refreshed.
rsync -rlpH --size-only --no-times --partial --human-readable --info=progress2 \
  "$SRC" "$DST" \
  2>&1 | tee "$LOG_DIR/rsync_$(date +%Y%m%d_%H%M%S).log"

echo "Rewriting copied text references from PAS3272 to PAS2836..."
python - <<'PY'
from __future__ import annotations

from pathlib import Path
import os

dst = Path(os.environ.get("DST", "/fs/scratch/PAS2836/kopanev.1/"))
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
for path in dst.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in suffixes:
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    new = text.replace("/fs/scratch/PAS3272/kopanev.1", "/fs/scratch/PAS2836/kopanev.1")
    new = new.replace("PAS3272", "PAS2836")
    if new != text:
        path.write_text(new, encoding="utf-8")
        changed += 1
print(f"rewrote {changed} copied text files")
PY

echo "Refreshing mtimes under destination tree..."
find "$DST" -exec touch -h {} +

echo "Destination size:"
du -sh "$DST"
echo "End: $(date --iso-8601=seconds)"
