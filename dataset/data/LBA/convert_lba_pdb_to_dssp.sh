#!/bin/bash
set -euo pipefail

LBA_BASE="/workspace/workspace/CD_Conv_Plus_ProNet/dataset/data/LBA"
PDB_DIR="$LBA_BASE/pdb_pocket"
DSSP_DIR="$LBA_BASE/dssp_pocket"

# use your existing mkdssp path (same you used for ProtFunct)
MKDSSP="/workspace/workspace/CD_Conv_Plus_ProNet/dssp-2.3.0/local/bin/mkdssp"

echo "🔁 PDB → DSSP (LBA)…"
for SPLIT in train val test; do
  echo "📂 $SPLIT"
  mkdir -p "$DSSP_DIR/$SPLIT"
  shopt -s nullglob
  for PDB in "$PDB_DIR/$SPLIT"/*.pdb; do
    base="$(basename "$PDB" .pdb)"
    out="$DSSP_DIR/$SPLIT/$base.dssp"
    "$MKDSSP" -i "$PDB" -o "$out" >/dev/null 2>&1 || echo "  ⚠️ failed: $base"
  done
done
echo "✅ done."
