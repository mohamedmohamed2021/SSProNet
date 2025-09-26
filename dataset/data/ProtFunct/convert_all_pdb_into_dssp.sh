#!/bin/bash
set -euo pipefail

BASE_DIR="/workspace/workspace/CD_Conv_Plus_ProNet/dataset/data/ProtFunct"
declare -A SPLIT_FILE=( ["Train"]="training.txt" ["Val"]="validation.txt" ["Test"]="testing.txt" )
MKDSSP="/workspace/workspace/CD_Conv_Plus_ProNet/dssp-2.3.0/local/bin/mkdssp"

echo "🔁 PDB → DSSP (ProtFunct)…"
for SPLIT in "${!SPLIT_FILE[@]}"; do
  PDB_DIR="$BASE_DIR/pdb_files/$SPLIT"
  DSSP_DIR="$BASE_DIR/dssp_files/$SPLIT"
  mkdir -p "$DSSP_DIR"

  echo "📂 $SPLIT"
  shopt -s nullglob
  for PDB in "$PDB_DIR"/*.pdb; do
    base="$(basename "$PDB" .pdb)"
    out="$DSSP_DIR/$base.dssp"
    echo "  → $(basename "$PDB") → $(basename "$out")"
    "$MKDSSP" -i "$PDB" -o "$out" >/dev/null 2>&1 || echo "  ⚠️ failed: $base"
  done
done
echo "✅ done."
