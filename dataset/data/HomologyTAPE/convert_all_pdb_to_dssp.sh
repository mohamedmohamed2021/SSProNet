#!/bin/bash

echo "🔁 Starting PDB → DSSP conversion for all dataset splits..."

# Base dataset directory (same as in your HDF5 → PDB script)
BASE_DIR="/workspace/workspace/CD_Conv_Plus_ProNet/dataset/data/HomologyTAPE"
SPLITS=("training" "test_fold" "test_family" "test_superfamily" "validation")

# Loop over each data split
for SPLIT in "${SPLITS[@]}"; do
    echo "📂 Processing split: $SPLIT"
    
    PDB_DIR="$BASE_DIR/$SPLIT/pdb_files"
    DSSP_DIR="$BASE_DIR/$SPLIT/dssp_files"
    mkdir -p "$DSSP_DIR"

    for PDB_FILE in "$PDB_DIR"/*.pdb; do
        if [[ -f "$PDB_FILE" ]]; then
            BASENAME=$(basename "$PDB_FILE" .pdb)
            DSSP_FILE="$DSSP_DIR/${BASENAME}.dssp"

            echo "  → $BASENAME.pdb → $BASENAME.dssp"
            mkdssp -i "$PDB_FILE" -o "$DSSP_FILE" > /dev/null 2>&1

            if [[ $? -ne 0 ]]; then
                echo "  ⚠️  Conversion failed for: $BASENAME.pdb"
            fi
        fi
    done
done

echo "✅ All PDB → DSSP conversions completed."
