#!/bin/bash

echo "🔁 Starting HDF5 → PDB conversion with DSSP-compliant formatting..."

# Base dataset directory
BASE_DIR="/workspace/workspace/CD_Conv_Plus_ProNet/dataset/data/HomologyTAPE"
SPLITS=("training" "test_fold" "test_family" "test_superfamily" "validation")

for SPLIT in "${SPLITS[@]}"; do
    echo "📂 Processing split: $SPLIT"
    SPLIT_DIR="$BASE_DIR/$SPLIT"
    PDB_DIR="$SPLIT_DIR/pdb_files"
    mkdir -p "$PDB_DIR"

    for H5_FILE in "$SPLIT_DIR"/*.hdf5; do
        if [[ -f "$H5_FILE" ]]; then
            BASENAME=$(basename "$H5_FILE" .hdf5)
            PDB_PATH="$PDB_DIR/$BASENAME.pdb"
            echo "  → Converting: $BASENAME.hdf5 → $BASENAME.pdb"

            python3 - <<EOF
import h5py

def format_pdb_atom_line(index, name, resname, chain, resid, x, y, z, occ=1.00, temp=0.00, element=''):
    return (
        f"ATOM  {index:>5d} "
        f"{name:^4}"     # atom name, centered
        f" {resname:>3} {chain}"  # residue name + chain
        f"{resid:>4d}    "        # residue id
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occ:6.2f}{temp:6.2f}          "
        f"{element:>2}"
    ).ljust(80)


f = h5py.File("$H5_FILE", "r")
p = open("$PDB_PATH", "w")

a = [n.decode("utf-8") for n in f["atom_names"][:]]
r = [n.decode("utf-8") for n in f["atom_residue_names"][:]]
i = f["atom_residue_id"][:]
c = [n.decode("utf-8") for n in f["atom_chain_names"][:]]
x = f["atom_pos"][0]

for j in range(len(a)):
    line = format_pdb_atom_line(
        index=j + 1,
        name=a[j],
        resname=r[j],
        chain=c[j],
        resid=int(i[j]) + 1,
        x=x[j][0],
        y=x[j][1],
        z=x[j][2],
        element=a[j][0] if a[j] else ''
    )
    p.write(line + "\\n")

p.write("END\\n")
p.close()
f.close()
EOF
        fi
    done
done

echo "✅ All HDF5 → PDB conversions completed with DSSP formatting."
