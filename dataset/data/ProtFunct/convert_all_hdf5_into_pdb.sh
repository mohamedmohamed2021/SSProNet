#!/bin/bash
set -euo pipefail

BASE_DIR="/workspace/workspace/CD_Conv_Plus_ProNet/dataset/data/ProtFunct"
H5_DIR="$BASE_DIR/data"
declare -A SPLIT_FILE=( ["Train"]="training.txt" ["Val"]="validation.txt" ["Test"]="testing.txt" )

echo "🔁 HDF5 → PDB (ProtFunct)…"
for SPLIT in "${!SPLIT_FILE[@]}"; do
  LIST_FILE="$BASE_DIR/${SPLIT_FILE[$SPLIT]}"
  PDB_DIR="$BASE_DIR/pdb_files/$SPLIT"
  mkdir -p "$PDB_DIR"

  echo "📂 $SPLIT  (list: $(basename "$LIST_FILE"))"
  while IFS= read -r name || [[ -n "$name" ]]; do
    [[ -z "$name" ]] && continue
    H5_PATH="$H5_DIR/$name"
    [[ -f "$H5_PATH" ]] || H5_PATH="$H5_DIR/${name}.hdf5"
    [[ -f "$H5_PATH" ]] || { echo "  ⚠️ missing: $name(.hdf5)"; continue; }

    base="$(basename "$H5_PATH" .hdf5)"
    PDB_PATH="$PDB_DIR/${base}.pdb"
    echo "  → $(basename "$H5_PATH") → $(basename "$PDB_PATH")"

    python3 - "$H5_PATH" "$PDB_PATH" <<'PY'
import sys, h5py

H5, OUT = sys.argv[1], sys.argv[2]

# fallback mapping (used ONLY if residue names per-atom are absent)
AA3 = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
       "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
       "ASX","GLX","XLE","SEC","PYL","UNK"]  # last entries are safe fallbacks

def decode(arr):
    return [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]

def format_line(idx, aname, resname, chain, resid, x,y,z, occ=1.00, temp=0.00):
    # DSSP-friendly PDB format (fixed width, 80 cols)
    return (f"ATOM  {idx:>5d} "
            f"{aname:^4}"
            f" {resname:>3} {chain}"
            f"{resid:>4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{occ:6.2f}{temp:6.2f}          "
            f"{(aname[0] if aname else ' '):>2}").ljust(80)

with h5py.File(H5, "r") as f, open(OUT, "w") as out:
    keys = set(f.keys())
    # atoms
    atom_names = decode(f["atom_names"][:])
    # coordinates
    pos = f["atom_pos"][()]
    if pos.ndim == 3: pos = pos[0]
    # chain names (optional)
    if "atom_chain_names" in keys:
        chain_names = decode(f["atom_chain_names"][:])
    else:
        chain_names = ["A"] * len(atom_names)

    # residue name/id per atom (prefer direct fields; otherwise derive)
    if "atom_residue_names" in keys:
        res_names = decode(f["atom_residue_names"][:])
    else:
        # derive from amino_types + atom_amino_id if available
        if "amino_types" in keys and "atom_amino_id" in keys:
            amino_types = f["amino_types"][:]
            atom_amino_id = f["atom_amino_id"][:]
            # guard unknowns
            res_names = []
            for aid in atom_amino_id:
                t = int(amino_types[aid]) if 0 <= int(aid) < len(amino_types) else 25
                t = t if 0 <= t < len(AA3) else 25
                res_names.append(AA3[t])
        else:
            res_names = ["UNK"] * len(atom_names)

    if "atom_residue_id" in keys:
        resid = f["atom_residue_id"][:].astype(int)
    elif "atom_amino_id" in keys:
        resid = f["atom_amino_id"][:].astype(int) + 1
    else:
        resid = list(range(1, len(atom_names)+1))

    # write PDB
    for i,(an, rn, ch, rid, (x,y,z)) in enumerate(zip(atom_names, res_names, chain_names, resid, pos), start=1):
        out.write(format_line(i, an, rn, ch, int(rid), float(x), float(y), float(z)) + "\n")
    out.write("END\n")
PY
  done < "$LIST_FILE"
done

echo "✅ done."
