#!/usr/bin/env python3
import os, os.path as osp, re, argparse
import pandas as pd

try:
    from atom3d.datasets.lmdb import Dataset as LMDBDataset
except Exception:
    from atom3d.datasets import LMDBDataset  # fallback

def safe_id(s):
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', str(s))
    return s[:120]  # trim just in case

def write_pdb(df: pd.DataFrame, out_path: str):
    # Column inference (ATOM3D is consistent but allow minor variants)
    name_col  = next((c for c in ['name','atom_name','atom'] if c in df.columns), None)
    resn_col  = next((c for c in ['resname','res_name'] if c in df.columns), None)
    chain_col = next((c for c in ['chain','chain_id'] if c in df.columns), None)
    resid_col = next((c for c in ['residue','resid','resseq','res_num','residue_number'] if c in df.columns), None)
    for need in (name_col, resn_col, resid_col):
        if need is None:
            raise RuntimeError(f"Missing required columns in pocket df: {df.columns.tolist()}")

    tmp = df.copy()
    tmp['__name']  = tmp[name_col].astype(str).str.upper()
    tmp['__resn']  = tmp[resn_col].astype(str).str.upper()
    tmp['__chain'] = (tmp[chain_col].astype(str).str[0] if chain_col else 'A')
    tmp['__resid'] = tmp[resid_col].astype(int)

    # sort by (chain, resid, atom order N,CA,C,O, then others)
    atom_order = {'N':0, 'CA':1, 'C':2, 'O':3}
    tmp['__aord'] = tmp['__name'].map(atom_order).fillna(9).astype(int)
    tmp = tmp.sort_values(['__chain','__resid','__aord']).reset_index(drop=True)

    with open(out_path, 'w') as f:
        for i, r in tmp.iterrows():
            an  = r['__name']
            rn  = (r['__resn'][:3]).rjust(3)
            ch  = r['__chain'] if isinstance(r['__chain'], str) and len(r['__chain'])>0 else 'A'
            rid = int(r['__resid'])
            x, y, z = float(r['x']), float(r['y']), float(r['z'])
            elem = (an[0] if len(an)>0 else ' ')
            # DSSP-friendly fixed-width ATOM line (80 cols)
            line = f"ATOM  {i+1:>5d} {an:^4}{rn:>4} {ch}{rid:>4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2}"
            f.write(line.ljust(80) + "\n")
        f.write("END\n")

def export_split(lmdb_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ds = LMDBDataset(lmdb_dir)
    for item in ds:
        pocket = item['atoms_pocket']   # protein pocket atoms ONLY (no ligand)
        base = safe_id(item.get('id','sample'))
        out = osp.join(out_dir, f"{base}.pdb")
        write_pdb(pocket, out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lba_root', required=True, help='Path to LBA/data (the folder that contains train/ val/ test/ LMDBs)')
    ap.add_argument('--out_root', required=True, help='Where to write PDBs (will create train/ val/ test)')
    args = ap.parse_args()
    for split in ['train','val','test']:
        export_split(osp.join(args.lba_root, split), osp.join(args.out_root, split))

if __name__ == '__main__':
    main()
