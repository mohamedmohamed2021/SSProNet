import os, os.path as osp
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset

# ATOM3D LMDB
try:
    from atom3d.datasets.lmdb import Dataset as LMDBDataset
except Exception:
    from atom3d.datasets import LMDBDataset  # compat

# ------------ helpers (aligned to your EC/FOLD code paths) ------------
AA3_TO_IDX = {
    'ALA':0,'ARG':1,'ASN':2,'ASP':3,'CYS':4,'GLU':5,'GLN':6,'GLY':7,'HIS':8,
    'ILE':9,'LEU':10,'LYS':11,'MET':12,'PHE':13,'PRO':14,'SER':15,'THR':16,
    'TRP':17,'TYR':18,'VAL':19,
    # map variants → parent
    'HID':8,'HIE':8,'HIP':8,'ASH':3,'GLH':5,'CYM':4,'CYX':4,'MSE':12,
    'UNK':25,'XAA':25
}

def _aa_idx(resname: str) -> int:
    return AA3_TO_IDX.get(resname.upper(), 25)

def _normalize(t, dim=-1):
    return torch.nan_to_num(t / (t.norm(dim=dim, keepdim=True) + 1e-8))

def _dihedral(v1, v2, v3):
    n1 = torch.cross(v1, v2, dim=-1)
    n2 = torch.cross(v2, v3, dim=-1)
    a = (n1 * n2).sum(-1)
    b = torch.nan_to_num((torch.cross(n1, n2, dim=-1) * v2).sum(-1) / (v2.norm(dim=1) + 1e-8))
    return torch.atan2(b, a)

def _bb_embs_from_N_CA_C(pos_n, pos_ca, pos_c):
    # 6D: cos/sin of (phi, psi, omega) per residue (same design as your EC/FOLD). 
    X = torch.stack([pos_n, pos_ca, pos_c], dim=1).reshape(-1, 3)
    dX = X[1:] - X[:-1]
    U = _normalize(dX)
    u0, u1, u2 = U[:-2], U[1:-1], U[2:]
    ang = _dihedral(u0, u1, u2)
    ang = F.pad(ang, (1, 2)).view(-1, 3)
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=1)

def _pick(df: pd.DataFrame, choices: List[str]):
    for c in choices:
        if c in df.columns: return c
    return None

def _parse_dssp(dssp_path: str, n_res: int):
    # Minimal DSSP parser aligned with your EC/FOLD mapping (H/E/T/S/G/B/I→0..6; else C→7). 
    ss_map = {'H':0,'E':1,'T':2,'S':3,'G':4,'B':5,'I':6}
    ss = torch.full((n_res,), 7, dtype=torch.long)   # default coil
    acc = torch.zeros((n_res,), dtype=torch.float)
    hbonds = []  # list of (src,tgt,energy)
    if not (dssp_path and osp.exists(dssp_path)):
        return ss, acc, hbonds

    import re
    with open(dssp_path, 'r') as f:
        started = False
        rows = []
        for line in f:
            if not started:
                if line.startswith("  #  RESIDUE"): started = True
                continue
            if len(line) < 83:  # ensure columns present
                continue
            try:
                res_idx = int(line[0:5].strip()) - 1  # 0-based
                if 0 <= res_idx < n_res:
                    ss_char = line[16].strip()
                    ss[res_idx]  = ss_map.get(ss_char, 7)
                    acc[res_idx] = float(line[34:38].strip())
                    # HBonds: nh_o_1 (39:50), nh_o_2 (61:72) with (offset, energy)
                    def hb(s):
                        m = re.match(r"\s*(-?\d+),\s*(-?\d*\.?\d+)", s)
                        return (int(m[1]), float(m[2])) if m else (0, 0.0)
                    nh_o_1 = hb(line[39:50]); nh_o_2 = hb(line[61:72])
                    for partner, energy in (nh_o_1, nh_o_2):
                        tgt = res_idx + partner
                        if 0 <= tgt < n_res:
                            hbonds.append((res_idx, tgt, energy))
            except Exception:
                continue
    return ss, acc, hbonds

# -------------------- Dataset --------------------
class LBADataset_SSProNet(InMemoryDataset):
    """
    ATOM3D LBA (LMDB) → residue-level graphs for SSProNet.

    Per-sample Data contains:
      - x : (N,1) long AA indices in [0..25]
      - coords_ca, coords_n, coords_c : (N,3) float
      - bb_embs : (N,6) float
      - side_chain_embs : (N,8) float  (set to zeros here; safe default)
      - ss : (N,) long in [0..7] (DSSP if available; else all 7)
      - acc : (N,) float (DSSP if available; else zeros)
      - label : scalar float (pK = neglog_affinity)
    """
    def __init__(self, root: str, dssp_root: str = None, transform=None, pre_transform=None, pre_filter=None):
        self.split = osp.basename(root.rstrip('/'))
        self.root_dir = root
        self.dssp_root = dssp_root
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root_dir, 'processed')

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def process(self):
        ds = LMDBDataset(self.root_dir)
        out = []
        skipped = 0
        for item in tqdm(ds, desc=f"LBA→SSProNet ({self.split})"):
            try:
                out.append(self._to_residue_graph(item))
            except Exception:
                skipped += 1
                continue
        if skipped:
            print(f"⚠️ Skipped {skipped} entries.")
        data, slices = self.collate(out)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])

    # ---------------- core conversion ----------------
    def _to_residue_graph(self, item: Dict) -> Data:
        pocket: pd.DataFrame = item['atoms_pocket']  # protein atoms
        ligand: pd.DataFrame = item['atoms_ligand']  # ligand atoms

        # Column names can vary slightly across ATOM3D versions
        name_c   = _pick(pocket, ['name','atom_name','atom'])
        elem_c   = _pick(pocket, ['element'])
        resn_c   = _pick(pocket, ['resname','res_name'])
        chain_c  = _pick(pocket, ['chain','chain_id'])
        resid_c  = _pick(pocket, ['residue','resid','resseq','res_num','residue_number'])
        assert name_c and elem_c and resn_c and resid_c, f"Missing required columns in pocket: {pocket.columns.tolist()}"

        # Group atoms by residue (chain,resid) if chain exists, else by resid only
        group_cols = [c for c in [chain_c, resid_c] if c is not None]
        residues = []
        for _, grp in pocket.groupby(group_cols, dropna=False):
            resname = str(grp[resn_c].iloc[0]).upper()
            aa = _aa_idx(resname)

            def pick(atom_name: str):
                m = grp[grp[name_c].str.upper() == atom_name]
                if len(m) == 0: return None
                return torch.tensor(m[['x','y','z']].to_numpy()[0], dtype=torch.float32)

            ca = pick('CA')
            n  = pick('N')
            c  = pick('C')
            # fallback: if missing, use CA (or centroid if CA missing)
            if ca is None:
                ca = torch.tensor(grp[['x','y','z']].to_numpy().mean(axis=0), dtype=torch.float32)
            if n  is None: n = ca.clone()
            if c  is None: c = ca.clone()
            residues.append((aa, n, ca, c))

        if len(residues) == 0:
            raise RuntimeError("No residues in pocket.")

        aa_idx = torch.tensor([r[0] for r in residues], dtype=torch.long).unsqueeze(1)
        pos_n  = torch.stack([r[1] for r in residues], dim=0)
        pos_ca = torch.stack([r[2] for r in residues], dim=0)
        pos_c  = torch.stack([r[3] for r in residues], dim=0)

        bb = _bb_embs_from_N_CA_C(pos_n, pos_ca, pos_c)
        bb[torch.isnan(bb)] = 0

        # side-chain embeddings: 8-dim placeholder (your EC/FOLD code uses 8; safe to start with zeros) 
        side_chain = torch.zeros((aa_idx.shape[0], 8), dtype=torch.float32)

        # DSSP / H-bonds (optional)
        base_id = str(item.get('id', 'sample'))
        dssp_path = None
        if self.dssp_root:
            # expected: <dssp_root>/<split>/<base>.dssp
            cand = osp.join(self.dssp_root, self.split, f"{base_id}.dssp")
            dssp_path = cand if osp.exists(cand) else None
            ss, acc, hbonds = _parse_dssp(dssp_path, n_res=len(residues)) if dssp_path else (
            torch.full((len(residues),), 7, dtype=torch.long),
            torch.zeros((len(residues),), dtype=torch.float),
            []
        )

        # pK label
        label = torch.tensor(float(item['scores']['neglog_aff']), dtype=torch.float32)

        data = Data(
            x=aa_idx,
            coords_ca=pos_ca, coords_n=pos_n, coords_c=pos_c,
            bb_embs=bb, side_chain_embs=side_chain,
            ss=ss, acc=acc,
            hbonds=hbonds,       # SSProNet can consume this (optional)
            label=label,
        )
        data.id = base_id
        return data
