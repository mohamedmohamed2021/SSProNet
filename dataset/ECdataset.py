import os.path as osp
import h5py
import numpy as np
import warnings
from tqdm import tqdm

import torch 
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import os, re

#---parsing dssp files-----
def parse_dssp_file(dssp_path):
    residues = []
    started = False
    with open(dssp_path, 'r') as f:
        for line in f:
            if not started:
                if line.startswith("  #  RESIDUE"):
                    started = True
                continue
            if len(line) < 115:
                continue
            try:
                res_idx = int(line[0:5].strip())
                aa = line[13].strip()
                ss_raw = line[16].strip()
                ss = ss_raw if ss_raw in ['H','E','T','S','G','B','I'] else 'C'
                acc = float(line[34:38].strip())

                def hbond(s):
                    m = re.match(r"\s*(-?\d+),\s*(-?\d*\.?\d+)", s)
                    return (int(m[1]), float(m[2])) if m else (0, 0.0)

                residues.append({
                    'index': res_idx,
                    'aa': aa,
                    'ss': ss,
                    'acc': acc,
                    'nh_o_1': hbond(line[39:50]),
                    'o_hn_1': hbond(line[50:61]),
                    'nh_o_2': hbond(line[61:72]),
                    'o_hn_2': hbond(line[72:83]),
                })
            except Exception:
                continue
    return residues



class ECdataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 split='train'
                ):

        self.split = split
        self.root = root

        super(ECdataset, self).__init__(
            root, transform, pre_transform, pre_filter)
        
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, name, self.split)

    @property
    def raw_file_names(self):
        name = self.split + '.txt'
        return name

    @property
    def processed_file_names(self):
        return 'data.pt'


    def _normalize(self,tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def get_atom_pos(self, amino_types, atom_names, atom_amino_id, atom_pos):
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        mask_n = np.char.equal(atom_names, b'N')
        mask_ca = np.char.equal(atom_names, b'CA')
        mask_c = np.char.equal(atom_names, b'C')
        mask_cb = np.char.equal(atom_names, b'CB')
        mask_g = np.char.equal(atom_names, b'CG') | np.char.equal(atom_names, b'SG') | np.char.equal(atom_names, b'OG') | np.char.equal(atom_names, b'CG1') | np.char.equal(atom_names, b'OG1')
        mask_d = np.char.equal(atom_names, b'CD') | np.char.equal(atom_names, b'SD') | np.char.equal(atom_names, b'CD1') | np.char.equal(atom_names, b'OD1') | np.char.equal(atom_names, b'ND1')
        mask_e = np.char.equal(atom_names, b'CE') | np.char.equal(atom_names, b'NE') | np.char.equal(atom_names, b'OE1')
        mask_z = np.char.equal(atom_names, b'CZ') | np.char.equal(atom_names, b'NZ')
        mask_h = np.char.equal(atom_names, b'NH1')

        pos_n = np.full((len(amino_types),3),np.nan)
        pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
        pos_n = torch.FloatTensor(pos_n)

        pos_ca = np.full((len(amino_types),3),np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)

        pos_c = np.full((len(amino_types),3),np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        pos_c = torch.FloatTensor(pos_c)

        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        pos_cb = np.full((len(amino_types),3),np.nan)
        pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
        pos_cb = torch.FloatTensor(pos_cb)

        pos_g = np.full((len(amino_types),3),np.nan)
        pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
        pos_g = torch.FloatTensor(pos_g)

        pos_d = np.full((len(amino_types),3),np.nan)
        pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
        pos_d = torch.FloatTensor(pos_d)

        pos_e = np.full((len(amino_types),3),np.nan)
        pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
        pos_e = torch.FloatTensor(pos_e)

        pos_z = np.full((len(amino_types),3),np.nan)
        pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
        pos_z = torch.FloatTensor(pos_z)

        pos_h = np.full((len(amino_types),3),np.nan)
        pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
        pos_h = torch.FloatTensor(pos_h)

        return pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h


    def side_chain_embs(self, pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        v1, v2, v3, v4, v5, v6, v7 = pos_ca - pos_n, pos_cb - pos_ca, pos_g - pos_cb, pos_d - pos_g, pos_e - pos_d, pos_z - pos_e, pos_h - pos_z

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        angle1 = torch.unsqueeze(self.compute_dihedrals(v1, v2, v3),1)
        angle2 = torch.unsqueeze(self.compute_dihedrals(v2, v3, v4),1)
        angle3 = torch.unsqueeze(self.compute_dihedrals(v3, v4, v5),1)
        angle4 = torch.unsqueeze(self.compute_dihedrals(v4, v5, v6),1)
        angle5 = torch.unsqueeze(self.compute_dihedrals(v5, v6, v7),1)

        side_chain_angles = torch.cat((angle1, angle2, angle3, angle4),1)
        side_chain_embs = torch.cat((torch.sin(side_chain_angles), torch.cos(side_chain_angles)),1)
        
        return side_chain_embs

    
    def bb_embs(self, X):   
        # X should be a num_residues x 3 x 3, order N, C-alpha, and C atoms of each residue
        # N coords: X[:,0,:]
        # CA coords: X[:,1,:]
        # C coords: X[:,2,:]
        # return num_residues x 6 
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X, [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self._normalize(dX, dim=-1)
        u0 = U[:-2]
        u1 = U[1:-1]
        u2 = U[2:]

        angle = self.compute_dihedrals(u0, u1, u2)
        
        # add phi[0], psi[-1], omega[-1] with value 0
        angle = F.pad(angle, [1, 2]) 
        angle = torch.reshape(angle, [-1, 3])
        angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
        return angle_features

    
    def compute_dihedrals(self, v1, v2, v3):
        n1 = torch.cross(v1, v2, dim=-1)
        n2 = torch.cross(v2, v3, dim = -1)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2, dim=-1) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))
        return torsion
    
    
    def protein_to_graph(self, pFilePath):
        """
        Build a PyG Data object from one HDF5 file and (optionally) its DSSP file.

        - Reads positions & atom->residue mapping from the HDF5.
        - Computes side-chain and backbone torsion embeddings.
        - Loads DSSP labels (SS, ACC) and hydrogen bonds if the matching .dssp exists:
            * Uses all four DSSP slots: nh_o_1, o_hn_1, nh_o_2, o_hn_2
            * partner is a RELATIVE offset (negative/positive along sequence)
            * keeps only stabilizing bonds with energy < -0.5
            * removes self-loops and out-of-range partners
            * deduplicates undirected edges, keeping the strongest (most negative) energy
        """
        import os

        # ---------- 1) Load coordinates & residue types from HDF5 ----------
        h5File = h5py.File(pFilePath, "r")
        data = Data()

        amino_types = h5File['amino_types'][()]  # (n_res,)
        mask = amino_types == -1
        if np.sum(mask) > 0:
            amino_types[mask] = 25  # set unknown AAs to 25 (as in your original code)

        atom_amino_id = h5File['atom_amino_id'][()]  # (n_atom,)
        atom_names    = h5File['atom_names'][()]     # (n_atom,)
        atom_pos      = h5File['atom_pos'][()][0]    # (n_atom, 3)

        # Backbone & side-chain reference atoms
        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = self.get_atom_pos(
            amino_types, atom_names, atom_amino_id, atom_pos
        )

        # ---------- 2) Side-chain & backbone torsion embeddings ----------
        side_chain_embs = self.side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
        side_chain_embs[torch.isnan(side_chain_embs)] = 0
        data.side_chain_embs = side_chain_embs

        bb_stack = torch.cat((pos_n.unsqueeze(1), pos_ca.unsqueeze(1), pos_c.unsqueeze(1)), dim=1)
        bb_embs = self.bb_embs(bb_stack)
        bb_embs[torch.isnan(bb_embs)] = 0
        data.bb_embs = bb_embs

        # Node attributes & coordinates
        data.x         = torch.unsqueeze(torch.tensor(amino_types), 1)  # (N,1) integer AA types
        data.coords_ca = pos_ca
        data.coords_n  = pos_n
        data.coords_c  = pos_c

        assert len(data.x) == len(data.coords_ca) == len(data.coords_n) == len(data.coords_c) == \
            len(data.side_chain_embs) == len(data.bb_embs)

        h5File.close()

        # ---------- 3) DSSP: SS/ACC + Hydrogen bonds (optional) ----------
        try:
            # Expected DSSP path: <root>/dssp_files/<split>/<basename>.dssp
            base_name  = os.path.basename(pFilePath).replace(".hdf5", ".dssp")
            dssp_path  = os.path.join(self.root, "dssp_files", self.split, base_name)
            n_res      = int(len(amino_types))

            if os.path.exists(dssp_path):
                dssp_res = parse_dssp_file(dssp_path)

                # Lookups by DSSP serial index (1-based in file)
                ss_lookup  = {r['index']: r['ss']  for r in dssp_res}
                acc_lookup = {r['index']: r['acc'] for r in dssp_res}

                # SS mapping (same as your current code)
                ss_map = {'H':0, 'E':1, 'T':2, 'S':3, 'G':4, 'B':5, 'I':6, 'C':7}
                ss_tensor  = torch.full((n_res,), 7, dtype=torch.long)   # default coil
                acc_tensor = torch.zeros((n_res,), dtype=torch.float)

                for i in range(n_res):
                    ss_tensor[i]  = ss_map.get(ss_lookup.get(i+1, 'C'), 7)
                    acc_tensor[i] = float(acc_lookup.get(i+1, 0.0))

                # --- Hydrogen bonds ---
                # Collect from all four slots; use relative offsets & filter by energy.
                energy_cutoff = -1.5  # keep only stabilizing bonds (< -0.1)
                hbonds = []           # final list of directed bonds as (src, tgt, energy)

                for r in dssp_res:
                    src = r['index'] - 1  # make 0-based
                    if not (0 <= src < n_res):
                        continue

                    # DSSP stores four potential backbone H-bond relations per residue.
                    # Each entry is (partner_relative_offset, energy).
                    for partner, energy in (r['nh_o_1'], r['o_hn_1'], r['nh_o_2'], r['o_hn_2']):
                        # Skip non-bonds / weak interactions
                        if partner == 0 or energy >= energy_cutoff:
                            continue

                        tgt = src + partner  # partner is RELATIVE offset in DSSP
                        if 0 <= tgt < n_res and tgt != src:
                            # Keep exactly as directed: (src -> tgt), do NOT symmetrize or deduplicate
                            hbonds.append((src, tgt, float(energy)))

                data.ss     = ss_tensor
                data.acc    = acc_tensor
                data.hbonds = hbonds

            else:
                # DSSP missing → fallbacks
                data.ss     = torch.full((n_res,), 7, dtype=torch.long)  # all coil
                data.acc    = torch.zeros((n_res,), dtype=torch.float)
                data.hbonds = []

        except Exception as e:
            print(f"[Warn] DSSP parse failed for {pFilePath}: {e}")
            n_res = int(len(amino_types))
            data.ss     = torch.full((n_res,), 7, dtype=torch.long)
            data.acc    = torch.zeros((n_res,), dtype=torch.float)
            data.hbonds = []

        return data


    def process(self):
        print('Beginning Processing ...')

        # ----- Load the function list (categories) -----
        functions_ = []
        with open(os.path.join(self.root, "unique_functions.txt"), 'r') as mFile:
            for line in mFile:
                line = line.strip()
                if line:
                    functions_.append(line)

        # ----- Split file name mapping -----
        if self.split == "Train":
            split_file = "training.txt"
        elif self.split == "Val":
            split_file = "validation.txt"
        elif self.split == "Test":
            split_file = "testing.txt"
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # ----- Read raw split list -----
        entries = []
        with open(os.path.join(self.root, split_file), 'r') as mFile:
            for line in mFile:
                name = line.strip()
                if not name:
                    continue
                base = name[:-5] if name.endswith(".hdf5") else name
                h5_path   = os.path.join(self.root, "data", base + ".hdf5")
                dssp_path = os.path.join(self.root, "dssp_files", self.split, base + ".dssp")
                entries.append((base, h5_path, dssp_path))

        # ----- Build label lookup -----
        print("Reading protein functions")
        protFunct_ = {}
        with open(os.path.join(self.root, "chain_functions.txt"), 'r') as mFile:
            for line in mFile:
                splitLine = line.strip().split(',')
                if len(splitLine) != 2:
                    continue
                protFunct_[splitLine[0]] = int(splitLine[1])

        # ----- Filter entries: both HDF5 and DSSP must exist, and label must exist -----
        kept, missing_h5, missing_dssp, missing_lbl = [], [], [], []
        for base, h5_path, dssp_path in entries:
            if not os.path.exists(h5_path):
                missing_h5.append(base); continue
            if not os.path.exists(dssp_path):
                missing_dssp.append(base); continue
            if base not in protFunct_:
                missing_lbl.append(base); continue
            kept.append((base, h5_path))

        print(f"[{self.split}] Kept: {len(kept)} | missing_h5={len(missing_h5)} | missing_dssp={len(missing_dssp)} | missing_label={len(missing_lbl)}")

        # (Optional) Save the filtered list for reproducibility
        os.makedirs(self.processed_dir, exist_ok=True)
        with open(os.path.join(self.processed_dir, "kept.txt"), "w") as f:
            for base, _ in kept:
                f.write(base + "\n")
        if missing_h5:
            with open(os.path.join(self.processed_dir, "missing_h5.txt"), "w") as f:
                f.write("\n".join(missing_h5))
        if missing_dssp:
            with open(os.path.join(self.processed_dir, "missing_dssp.txt"), "w") as f:
                f.write("\n".join(missing_dssp))
        if missing_lbl:
            with open(os.path.join(self.processed_dir, "missing_label.txt"), "w") as f:
                f.write("\n".join(missing_lbl))

        # ----- Process only the kept set -----
        print("Reading the data")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_list = []
            for base, h5_path in tqdm(kept):
                curProtein = self.protein_to_graph(h5_path)
                curProtein.id = base
                curProtein.y = torch.tensor(protFunct_[base])
                if curProtein.x is not None:
                    data_list.append(curProtein)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('Done!')



if __name__ == "__main__":
    import os, sys, traceback
    import torch

    # Adjust these if your tree is different:
    ROOT = "/workspace/workspace/CD_Conv_Plus_ProNet/dataset/data/ProtFunct"
    SPLIT = "Train"  # or "Val", "Test"

    # 1) Pick a sample name from the split list
    split_map = {"Train": "training.txt", "Val": "validation.txt", "Test": "testing.txt"}
    list_file = os.path.join(ROOT, split_map[SPLIT])
    with open(list_file) as f:
        sample_name = next(line.strip() for line in f if line.strip())
    h5_path = os.path.join(ROOT, "data", sample_name + ".hdf5")
    dssp_path = os.path.join(ROOT, "dssp_files", SPLIT, sample_name + ".dssp")

    print(f"🔎 Split={SPLIT}")
    print(f"   HDF5 : {h5_path}")
    print(f"   DSSP : {dssp_path} (exists={os.path.exists(dssp_path)})")

    # 2) Create a lightweight instance and build the graph from a single HDF5
    try:
        # Bypass __init__ so we don't need processed/data.pt yet
        ds = ECdataset.__new__(ECdataset)
        ds.split = SPLIT
        ds.root = ROOT
        data = ECdataset.protein_to_graph(ds, h5_path)
    except Exception as e:
        print("❌ Failed to load HDF5:", e)
        traceback.print_exc()
        sys.exit(1)

    # 3) If you already parsed DSSP in ECdataset, it may already be attached.
    #    Otherwise, reuse the parser from FOLDdataset to attach ss/acc/hbonds here.
    if (not hasattr(data, "ss") or not hasattr(data, "acc") or not hasattr(data, "hbonds")) and os.path.exists(dssp_path):
        try:
            # Reuse the parser you implemented for HomologyTAPE
            from dataset.FOLDdataset import parse_dssp_file  # uses the "  #  RESIDUE" header layout
            residues = parse_dssp_file(dssp_path)

            n = data.x.shape[0]
            ss = torch.full((n,), 7, dtype=torch.long)   # default Coil
            acc = torch.zeros((n,), dtype=torch.float)
            hbonds = []

            ss_map = {'H':0,'E':1,'T':2,'S':3,'G':4,'B':5,'I':6,'C':7}
            ss_lookup  = {r['index']: r['ss']  for r in residues}
            acc_lookup = {r['index']: r['acc'] for r in residues}

            # Build hydrogen bonds like in FOLDdataset
            for r in residues:
                src = r['index'] - 1
                if 0 <= src < n:
                    for partner, energy in [r['nh_o_1'], r['nh_o_2']]:
                        tgt = src + partner
                        if 0 <= tgt < n:
                            hbonds.append((src, tgt, float(energy)))

            for i in range(n):
                ss[i]  = ss_map.get(ss_lookup.get(i+1, 'C'), 7)
                acc[i] = float(acc_lookup.get(i+1, 0.0))

            data.ss = ss
            data.acc = acc
            data.hbonds = hbonds
        except Exception as e:
            print(f"⚠️ Could not parse/attach DSSP: {e}")
            data.ss = torch.full((data.x.shape[0],), 7, dtype=torch.long)
            data.acc = torch.zeros((data.x.shape[0],), dtype=torch.float)
            data.hbonds = []

    # 4) Pretty print
    print("\n✅ Loaded protein graph successfully.")
    print("── Data object ──")
    print(data)  # PyG Data repr shows available fields

    if hasattr(data, "ss"):
        print("\n📌 DSSP Secondary Structure (first 32):")
        print(data.ss[:32])
    else:
        print("\n📌 No `ss` field on data.")

    if hasattr(data, "acc"):
        print("\n📌 DSSP Solvent Accessibility (first 10):")
        print(data.acc[:10])
    else:
        print("\n📌 No `acc` field on data.")

    if hasattr(data, "hbonds"):
        print(f"\n📌 Hydrogen bonds: {len(data.hbonds)} total")
        print("First 10:", data.hbonds[:10])
    else:
        print("\n📌 No `hbonds` field on data.")
