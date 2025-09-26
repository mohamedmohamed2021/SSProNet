import os.path as osp
import h5py 
import numpy as np
import warnings 
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch_geometric.data import (InMemoryDataset, Data)
import re, os

#-----------------------------
# I have added a process to treats .dssp files to extract secondary structure informationn.
#-----------------------------

import re

def parse_dssp_file(dssp_path):
    """
    Parse DSSP output into a list of dictionaries (one per residue).
    Each dictionary will contain:
        - index: int
        - aa: str
        - ss: str (secondary structure class: H, E, C, etc.)
        - acc: float (solvent accessibility)
        - nh_o_1, o_hn_1, nh_o_2, o_hn_2: hydrogen bond partners and energies
    """
    residues = []
    started = False
    with open(dssp_path, 'r') as f:
        for line in f:
            if not started:
                if line.startswith("  #  RESIDUE"):  # Header
                    started = True
                continue
            
            if len(line) < 115:
                continue  # skip incomplete lines

            try:
                res_idx = int(line[0:5].strip())
                aa = line[13].strip()
                ss_raw = line[16].strip()
                ss = ss_raw if ss_raw in ['H', 'E', 'T', 'S', 'G', 'B', 'I'] else 'C'  # fallback to coil
                acc = float(line[34:38].strip())

                # Hydrogen bond columns may contain things like " 14,-1.5"
                def extract_hbond(s):
                    match = re.match(r"\s*(-?\d+),\s*(-?\d*\.?\d+)", s)
                    return (int(match.group(1)), float(match.group(2))) if match else (0, 0.0)

                nh_o_1 = extract_hbond(line[39:50])
                o_hn_1 = extract_hbond(line[50:61])
                nh_o_2 = extract_hbond(line[61:72])
                o_hn_2 = extract_hbond(line[72:83])

                residues.append({
                    'index': res_idx,
                    'aa': aa,
                    'ss': ss,
                    'acc': acc,
                    'nh_o_1': nh_o_1,
                    'o_hn_1': o_hn_1,
                    'nh_o_2': nh_o_2,
                    'o_hn_2': o_hn_2
                })
            except Exception as e:
                continue

    return residues


class FOLDdataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter= None, split='train',
        #new
        n_node: int = -1,
        ct_lst: tuple = (50, 100, 150, 200, 250, 300, 400, 500, 2000), 
        size_mode: str = "residues",
        **kwargs ):

        """
        n_node: bucket index (−1 means 'no bucketing / full split')
        ct_lst: ascending list of bucket upper bounds
        size_mode: 'residues' (default), 'graph_nodes' (slower), or 'ca_atoms' (if available)
        """
        self.split_orig = split
        self.root= root
        self.n_node = int(n_node)
        self.ct_lst = list(ct_lst)
        self.size_mode = size_mode

        # For TEST splits, append cutoff to split name when bucketed: --> f"{split}_{ct_hi}"
        test_like = self.split_orig in ('test_fold', 'test_superfamily', 'test_family')
        if test_like and (self.n_node >= 0):
            ct_hi = int(self.ct_lst[self.n_node])
            self.split = f"{self.split_orig}_{ct_hi}"
        else:
            self.split = self.split_orig

        super(FOLDdataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        # IMPORTANT: load using the **effective** processed path
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed', self.split)
    
    @property	
    def raw_file_names(self):
        return f"{self.split_orig}.txt"
    
    @property
    def processed_file_names(self):
        # If your original version used f'{self.split}.pt' already, keep it.
        # Just ensure we use the *effective* split with the cutoff suffix.
        return [f'{self.split}.pt']

    #---------- SCHull like processing HELPERS: Start----------
    def _num_residues_h5(self, h5_path: str) -> int:
        """
        Fast path: use residue array length (e.g., 'amino_types') from your HDF5.
        Adjust the key if your file uses a different one.
        """
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'amino_types' in f:
                    return int(len(f['amino_types'][()]))
                # Fallbacks (customize if needed):
                if 'residue_types' in f:
                    return int(len(f['residue_types'][()]))
                if 'coords_ca' in f:
                    return int(f['coords_ca'][()].shape[0])
        except Exception:
            pass
        return -1

    
    def _in_bucket(self, n: int, idx: int) -> bool:
        """SCHull buckets: idx==0 → n < ct[0]; else ct[idx-1] ≤ n < ct[idx]."""
        if idx < 0:
            return True
        if idx == 0:
            return n < self.ct_lst[0]
        lo = self.ct_lst[idx - 1]
        hi = self.ct_lst[idx]
        return (n >= lo) and (n < hi)

    #---------- SCHull like processing HELPERS: End----------

    def normalize(self,tensor, dim=1):
        ''' Normalize 'tensor' along 'dim' without 'nan's'''
        return torch.nan_to_num(torch.div(tensor,torch.norm(tensor,dim=dim,keepdim=True)))

    def get_atom_pos(self, amino_types, atom_names, atom_amino_id, atom_pos):
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/E1, _Z, NH1
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
        ''' Side chain embeddings '''
        # computes direction vectors 
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
    
    def compute_dihedrals(self, v1, v2, v3):
        # v1, v2, v3: are direction vectors (defined in the 'side_chain_embs'fct and also in 'bb_embs' fct as u0,u1,u2)
        # e.g. v1 = pos_ca - pos_n 
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))
        return torsion

    def bb_embs(self, X):   
        ''' Backbone embeddings '''
        # X should be a num_residues x 3 x 3, order N, C-alpha, and C atoms of each residue
        # N coords: X[:,0,:]
        # CA coords: X[:,1,:]
        # C coords: X[:,2,:]
        # return num_residues x 6 
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X, [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self.normalize(dX, dim=-1)
        u0 = U[:-2]
        u1 = U[1:-1]
        u2 = U[2:]

        angle = self.compute_dihedrals(u0, u1, u2)
        
        # add phi[0], psi[-1], omega[-1] with value 0
        angle = F.pad(angle, [1, 2]) 
        angle = torch.reshape(angle, [-1, 3])
        angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
        return angle_features
    
    def protein_to_graph(self,pFilePath):
        ''' Convert a protein to a graph '''
        h5File = h5py.File(pFilePath, 'r')
        data = Data()

        amino_types = h5File['amino_types'][()] # size: (n_amino_acids,)
        mask = amino_types == -1
        
        if np.sum(mask) > 0:
            amino_types[mask] = 25 # for amino acid types, set the value of -1 to 25
        atom_amino_id = h5File['atom_amino_id'][()] # size: (n_atom,)
        atom_names = h5File['atom_names'][()] # size: (n_atom,)
        atom_pos = h5File['atom_pos'][()][0] #size: (n_atom,3)

        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = self.get_atom_pos(amino_types, atom_names, atom_amino_id, atom_pos)
        
        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        side_chain_embs = self.side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
        side_chain_embs[torch.isnan(side_chain_embs)] = 0
        data.side_chain_embs = side_chain_embs

        # three backbone torsion angles
        bb_embs = self.bb_embs(torch.cat((torch.unsqueeze(pos_n,1), torch.unsqueeze(pos_ca,1), torch.unsqueeze(pos_c,1)),1))
        bb_embs[torch.isnan(bb_embs)] = 0
        data.bb_embs = bb_embs

        data.x = torch.unsqueeze(torch.tensor(amino_types),1)
        data.coords_ca = pos_ca
        data.coords_n = pos_n
        data.coords_c = pos_c

        assert len(data.x)==len(data.coords_ca)==len(data.coords_n)==len(data.coords_c)==len(data.side_chain_embs)==len(data.bb_embs)

        h5File.close()

        #-------add DSSP information-------
        # Load DSSP file if exists
        try:
            dssp_path = os.path.join(os.path.dirname(pFilePath), "dssp_files", os.path.basename(pFilePath).replace(".hdf5", ".dssp"))
            if os.path.exists(dssp_path):
                dssp_residues = parse_dssp_file(dssp_path)
                ss_lookup = {res['index']: res['ss'] for res in dssp_residues}
                acc_lookup = {res['index']: res['acc'] for res in dssp_residues}
                hbonds = []
                
                n_res = len(amino_types)  # <--- total residues in this HDF5

                for res in dssp_residues:
                    src_idx = res['index'] - 1 # zero-based indexing
                    if not (0 <= src_idx < n_res):
                        continue  # skip residues (source) outside the range
                    for partner, energy in [res['nh_o_1'], res['nh_o_2']]:
                        tgt_idx = src_idx + partner
                        if 0 <= tgt_idx < len(amino_types):
                            hbonds.append((src_idx, tgt_idx, energy))

                # DSSP-derived features
                ss_tensor = torch.zeros((len(amino_types),), dtype=torch.long)
                acc_tensor = torch.zeros((len(amino_types),), dtype=torch.float)
                ss_dict = {'H': 0, 'E': 1, 'T': 2, 'S': 3, 'G': 4, 'B': 5, 'I': 6, 'C': 7} # One-hot encoding


                for i in range(len(amino_types)):
                    ss_label = ss_lookup.get(i + 1, 'C')
                    ss_tensor[i] = ss_dict.get(ss_label, 7)
                    acc_tensor[i] = acc_lookup.get(i + 1, 0.0)

                data.ss = ss_tensor # secondary structure
                data.acc = acc_tensor # accessibility
                data.hbonds = hbonds # for edge construction later

            else:
                data.ss = torch.full((len(amino_types),), 7, dtype=torch.long) # all coil
                data.acc = torch.zeros((len(amino_types),), dtype=torch.float)
                data.hbonds = []


        except Exception as e:
            print(f"[Warning] Failed to parse DSSP for {pFilePath}: {e}")
            data.ss = torch.full((len(amino_types),), 7, dtype=torch.long) # all coil
            data.acc = torch.zeros((len(amino_types),), dtype=torch.float)
            data.hbonds = []
          
        return data

    def process(self):
        import warnings
        from tqdm import tqdm

        print('Beginning Processing ...')

        # ── 1) Load class map (fold → integer label) ─────────────────────────────────
        classes_ = {}
        class_map_path = osp.join(self.root, "class_map.txt")
        with open(class_map_path, 'r', encoding='utf-8') as mFile:
            for line in mFile:
                line = line.rstrip()
                if not line:
                    continue
                k, v = line.split('\t')
                classes_[k] = int(v)

        # ── 2) Read split list using the *original* split name ────────────────────────
        #     (e.g., "test_fold.txt" even if self.split == "test_fold_100")
        split_txt_path = osp.join(self.root, f"{self.split_orig}.txt")
        entries = []  # (domain_id, fold_name, label, h5_path, dssp_path)
        with open(split_txt_path, 'r', encoding='utf-8') as mFile:
            for curLine in mFile:
                curLine = curLine.rstrip()
                if not curLine or curLine.startswith('#'):
                    continue
                parts = curLine.split('\t')
                # Expect lines like: "<domain_id>\t<fold_name>"
                if len(parts) < 2:
                    raise ValueError(f"Malformed line in {split_txt_path}: {curLine}")

                domain_id = parts[0]          # e.g., d1a1ha1
                fold_name = parts[-1]         # e.g., a.1
                if fold_name not in classes_:
                    raise KeyError(f"Fold '{fold_name}' not found in class_map.txt")

                label = classes_[fold_name]
                # IMPORTANT: use split_orig to read raw files on disk
                h5_path   = osp.join(self.root, self.split_orig, domain_id + ".hdf5")
                dssp_path = osp.join(self.root, self.split_orig, "dssp_files", domain_id + ".dssp")
                entries.append((domain_id, fold_name, label, h5_path, dssp_path))

        # ── 3) Filter for existence: require BOTH .hdf5 and .dssp ────────────────────
        kept = []                # (domain_id, fold_name, label, h5_path, dssp_path)
        missing_h5 = []          # domain_id
        missing_dssp = []        # domain_id

        for domain_id, fold_name, label, h5_path, dssp_path in entries:
            if not osp.exists(h5_path):
                missing_h5.append(domain_id)
                continue
            if not osp.exists(dssp_path):
                missing_dssp.append(domain_id)
                continue
            kept.append((domain_id, fold_name, label, h5_path, dssp_path))

        # ── 3b) SCHull-style bucket filter for TEST splits only ───────────────────────
        test_like = self.split_orig in ('test_fold', 'test_superfamily', 'test_family')
        if test_like and (self.n_node >= 0):
            kept_in_bucket = []
            for (dom, fold_name, label, h5_path, dssp_path) in kept:
                # Decide "size" (default: residues; add other modes later if needed)
                if self.size_mode == 'residues':
                    n = self._num_residues_h5(h5_path)
                else:
                    n = self._num_residues_h5(h5_path)
                if n > 0 and self._in_bucket(n, self.n_node):
                    kept_in_bucket.append((dom, fold_name, label, h5_path, dssp_path))
            kept = kept_in_bucket

        # ── 4) Logs (split-scoped to avoid overwriting across buckets) ───────────────
        os.makedirs(self.processed_dir, exist_ok=True)
        lists_dir = osp.join(self.processed_dir, "lists")
        os.makedirs(lists_dir, exist_ok=True)

        kept_ids_path       = osp.join(lists_dir, f"{self.split}_kept_ids.txt")
        skipped_h5_path     = osp.join(lists_dir, f"{self.split}_skipped_missing_h5.txt")
        skipped_dssp_path   = osp.join(lists_dir, f"{self.split}_skipped_missing_dssp.txt")
        filtered_split_path = osp.join(lists_dir, f"{self.split}_filtered.txt")

        with open(kept_ids_path, 'w', encoding='utf-8') as f:
            for domain_id, fold_name, _, _, _ in kept:
                f.write(domain_id + '\n')

        with open(skipped_h5_path, 'w', encoding='utf-8') as f:
            for domain_id in missing_h5:
                f.write(domain_id + '\n')

        with open(skipped_dssp_path, 'w', encoding='utf-8') as f:
            for domain_id in missing_dssp:
                f.write(domain_id + '\n')

        # Emit a filtered split file (domain_id \t fold_name), for reproducibility
        with open(filtered_split_path, 'w', encoding='utf-8') as f:
            for domain_id, fold_name, _, _, _ in kept:
                f.write(f"{domain_id}\t{fold_name}\n")

        print(f"[{self.split}] Kept: {len(kept)}, "
            f"Skipped missing .hdf5: {len(missing_h5)}, "
            f"Skipped missing .dssp: {len(missing_dssp)}")
        if missing_h5:
            print(f"  → Log of missing .hdf5: {skipped_h5_path}")
        if missing_dssp:
            print(f"  → Log of missing .dssp: {skipped_dssp_path}")
        print(f"  → Filtered split written to: {filtered_split_path}")

        # ── 4b) Handle empty buckets gracefully (save an empty dataset) ──────────────
        if len(kept) == 0:
            from torch_geometric.data import Data
            try:
                data, slices = self.collate([])   # works on recent PyG; yields length 0 dataset
            except Exception:
                # Fallback: still try to persist something valid
                data, slices = self.collate([])
            torch.save((data, slices), self.processed_paths[0])
            print(f"[{self.split}] No items in this bucket; saved empty dataset to {self.processed_paths[0]}")
            return

        # ── 5) Build data_list strictly from the kept set ────────────────────────────
        print("Reading the data ...")
        data_list = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for domain_id, _fold_name, label, h5_path, dssp_path in tqdm(kept):
                fileName = domain_id  # stable id

                # Be tolerant to the builder signature:
                
                curProtein = self.protein_to_graph(h5_path)  # preferred: uses DSSP features
                

                curProtein.id = fileName
                curProtein.y  = torch.tensor(label, dtype=torch.long)

                # Keep only valid samples
                if getattr(curProtein, 'x', None) is not None:
                    data_list.append(curProtein)

        # ── 6) (Optional) pre_filter / pre_transform hooks ───────────────────────────
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # ── 7) Collate & persist ─────────────────────────────────────────────────────
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f'Done! Saved {len(data_list)} proteins to {self.processed_paths[0]}')





# ====== TESTS / DEMO ======
import os
from collections import Counter
import torch


def test_schull_logic():
    root = "/workspace/workspace/CD_Conv_Plus_ProNet/dataset/data/HomologyTAPE"

    # 1. Test TRAIN split (should keep *all* valid files, no bucket filtering)
    train_dataset = FOLDdataset(root=root, split="training")
    print(f"[TRAIN] Number of proteins: {len(train_dataset)}")

    # Check how many DSSP features exist
    ss_counts = Counter()
    for data in train_dataset:
        ss_counts.update(data.ss.tolist())
    print("[TRAIN] SS distribution across all proteins:", ss_counts)

    # 2. Test TEST split without bucket (full test set)
    test_dataset = FOLDdataset(root=root, split="test_fold")
    print(f"[TEST full] Number of proteins: {len(test_dataset)}")

    # 3. Test TEST split *with* SCHull-style bucketing
    for bucket_idx in range(3):  # try first 3 buckets
        test_bucket = FOLDdataset(root=root, split="test_fold", n_node=bucket_idx)
        print(f"[TEST bucket {bucket_idx}] {len(test_bucket)} proteins saved in {test_bucket.processed_paths[0]}")

    # 4. Verify logs exist
    logs_dir = os.path.join(test_dataset.processed_dir, "lists")
    print("\n[LOG FILES]")
    if os.path.isdir(logs_dir):
        for fname in os.listdir(logs_dir):
            print("  ", fname)
    else:
        print("  (no logs found yet)")

def test_single_protein():
    """Quick unit test: load one HDF5 + DSSP and print its features."""
    h5_path = "/workspace/workspace/CD_Conv_Plus_ProNet/dataset/data/HomologyTAPE/training/d1a1ha1.hdf5"
    dataset = FOLDdataset(root="/workspace/workspace/CD_Conv_Plus_ProNet/dataset/data/HomologyTAPE",
                          split="training")
    data = dataset.protein_to_graph(h5_path)

    print("✅ Loaded protein graph successfully.\n")
    print("Available attributes in `data` object:", data)

    print("\n📌 DSSP Secondary Structure Labels (ss):")
    print(data.ss)

    print("\n📌 DSSP Solvent Accessibility (acc):")
    print(data.acc)

    print("\n📌 Extracted Hydrogen Bonds (hbonds):")
    print(f"Total hydrogen bonds: {len(data.hbonds)}")
    print(f"First 10 hydrogen bonds: {data.hbonds[:10]}")
    for src, tgt, energy in data.hbonds[:10]:
        print(f"{src} → {tgt}, energy = {energy:.3f}")

    print(f"\n#residues in HDF5: {len(data.x)}, #DSSP labels: {len(data.ss)}")
    unique, counts = torch.unique(data.ss, return_counts=True)
    print("SS distribution:", dict(zip(unique.tolist(), counts.tolist())))

if __name__ == '__main__':
    # Run both tests
    # test_single_protein()
    test_schull_logic()
