# SSProNet: Secondary-Structure-aware Graph Neural Network for Protein Representation Learning

This repository contains the implementation used in our paper **“SSProNet: Secondary Structure aware Graph Neural Network for Protein Representation Learning.”**  
It trains and evaluates SSProNet on three datasets: **FOLD**, **ProtFunct (EC)**, and **LBA**.

---

## Table of contents
- [Requirements](#requirements)
- [Installation](#installation)
  - [Python env + packages](#python-env--packages)
  - [Install DSSP from the included folder](#install-dssp-from-the-included-folder)
- [Datasets](#datasets)
  - [FOLD (HomologyTAPE)](#fold-homologytape)
  - [ProtFunct (EC)](#protfunct-ec)
  - [LBA](#lba)
- [Training & evaluation](#training--evaluation)
- [Outputs & Logging](#outputs--logging)
- [Repo structure](#repo-structure)
- [Citation](#citation)
- [License](#license)

---

## Requirements

- **Python** 3.10+ (3.11 also OK)
- **PyTorch**, **PyTorch Geometric**, **torch-scatter**, **torch-sparse**
- **DSSP** (secondary structure & hydrogen bonds)

> 🔧 **Versions we used in experiments**  
> PyTorch: **2.8.0+cu128**  
> (Install a build compatible with your CUDA; see below for the pattern.)

---

## Installation

### Python env + packages

```bash
# (recommended) conda
conda create -n sspronet python=3.10 -y
conda activate sspronet

# ---- Install PyTorch (match your CUDA) ----
# Example for the version we've uses which is "torch==2.8.0+cu128" (adjust to your system if needed)
pip install torch==2.8.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ---- Install PyG stack (match your torch & CUDA) ----
pip install \
  torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# ---- Common Python packages ----
pip install numpy tqdm tensorboard pandas
```

### Install DSSP from the included folder
- We already vendor the DSSP source at dssp-2.3.0/. No need to fetch it elsewhere.

- Here is how to install it (we used Linux (Ubuntu) os ):  

```bash 
sudo apt-get update
sudo apt-get install -y build-essential cmake

cd dssp-2.3.0
mkdir -p build local
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=./local
cmake --build build -j
cmake --install build

# Verify:
./local/bin/mkdssp --version

```
- This produce the binary at (all conversion scripts already point to this path) :
```bash
dssp-2.3.0/local/bin/mkdssp
```

## Datasets

We tested on FOLD, ProtFunct (EC), and LBA.  Each has a dedicated dataloader in ./dataset.  

- Download sources:
    - FOLD / ProtFunct: https://github.com/phermosilla/IEConv_proteins#download-the-preprocessed-datasets
    - LBA : https://github.com/hehefan/Continuous-Discrete-Convolution

- Place them under : 
```bash
dataset/data/HomologyTAPE/   # FOLD
dataset/data/ProtFunct/      # EC / ProtFunct
dataset/data/LBA/            # LBA

```

### FOLD (HomologyTAPE)
Convert HDF5 → PDB → DSSP:
```bash 
bash dataset/data/HomologyTAPE/convert_all_hdf5_to_pdb.sh
bash dataset/data/HomologyTAPE/convert_all_pdb_to_dssp.sh
```

After conversion, the folder looks like : 
```bash 
dataset/data/HomologyTAPE/
├─ training/
├─ validation/
├─ test_fold/
├─ test_family/
├─ test_superfamily/
├─ processed/
├─ ... (split lists, logs, etc.)
```
Each split contains:
```bash
<split>/pdb_files/*.pdb
<split>/dssp_files/*.dssp
```

### ProtFunct (EC)
Convert HDF5 → PDB → DSSP:
```bash
bash dataset/data/ProtFunct/convert_all_hdf5_into_pdb.sh
bash dataset/data/ProtFunct/convert_all_pdb_into_dssp.sh
```
After conversion, the folder looks like:
```bash
dataset/data/ProtFunct/
├─ data/  
├─ dssp_files/Train|Val|Test  
├─ processed/  
├─ training.txt
├─ validation.txt
├─ testing.txt
├─ chain_functions.txt
├─ unique_functions.txt
└─ poolings/
```
### LBA
Convert pocket PDB → DSSP:
```bash
bash dataset/data/LBA/convert_lba_pdb_to_dssp.sh
```
After conversion: 
```bash
dataset/data/LBA/
├─ data/
├─ pdb_pocket/
├─ dssp_pocket/
├─ indices/
├─ targets/
├─ convert_lba_pdb_to_dssp.sh
└─ pdb_vs_dssp.py
```

## Training & evaluation
- EC (ProtFunct): 
```bash 
bash run_ProNet_EC.sh 
```
- FOLD: 
```bash
bash run_ProNet_FOLD.sh
```
- LBA: 
```bash
bash run_ProNet_LBA.sh
```
## Repo structure
```bash
SSProNet/
├─ dataset/
│  ├─ __init__.py
│  ├─ ECdataset.py          # ProtFunct dataloader + DSSP integration
│  ├─ FOLDdataset.py        # FOLD dataloader + DSSP + SCHull buckets
│  ├─ LBAdataset.py         # LBA dataloader + optional DSSP
│  └─ data/
│     ├─ HomologyTAPE/      # FOLD + conversion scripts
│     ├─ ProtFunct/         # EC + conversion scripts
│     └─ LBA/               # LBA + conversion script
├─ dssp-2.3.0/              # vendored DSSP; build installs to ./local/bin/mkdssp
├─ method/
│  ├─ __init__.py
│  └─ SSProNet_new_ec.py    # SSProNet (current variant used by EC runner)
├─ tools_LBA/
│  └─ lba_export_pdbs.py    # helper for LBA pocket exports
├─ run_ProNet.py            # unified runner for EC/FOLD (with SCHull for FOLD)
├─ run_ProNet_LBA.py        # runner for LBA
├─ run_ProNet_EC.sh         # convenience launcher for EC
├─ run_ProNet_FOLD.sh       # convenience launcher for FOLD
└─ run_ProNet_LBA.sh        # convenience launcher for LBA

```

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License**.  



