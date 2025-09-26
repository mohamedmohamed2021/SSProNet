# Code to run xperiments on Fold and EC datasets in our paper 
# "Learning Hierarchical Protein Representations via Complete 3D Graph Networks" 
# (https://openreview.net/forum?id=9X-hgLDLYkQ)

##################################### Default hyperparameters for ECdataset #####################################
# device=0
# dataset='func'
# dataset_path='dataset/' # make sure that the folder 'ProtFunct' is under this path
# cutoff=10.0
# batch_size=32
# eval_batch_size=32

# level='backbone'
# num_blocks=4
# hidden_channels=128
# out_channels=384

# epochs=400
# lr=0.0005
# lr_decay_step_size=60
# lr_decay_factor=0.5

# mask_aatype=0.2
# dropout=0.3
# num_workers=5

# python run_ProNet.py --device $device --dataset $dataset --dataset_path $dataset_path --cutoff $cutoff \
# --batch_size $batch_size --eval_batch_size $eval_batch_size \
# --level $level --num_blocks $num_blocks --hidden_channels $hidden_channels --out_channels $out_channels \
# --epochs $epochs \
# --lr $lr --lr_decay_step_size $lr_decay_step_size --lr_decay_factor $lr_decay_factor \
# --mask_aatype $mask_aatype --dropout $dropout \
# --num_workers $num_workers \
# --mask --noise --deform --euler_noise --data_augment_eachlayer

##################################### Default hyperparameters for ECdataset #####################################
# device=0
# dataset='fold'
# dataset_path='dataset/' # make sure that the folder 'HomologyTAPE' is under this path
# cutoff=10.0
# batch_size=32
# eval_batch_size=32

# level='backbone'
# num_blocks=4
# hidden_channels=128
# out_channels=1195

# epochs=1000
# lr=0.0005
# lr_decay_step_size=150
# lr_decay_factor=0.5

# mask_aatype=0.2
# dropout=0.3
# num_workers=5

# python run_ProNet.py --device $device --dataset $dataset --dataset_path $dataset_path --cutoff $cutoff \
# --batch_size $batch_size --eval_batch_size $eval_batch_size \
# --level $level --num_blocks $num_blocks --hidden_channels $hidden_channels --out_channels $out_channels \
# --epochs $epochs \
# --lr $lr --lr_decay_step_size $lr_decay_step_size --lr_decay_factor $lr_decay_factor \
# --mask_aatype $mask_aatype --dropout $dropout \
# --num_workers $num_workers \
# --mask --noise --deform --euler_noise --data_augment_eachlayer

import os
import csv
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import argparse
import random

import torch
import torch.optim as optim
from torch import nn 
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

# Old method
# from method.pronet import ProNet 

# New CompleteCD_Conv method
#from method.Complete_CD_Conv import ProNet

# new method: SeqRadNet
#from method.SeqRadNet_Nature import SeqRadNet

# new method: SeqRadNet_ProNet
#from method.SeqRadNet_ProNet import SeqRadNet

# new method: SeqRadNet_CoupleNet
#from method.SeqRadNet_CoupleNet import SeqRadNet

#new method:SSProNet
from method.SSProNet_new_ec import SSProNet
from dataset.FOLDdataset import FOLDdataset
from dataset.ECdataset import ECdataset
from torch_geometric.data import DataLoader


import warnings
warnings.filterwarnings("ignore")

criterion = nn.CrossEntropyLoss()

num_fold = 1195
num_func = 384

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For extra reproducibility (important!)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args, model, loader, optimizer, device):
    model.train()

    loss_accum = 0
    preds = []
    functions = []
    for step, batch in enumerate(tqdm(loader, disable=args.disable_tqdm)):
        if args.mask:
            # random mask node aatype
            mask_indice = torch.tensor(np.random.choice(batch.num_nodes, int(batch.num_nodes * args.mask_aatype), replace=False))
            batch.x[:, 0][mask_indice] = 25
        if args.noise:
            # add gaussian noise to atom coords
            gaussian_noise = torch.clip(torch.normal(mean=0.0, std=0.1, size=batch.coords_ca.shape), min=-0.3, max=0.3)
            batch.coords_ca += gaussian_noise
            if args.level != 'aminoacid':
                batch.coords_n += gaussian_noise
                batch.coords_c += gaussian_noise
        if args.deform:
            # Anisotropic scale
            deform = torch.clip(torch.normal(mean=1.0, std=0.1, size=(1, 3)), min=0.9, max=1.1)
            batch.coords_ca *= deform
            if args.level != 'aminoacid':
                batch.coords_n *= deform
                batch.coords_c *= deform
        batch = batch.to(device)
                     
        try:
            pred = model(batch) 
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): 
                print('\n forward error \n')
                raise(e)
            else:
                print('OOM')
            torch.cuda.empty_cache()
            continue
        preds.append(torch.argmax(pred, dim=1))        
        function = batch.y
        functions.append(function)

        # ------- Debugging invalid labels before loss
        if torch.any(function >= args.out_channels) or torch.any(function < 0):
            print("❌ Invalid label detected!")
            print("Max label:", function.max().item(), " Min label:", function.min().item())
            print("Expected range: 0 to", args.out_channels - 1)
            print("Batch IDs:", [d.id for d in batch.to_data_list()])
            raise ValueError("Invalid class index in labels")
        #---------------------------------------------------
        
        optimizer.zero_grad()
        loss = criterion(pred, function)
        loss.backward()
        optimizer.step()

        loss_accum += loss.item()        

    functions = torch.cat(functions, dim=0)
    preds = torch.cat(preds, dim=0)
    acc = torch.sum(preds==functions)/functions.shape[0]
    
    return loss_accum/(step + 1), acc.item()

def evaluation(args, model, loader, device):
    model.eval()

    loss_accum, preds, functions = 0.0, [], []
    steps = 0
    for step, batch in enumerate(loader):
        steps += 1
        batch = batch.to(device)
        try:
            pred = model(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                print('\n forward error \n'); raise
            print('evaluation OOM'); torch.cuda.empty_cache(); continue

        preds.append(torch.argmax(pred, dim=1))
        function = batch.y
        functions.append(function)
        loss_accum += criterion(pred, function).item()

    if steps == 0:
        return float('nan'), float('nan')  # empty bucket

    functions = torch.cat(functions, dim=0)
    preds = torch.cat(preds, dim=0)
    acc = torch.sum(preds == functions) / functions.shape[0]
    return loss_accum / steps, acc.item()

# Helper to evaluate all buckets and pick the best
def eval_buckets_and_pick_best(args, model, loaders, device, ct_lst):
    """Evaluate a list of bucket loaders and return:
       - per_bucket: list of dicts [{'cutoff': hi, 'loss': ..., 'acc': ...}, ...]
       - best: dict with the best by accuracy (ties broken by higher cutoff index)
    """
    per_bucket = []
    best = None
    for i, loader in enumerate(loaders):
        # Skip truly empty datasets fast (PyG DataLoader exposes .dataset)
        ds_len = len(loader.dataset) if hasattr(loader, 'dataset') else None
        if ds_len is not None and ds_len == 0:
            loss, acc = float('nan'), float('nan')
        else:
            loss, acc = evaluation(args, model, loader, device)
        item = {'cutoff': ct_lst[i], 'loss': loss, 'acc': acc, 'n': ds_len}
        per_bucket.append(item)
        if not np.isnan(acc):
            if best is None or acc > best['acc']:
                best = item
    return per_bucket, best

    
def main():
    
    """# Set global seed
    seed = 42  # or any integer you like
    set_seed(seed)"""


    ### Args
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=9, help='Device to use')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of workers in Dataloader')

    ### Data
    parser.add_argument('--dataset', type=str, default='func', help='Func or fold')
    parser.add_argument('--dataset_path', type=str, default='./dataset/data', help='path to load and process the data')
    
    # data augmentation tricks, see appendix E in the paper (https://openreview.net/pdf?id=9X-hgLDLYkQ)
    parser.add_argument('--mask', action='store_true', help='Random mask some node type')
    parser.add_argument('--noise', action='store_true', help='Add Gaussian noise to node coords')
    parser.add_argument('--deform', action='store_true', help='Deform node coords')
    parser.add_argument('--data_augment_eachlayer', action='store_true', help='Add Gaussian noise to features')
    parser.add_argument('--euler_noise', action='store_true', help='Add Gaussian noise Euler angles')
    parser.add_argument('--mask_aatype', type=float, default=0.1, help='Random mask aatype to 25(unknown:X) ratio')
    
    ### Model
    parser.add_argument('--level', type=str, default='backbone', help='Choose from \'aminoacid\', \'backbone\', and \'allatom\' levels')
    parser.add_argument('--num_blocks', type=int, default=4, help='Model layers')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--out_channels', type=int, default=1195, help='Number of classes, 1195 for the fold data, 384 for the ECdata')
    parser.add_argument('--fix_dist', action='store_true')  
    parser.add_argument('--cutoff', type=float, default=10, help='Distance constraint for building the protein graph') 
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout')

    ### Training hyperparameter
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--lr_decay_step_size', type=int, default=60, help='Learning rate step size')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='Learning rate factor') 
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight Decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size during training')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size')
    
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--save_dir', type=str, default=None, help='Trained model path')

    parser.add_argument('--disable_tqdm', default=False, action='store_true')

    parser.add_argument('--use_schull_buckets', action='store_true',
                    help='Evaluate test splits per size bucket and report best cutoff (SCHull)')
    parser.add_argument('--ct_lst', type=str, default="50,100,150,200,250,300,400,500,2000",
                    help='Comma-separated upper bounds for residue-count buckets')
    parser.add_argument("--size_mode", type=str, default="residues",
                    help="Size measure for SCHull bucketing: 'residues', 'graph_nodes', or 'ca_atoms'")

    args = parser.parse_args()
    ct_lst = [int(x) for x in args.ct_lst.split(',')]
    print(args)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ##### load datasets
    print('Loading Train & Val & Test Data...')
    if args.dataset == 'func':
        try:
            train_set = ECdataset(root=args.dataset_path + '/ProtFunct', split='Train')
            val_set = ECdataset(root=args.dataset_path + '/ProtFunct', split='Val')
            test_set = ECdataset(root=args.dataset_path + '/ProtFunct', split='Test')
        except FileNotFoundError: 
            print('\n Please download data firstly, following https://github.com/divelab/DIG/tree/dig-stable/dig/threedgraph/dataset#ecdataset-and-folddataset and https://github.com/phermosilla/IEConv_proteins#download-the-preprocessed-datasets \n')
            raise(FileNotFoundError)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        print('Done!')
        print('Train, val, test:', train_set, val_set, test_set)

    elif args.dataset == 'fold':
        # ---- load data -------
        try:
            ct_lst = [int(x) for x in args.ct_lst.split(',')]  # parse ct_lst from args

            train_set = FOLDdataset(
                root=args.dataset_path + '/HomologyTAPE',
                split='training',
                size_mode=args.size_mode
            )
            val_set = FOLDdataset(
                root=args.dataset_path + '/HomologyTAPE',
                split='validation',
                size_mode=args.size_mode
            )
        
            # ---- Build bucketed test loader -------
            if args.use_schull_buckets:
                test_fold_sets = [
                    FOLDdataset(root=args.dataset_path + '/HomologyTAPE',
                                split='test_fold', n_node=i,
                                ct_lst=ct_lst, size_mode=args.size_mode)
                    for i in range(len(ct_lst))
                ]
                test_super_sets = [
                    FOLDdataset(root=args.dataset_path + '/HomologyTAPE',
                                split='test_superfamily', n_node=i,
                                ct_lst=ct_lst, size_mode=args.size_mode)
                    for i in range(len(ct_lst))
                ]
                test_family_sets = [
                    FOLDdataset(root=args.dataset_path + '/HomologyTAPE',
                                split='test_family', n_node=i,
                                ct_lst=ct_lst, size_mode=args.size_mode)
                    for i in range(len(ct_lst))
                ]
            # ------ regular test loader -------
            else:
                test_fold = FOLDdataset(
                    root=args.dataset_path + '/HomologyTAPE',
                    split='test_fold',
                    size_mode=args.size_mode
                )
                test_super = FOLDdataset(
                    root=args.dataset_path + '/HomologyTAPE',
                    split='test_superfamily',
                    size_mode=args.size_mode
                )
                test_family = FOLDdataset(
                    root=args.dataset_path + '/HomologyTAPE',
                    split='test_family',
                    size_mode=args.size_mode
                )
        
        except FileNotFoundError: 
            print('\n Please download data firstly, following '
                'https://github.com/divelab/DIG/tree/dig-stable/dig/threedgraph/dataset#ecdataset-and-folddataset '
                'and https://github.com/phermosilla/IEConv_proteins#download-the-preprocessed-datasets \n')
            raise

        # ---- loaders ----
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        ####---- bucketed test loaders ----
        if args.use_schull_buckets:
            test_fold_loaders   = [DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers) for ds in test_fold_sets]
            test_super_loaders  = [DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers) for ds in test_super_sets]
            test_family_loaders = [DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers) for ds in test_family_sets]
        ###---- regular test loaders ----
        else:
            test_fold_loader   = DataLoader(test_fold, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
            test_super_loader  = DataLoader(test_super, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
            test_family_loader = DataLoader(test_family, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        
        print('Done!')

    else:
        print('not supported dataset')
    

    # Initialize model: SSProNet
    model = SSProNet(
        level=args.level,
        num_blocks=args.num_blocks,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        mid_emb=32,
        num_radial=6,
        num_spherical=2,
        cutoff=args.cutoff,
        max_num_neighbors=32,
        int_emb_layers=2,
        out_layers=2,
        num_pos_emb=16,
        dropout=args.dropout,
        data_augment_eachlayer=args.data_augment_eachlayer,
        euler_noise=args.euler_noise
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor)
    
    
    if args.continue_training:
        save_dir = args.save_dir
        csv_log_path = os.path.join(save_dir, "epoch_metrics.csv")
        with open(csv_log_path, mode='w', newline='') as f:
            writer_csv = csv.writer(f)
            if args.dataset == 'func':
                writer_csv.writerow(["Epoch", "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc", "Test_Loss", "Test_Acc", "Epoch_Time", "Train_Time", "Test_Time"])
            elif args.dataset == 'fold':
                writer_csv.writerow(["Epoch", "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc", "Test_Fold_Loss", "Test_Fold_Acc", "Test_Super_Loss", "Test_Super_Acc", "Test_Family_Loss", "Test_Family_Acc", "Epoch_Time", "Train_Time", "Test_Time"])
        
        bucket_csv_path = os.path.join(save_dir, "bucket_metrics.csv")
        with open(bucket_csv_path, mode='w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["Epoch", "Split", "Cutoff", "N_Samples", "Acc", "Loss"])

        checkpoint = torch.load(save_dir + '/best_val.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        save_dir = './trained_models_{dataset}/{level}/layer{num_blocks}_cutoff{cutoff}_hidden{hidden_channels}_batch{batch_size}_lr{lr}_{lr_decay_factor}_{lr_decay_step_size}_dropout{dropout}__{time}'.format(
            dataset=args.dataset, level=args.level, 
            num_blocks=args.num_blocks, cutoff=args.cutoff, hidden_channels=args.hidden_channels, batch_size=args.batch_size, 
            lr=args.lr, lr_decay_factor=args.lr_decay_factor, lr_decay_step_size=args.lr_decay_step_size, dropout=args.dropout, time=datetime.now())
        print('saving to...', save_dir)
        os.makedirs(save_dir, exist_ok=True)

        csv_log_path = os.path.join(save_dir, "epoch_metrics.csv")
        with open(csv_log_path, mode='w', newline='') as f:
            writer_csv = csv.writer(f)
            if args.dataset == 'func':
                writer_csv.writerow(["Epoch", "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc", "Test_Loss", "Test_Acc", "Epoch_Time", "Train_Time", "Test_Time"])
            elif args.dataset == 'fold':
                writer_csv.writerow(["Epoch", "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc", "Test_Fold_Loss", "Test_Fold_Acc", "Test_Super_Loss", "Test_Super_Acc", "Test_Family_Loss", "Test_Family_Acc", "EpochTime", "TrainTime", "TestTime"])
        
        bucket_csv_path = os.path.join(save_dir, "bucket_metrics.csv")
        with open(bucket_csv_path, mode='w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["Epoch", "Split", "Cutoff", "N_Samples", "Acc", "Loss"])
        start_epoch = 1
        
    num_params = sum(p.numel() for p in model.parameters()) 
    print('num_parameters:', num_params)

    if args.dataset == 'func':
        writer = SummaryWriter(log_dir=save_dir)
        best_val_acc = 0
        test_at_best_val_acc = 0

        for epoch in range(start_epoch, args.epochs+1):
            print('==== Epoch {} ===='.format(epoch))
            t_start = time.perf_counter()

            train_loss, train_acc = train(args, model, train_loader, optimizer, device)
            t_end_train = time.perf_counter()
            val_loss, val_acc = evaluation(args, model, val_loader, device)
            t_start_test = time.perf_counter()
            test_loss, test_acc = evaluation(args, model, test_loader, device)
            t_end_test = time.perf_counter()

            epoch_time = t_end_test - t_start
            train_time = t_end_train - t_start
            test_time = t_end_test - t_start_test

            if not save_dir == "" and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if not save_dir == "" and val_acc > best_val_acc:
                print('Saving best val checkpoint ...')
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }
                torch.save(checkpoint, os.path.join(save_dir, 'best_val.pt'))
                best_val_acc = val_acc
                test_at_best_val_acc = test_acc

            print('Train: Loss:{:.6f} Acc:{:.4f}, Validation: Loss:{:.6f} Acc:{:.4f}, Test: Loss:{:.6f} Acc:{:.4f}, '
                'test_acc@best_val:{:.4f}, time:{:.2f}s, train_time:{:.2f}s, test_time:{:.2f}s'.format(
                train_loss, train_acc, val_loss, val_acc,
                test_loss, test_acc, test_at_best_val_acc,
                epoch_time, train_time, test_time))

            # TensorBoard logging
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_acc', val_acc, epoch)
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)
            writer.add_scalar('test_at_best_val_acc', test_at_best_val_acc, epoch)

            # CSV logging
            with open(csv_log_path, mode='a', newline='') as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow([
                    epoch, train_loss, train_acc, val_loss, val_acc,
                    test_loss, test_acc,
                    epoch_time, train_time, test_time
                ])

            scheduler.step()

        writer.close()

        # Save last model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, os.path.join(save_dir, f"epoch{epoch}.pt"))

    if args.dataset == 'fold':
        writer = SummaryWriter(log_dir=save_dir)
        best_val_acc = 0
        test_fold_at_best_val_acc = 0
        test_super_at_best_val_acc = 0
        test_family_at_best_val_acc = 0

        for epoch in range(start_epoch, args.epochs+1):
            print('==== Epoch {} ===='.format(epoch))
            t_start = time.perf_counter()

            train_loss, train_acc = train(args, model, train_loader, optimizer, device)
            t_end_train = time.perf_counter()
            val_loss, val_acc = evaluation(args, model, val_loader, device)
            t_start_test = time.perf_counter()

            if args.use_schull_buckets:
                fold_buckets, best_fold   = eval_buckets_and_pick_best(args, model, test_fold_loaders,  device, ct_lst)
                super_buckets, best_super = eval_buckets_and_pick_best(args, model, test_super_loaders, device, ct_lst)
                family_buckets, best_family = eval_buckets_and_pick_best(args, model, test_family_loaders, device, ct_lst)
                # For compact printing/logging, define loss/acc as the best ones:
                test_fold_loss,  test_fold_acc  = best_fold['loss'],  best_fold['acc']
                test_super_loss, test_super_acc = best_super['loss'], best_super['acc']
                test_family_loss, test_family_acc = best_family['loss'], best_family['acc']
            else:
                test_fold_loss,  test_fold_acc  = evaluation(args, model, test_fold_loader,  device)
                test_super_loss, test_super_acc = evaluation(args, model, test_super_loader, device)
                test_family_loss, test_family_acc = evaluation(args, model, test_family_loader, device)

            t_end_test = time.perf_counter()

            epoch_time = t_end_test - t_start
            train_time = t_end_train - t_start
            test_time  = t_end_test - t_start_test

            if not save_dir == "" and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if not save_dir == "" and val_acc > best_val_acc:
                print('Saving best val checkpoint ...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()}
                torch.save(checkpoint, save_dir + '/best_val.pt')
                best_val_acc = val_acc
                test_fold_at_best_val_acc   = test_fold_acc
                test_super_at_best_val_acc  = test_super_acc
                test_family_at_best_val_acc = test_family_acc

            # ----- pretty printing
            if args.use_schull_buckets:
                def _fmt(per_bucket):
                    return '; '.join([f"≤{d['cutoff']}: n={d['n']}, acc={d['acc']:.4f}" if not np.isnan(d['acc']) else f"≤{d['cutoff']}: n=0"
                                    for d in per_bucket])
                print('[BUCKETS] fold   ->', _fmt(fold_buckets))
                print('[BUCKETS] super  ->', _fmt(super_buckets))
                print('[BUCKETS] family ->', _fmt(family_buckets))
                # Also log buckets to CSV
                with open(bucket_csv_path, mode='a', newline='') as f:
                    writer_csv = csv.writer(f)
                    for d in fold_buckets:
                        writer_csv.writerow([epoch, "fold", d['cutoff'], d['n'], d['acc'], d['loss']])
                    for d in super_buckets:
                        writer_csv.writerow([epoch, "superfamily", d['cutoff'], d['n'], d['acc'], d['loss']])
                    for d in family_buckets:
                        writer_csv.writerow([epoch, "family", d['cutoff'], d['n'], d['acc'], d['loss']])


            print('Train: Loss:{:.6f} Acc:{:.4f}, Validation: Loss:{:.6f} Acc:{:.4f}, '
                'Test_fold(best): Loss:{:.6f} Acc:{:.4f}, Test_super(best): Loss:{:.6f} Acc:{:.4f}, Test_family(best): Loss:{:.6f} Acc:{:.4f}, '
                'test_fold_acc@best_val:{:.4f}, test_super_acc@best_val:{:.4f}, test_family_acc@best_val:{:.4f}, '
                'time:{:.2f}s, train_time:{:.2f}s, test_time:{:.2f}s'.format(
                train_loss, train_acc, val_loss, val_acc,
                test_fold_loss, test_fold_acc, test_super_loss, test_super_acc, test_family_loss, test_family_acc,
                test_fold_at_best_val_acc, test_super_at_best_val_acc, test_family_at_best_val_acc,
                epoch_time, train_time, test_time))

            # TensorBoard (always log the *best* per-cutoff metrics)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc',  train_acc,  epoch)
            writer.add_scalar('val_loss',   val_loss,   epoch)
            writer.add_scalar('val_acc',    val_acc,    epoch)
            writer.add_scalar('test_fold_acc_best',   test_fold_acc,   epoch)
            writer.add_scalar('test_super_acc_best',  test_super_acc,  epoch)
            writer.add_scalar('test_family_acc_best', test_family_acc, epoch)

            # CSV logging (keep your columns; write the best)
            with open(csv_log_path, mode='a', newline='') as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow([epoch, train_loss, train_acc, val_loss, val_acc,
                                    test_fold_loss, test_fold_acc, test_super_loss, test_super_acc, test_family_loss, test_family_acc,
                                    epoch_time, train_time, test_time])

            scheduler.step()

        writer.close()
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
        torch.save(checkpoint, save_dir + "/epoch{}.pt".format(epoch))


if __name__ == "__main__":
    main()