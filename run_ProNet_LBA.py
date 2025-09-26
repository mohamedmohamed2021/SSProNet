import os, csv, time, argparse, random
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from method.SSProNet_new_ec import SSProNet
from dataset.LBAdataset import LBADataset_SSProNet


# ---- utils ----
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def _corr(x, y):
    xv = x - x.mean(); yv = y - y.mean()
    return (xv @ yv) / (xv.norm() * yv.norm() + 1e-8)


def _spearman(x, y):
    xr = torch.argsort(torch.argsort(x)); yr = torch.argsort(torch.argsort(y))
    return _corr(xr.float(), yr.float())


def _kendall(x, y):
    n = x.numel(); num_c = 0; num_d = 0
    for i in range(n - 1):
        sxi = torch.sign(x[i] - x[i+1:]); syi = torch.sign(y[i] - y[i+1:])
        num_c += (sxi * syi > 0).sum().item()
        num_d += (sxi * syi < 0).sum().item()
    den = n * (n - 1) / 2 + 1e-8
    return (num_c - num_d) / den


def rmse(pred, target): 
    return torch.sqrt(torch.mean((pred - target) ** 2))


# ---- train / eval ----
def train_one_epoch(model, loader, optimizer, criterion, device, args):
    model.train(); loss_sum = 0; count = 0
    for batch in tqdm(loader, disable=args.disable_tqdm):
        # --- augmentations ---
        if args.mask:
            mask_ind = torch.tensor(np.random.choice(batch.num_nodes, int(batch.num_nodes*args.mask_aatype), replace=False))
            batch.x[:, 0][mask_ind] = 25
        if args.noise:
            noise = torch.clip(torch.normal(mean=0.0, std=0.1, size=batch.coords_ca.shape), -0.3, 0.3)
            batch.coords_ca += noise
            if args.level != 'aminoacid':
                batch.coords_n += noise; batch.coords_c += noise
        if args.deform:
            deform = torch.clip(torch.normal(mean=1.0, std=0.1, size=(1,3)), 0.9, 1.1)
            batch.coords_ca *= deform
            if args.level != 'aminoacid':
                batch.coords_n *= deform; batch.coords_c *= deform
        # ----------------------
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch).squeeze(-1)          # [B]
        loss = criterion(pred, batch.label)
        loss.backward(); optimizer.step()
        loss_sum += loss.item() * batch.label.numel()
        count += batch.label.numel()
    return loss_sum / count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); all_p, all_y = [], []
    loss_sum = 0; count = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).squeeze(-1)
        loss = criterion(pred, batch.label)
        loss_sum += loss.item() * batch.label.numel()
        count += batch.label.numel()
        all_p.append(pred.cpu()); all_y.append(batch.label.cpu())
    p = torch.cat(all_p); y = torch.cat(all_y)
    return (loss_sum / count, rmse(p, y).item(), _corr(p, y).item(),
            _spearman(p, y).item(), _kendall(p, y))


# ---- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=True, help='.../LBA/data/')
    ap.add_argument('--dssp_path', required=True, help='path to DSSP root (train/ val/ test/)')
    ap.add_argument('--level', default='backbone', choices=['aminoacid','backbone','allatom'])
    ap.add_argument('--num_blocks', type=int, default=4)
    ap.add_argument('--hidden_channels', type=int, default=128)
    ap.add_argument('--out_channels', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--cutoff', type=float, default=10.0)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--eval_batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--mask', action='store_true'); ap.add_argument('--mask_aatype', type=float, default=0.1)
    ap.add_argument('--noise', action='store_true'); ap.add_argument('--deform', action='store_true')
    ap.add_argument('--disable_tqdm', action='store_true')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()

    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ----- dataset -----
    trainset = LBADataset_SSProNet(os.path.join(args.dataset_path, 'train'),
                                   dssp_root=os.path.join(args.dssp_path, 'train'))
    valset   = LBADataset_SSProNet(os.path.join(args.dataset_path, 'val'),
                                   dssp_root=os.path.join(args.dssp_path, 'val'))
    testset  = LBADataset_SSProNet(os.path.join(args.dataset_path, 'test'),
                                   dssp_root=os.path.join(args.dssp_path, 'test'))

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(valset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    # ----- model -----
    model = SSProNet(level=args.level, num_blocks=args.num_blocks,
                     hidden_channels=args.hidden_channels, out_channels=args.out_channels,
                     cutoff=args.cutoff, dropout=args.dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)

    # ----- logging dirs -----
    save_dir = './trained_models_lba/{level}_layer{num_blocks}_hidden{hidden_channels}_batch{batch_size}_lr{lr}_{time}'.format(
        level=args.level, num_blocks=args.num_blocks, hidden_channels=args.hidden_channels,
        batch_size=args.batch_size, lr=args.lr, time=datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    print("Saving logs to:", save_dir)

    csv_log_path = os.path.join(save_dir, "epoch_metrics.csv")
    with open(csv_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch","Train_Loss","Val_Loss","Val_RMSE","Val_r","Val_s","Val_k",
                         "Test_Loss","Test_RMSE","Test_r","Test_s","Test_k",
                         "Epoch_Time","Train_Time","Test_Time"])

    writer_tb = SummaryWriter(log_dir=save_dir)

    # ----- training loop -----
    best_val_rmse, best_state = float('inf'), None
    for ep in range(1, args.epochs+1):
        t_start = time.perf_counter()

        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args)
        t_end_train = time.perf_counter()
        val_loss, val_rmse, val_r, val_s, val_k = evaluate(model, val_loader, criterion, device)
        t_start_test = time.perf_counter()
        te_loss, te_rmse, te_r, te_s, te_k = evaluate(model, test_loader, criterion, device)
        t_end_test = time.perf_counter()

        epoch_time = t_end_test - t_start
        train_time = t_end_train - t_start
        test_time  = t_end_test - t_start_test

        print(f"Epoch {ep:03d} | TrainLoss {tr_loss:.4f} | Val RMSE {val_rmse:.3f} | r {val_r:.3f} | ρ {val_s:.3f} | τ {val_k:.3f} | Test RMSE {te_rmse:.3f}")

        # Save best checkpoint
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict()
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()},
                       os.path.join(save_dir, 'best_val.pt'))

        # TensorBoard
        writer_tb.add_scalar("train/loss", tr_loss, ep)
        writer_tb.add_scalar("val/rmse", val_rmse, ep)
        writer_tb.add_scalar("val/r", val_r, ep)
        writer_tb.add_scalar("val/spearman", val_s, ep)
        writer_tb.add_scalar("val/kendall", val_k, ep)
        writer_tb.add_scalar("test/rmse", te_rmse, ep)

        # CSV logging
        with open(csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, tr_loss, val_loss, val_rmse, val_r, val_s, val_k,
                             te_loss, te_rmse, te_r, te_s, te_k,
                             epoch_time, train_time, test_time])

        scheduler.step()

    writer_tb.close()

    # Save last model
    torch.save({'epoch': args.epochs, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()},
               os.path.join(save_dir, f"epoch{args.epochs}.pt"))


if __name__ == '__main__':
    main()
