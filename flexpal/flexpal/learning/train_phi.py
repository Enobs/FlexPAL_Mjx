# train_phi.py
# Learn a modern "success distance" Φ(x, xg) for potential-based shaping.

import os, math, argparse, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def load_npz_trajs(path):
    D = np.load(path)
    # 你如果保存了时间序列，可自行组装 (x_t, x_{t+1})
    # 这里给最小版：从 (L,X) 连续行近似拼邻接对；也可自行替换为真正轨迹切片
    X = D["X"].astype(np.float32)  # (N,3)
    # 轻量下采样（防爆显存）——可删
    if X.shape[0] > 10_000_000:
        idx = np.random.choice(X.shape[0]-1, size=10_000_000, replace=False)
    else:
        idx = np.arange(X.shape[0]-1)
    x_t  = X[idx]
    x_t1 = X[idx+1]
    # 目标从库里随机抽
    g_idx = np.random.randint(0, X.shape[0], size=idx.shape[0])
    x_g = X[g_idx]
    return x_t, x_t1, x_g

class PhiNet(nn.Module):
    def __init__(self, in_dim=6, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, 2*h), nn.ReLU(),
            nn.Linear(2*h, 2*h), nn.ReLU(),
            nn.Linear(2*h, h), nn.ReLU(),
            nn.Linear(h, 1),
        )
        self.out = nn.Softplus()  # Φ>=0
    def forward(self, x, g):
        z = torch.cat([x, g], dim=-1)
        return self.out(self.net(z)).squeeze(-1)

class TripletDS(Dataset):
    def __init__(self, x, x1, g, scale=1.0):
        self.x  = x/scale; self.x1 = x1/scale; self.g = g/scale
        self.scale = scale
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i):
        return self.x[i], self.x1[i], self.g[i]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="path to npz with X (and/or traj)")
    ap.add_argument("--out", default="phi.pt")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--scale_m", type=float, default=0.5, help="workspace scale (m) for normalization")
    args = ap.parse_args()

    x, x1, g = load_npz_trajs(args.npz)
    ds = TripletDS(x, x1, g, scale=args.scale_m)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device(args.device)
    phi = PhiNet(in_dim=6).to(device)
    opt = optim.Adam(phi.parameters(), lr=args.lr)
    margin = 0.1
    lam_rank, lam_cal = 1.0, 0.1

    def l2(a): return torch.linalg.vector_norm(a, dim=-1)

    for ep in range(args.epochs):
        phi.train()
        tot = 0.0
        for xb, x1b, gb in dl:
            xb, x1b, gb = xb.to(device), x1b.to(device), gb.to(device)
            d  = l2(xb - gb)         # 归一化后的距离（≈米/scale）
            d1 = l2(x1b - gb)
            sgn = torch.sign(d - d1 + 1e-8)      # >0 进步；<0 退步

            phi_t  = phi(xb,  gb)
            phi_t1 = phi(x1b, gb)

            # ranking: (phi_t - phi_t1)*sgn >= margin
            rank = torch.clamp(margin - (phi_t - phi_t1)*sgn, min=0).mean()
            # calibration: phi ≈ distance
            cal  = ((phi_t - d)**2).mean()

            loss = lam_rank*rank + lam_cal*cal
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*xb.size(0)

        print(f"[ep {ep+1}/{args.epochs}] loss={tot/len(ds):.6f}")

    torch.save({"model": phi.state_dict(), "scale_m": args.scale_m}, args.out)
    print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()
