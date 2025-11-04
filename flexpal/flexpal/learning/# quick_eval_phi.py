# quick_eval_phi.py
import numpy as np, torch
from train_phi import PhiNet, load_npz_trajs
def eval_phi(ckpt, npz, n_eval=1000000, scale_m=0.5, device="cuda"):
    ck = torch.load(ckpt, map_location=device); scale = ck.get("scale_m", scale_m)
    phi = PhiNet().to(device); phi.load_state_dict(ck["model"]); phi.eval()
    x, x1, g = load_npz_trajs(npz)
    idx = np.random.choice(len(x), size=min(n_eval, len(x)), replace=False)
    x, x1, g = x[idx]/scale, x1[idx]/scale, g[idx]/scale
    xt = torch.tensor(x, device=device); x1t = torch.tensor(x1, device=device); gt = torch.tensor(g, device=device)
    with torch.no_grad():
        d  = torch.linalg.vector_norm(xt-gt,  dim=-1)
        d1 = torch.linalg.vector_norm(x1t-gt, dim=-1)
        sgn = torch.sign(d - d1 + 1e-8)
        ph  = phi(xt,  gt); ph1 = phi(x1t, gt)
    rank_acc = ((ph - ph1)*sgn > 0).float().mean().item()
    mae = (ph - d).abs().mean().item()
    return rank_acc, mae
if __name__ == "__main__":
    ra, mae = eval_phi("/home/yinan/Documents/FlexPAL_Mjx/phi.pt", "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/lpos_samples_abs_actions_mp_59.npz")
    print(f"rank-acc={ra:.3f},  cal-MAE={mae:.3f}")
