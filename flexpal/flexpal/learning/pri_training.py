import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import torch, torch.nn as nn, torch.optim as optim

path = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/lpos_samples_abs_actions_mp.npz"   # 改成你的文件
D = np.load(path)
A = D["A"]          # (N, nu) 每步的动作（同一动作重复 T 次）
L = D["L"]          # (N, nu) 实际腱长
X = D["X"]          # (N, 3)
T = int(D["T"])
warmup = int(D["warmup"])
print("A shape:", A.shape, "nu:", A.shape[1], "T:", T, "warmup:", warmup)
# 1 mm 体素去重
vox = np.floor(X / 1e-3).astype(np.int32)
_, keep = np.unique(vox, axis=0, return_index=True)
L, X = L[keep], X[keep]


class Fwd(nn.Module):
    def __init__(self, nu, hidden=256): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nu, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3)
        )
    def forward(self, l): return self.net(l)

L_t = torch.tensor(L, dtype=torch.float32)
X_t = torch.tensor(X, dtype=torch.float32)

model = Fwd(nu=L.shape[1]).cuda() if torch.cuda.is_available() else Fwd(nu=L.shape[1])

opt = optim.Adam(model.parameters(), lr=3e-4)

for ep in range(50):
    opt.zero_grad()
    loss = ((model(L_t) - X_t)**2).mean()
    loss.backward(); opt.step()
