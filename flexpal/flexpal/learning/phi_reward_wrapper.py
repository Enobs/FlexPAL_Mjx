# phi_reward_wrapper.py
# Potential-based shaping wrapper for SB3/Gymnasium envs.
# r' = r + beta(t) * (gamma * Phi(s') - Phi(s))

import numpy as np
import torch
import gymnasium as gym

# --- model def (must match train_phi.py) ---
class _PhiNet(torch.nn.Module):
    def __init__(self, in_dim=6, h=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, h), torch.nn.ReLU(),
            torch.nn.Linear(h, 2*h),  torch.nn.ReLU(),
            torch.nn.Linear(2*h, 2*h), torch.nn.ReLU(),
            torch.nn.Linear(2*h, h), torch.nn.ReLU(),
            torch.nn.Linear(h, 1),
        )
        self.out = torch.nn.Softplus()
    def forward(self, x, g):
        z = torch.cat([x, g], dim=-1)
        return self.out(self.net(z)).squeeze(-1)

class _PhiModel:
    def __init__(self, ckpt_path, device="cpu"):
        ck = torch.load(ckpt_path, map_location=device)
        self.scale = ck.get("scale_m", 0.5)
        self.net = _PhiNet().to(device)
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.device = device
    @torch.no_grad()
    def phi(self, x, g):
        # x,g: (3,) numpy
        xt = torch.tensor(x / self.scale, dtype=torch.float32, device=self.device).unsqueeze(0)
        gt = torch.tensor(g / self.scale, dtype=torch.float32, device=self.device).unsqueeze(0)
        v = self.net(xt, gt).item()
        return float(v)

class PhiRewardWrapper(gym.Wrapper):
    """
    Potential-based shaping wrapper.
    - beta schedule: linear warmup from beta_start -> beta_end over warmup_frac of total steps
      (requires calling set_total_steps() once before training starts)
    - r_shape is clipped into [-shape_clip, shape_clip] for stability
    """
    def __init__(self, env, phi_ckpt, beta_start=0.2, beta_end=0.6, warmup_frac=0.3,
                 gamma=0.99, shape_clip=0.5, device="cpu"):
        super().__init__(env)
        self.model = _PhiModel(phi_ckpt, device=device)
        self.gamma = float(gamma)
        self.beta_start = float(beta_start)
        self.beta_end   = float(beta_end)
        self.warmup_frac= float(warmup_frac)
        self.shape_clip = float(shape_clip)

        self._last_phi = None
        self._goal = None
        self._t = 0
        self._T_total = 1  # avoid div by zero until set_total_steps is called

    def set_total_steps(self, total_timesteps: int):
        self._T_total = max(1, int(total_timesteps))

    def _beta_now(self):
        # linear warmup
        warmup_T = int(self.warmup_frac * self._T_total)
        if warmup_T <= 0:
            return self.beta_end
        k = min(1.0, self._t / warmup_T)
        return (1.0 - k) * self.beta_start + k * self.beta_end

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        if isinstance(obs, dict):
            self._goal = obs["desired_goal"][:3]
            x = obs["achieved_goal"][:3]
        else:
            self._goal = getattr(self.env, "goal", None)[:3]
            x = self.env.get_ee_pos()
        self._last_phi = self.model.phi(np.array(x, dtype=np.float32),
                                        np.array(self._goal, dtype=np.float32))
        return obs, info

    def step(self, action):
        obs, r_env, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, dict):
            x = obs["achieved_goal"][:3]
            g = obs["desired_goal"][:3]
        else:
            x = self.env.get_ee_pos()
            g = self._goal
        phi_now = self.model.phi(np.array(x, dtype=np.float32),
                                 np.array(g, dtype=np.float32))
        beta = self._beta_now()
        r_shape = beta * (self.gamma * phi_now - self._last_phi)
        r_shape = float(np.clip(r_shape, -self.shape_clip, self.shape_clip))
        self._last_phi = phi_now
        self._t += 1
        return obs, float(r_env + r_shape), terminated, truncated, info
