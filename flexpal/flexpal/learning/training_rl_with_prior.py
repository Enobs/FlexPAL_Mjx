# train_rl_with_prior.py
# One-night PPO training with potential-based shaping.
# Requirements: stable-baselines3, gymnasium, torch, your FlexPALSB3Env, phi_reward_wrapper.py

import os, argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.utils import set_random_seed
from phi_reward_wrapper import PhiRewardWrapper
# 你的环境
from flexpal.mujoco_cpu.flexpal_env_cpu import FlexPALSB3Env

# ---- Simple metrics callback ----
class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._last_pos_err = None
    def _on_step(self) -> bool:
        # 记录一些常用指标（如果 env 在 info 里提供）
        infos = self.locals.get("infos", [])
        if infos:
            ie = infos[-1]
            for k in ("pos_err", "ang_err", "soft_succ", "time_limit"):
                if k in ie:
                    self.logger.record(f"env/{k}", ie[k])
        return True

def make_env(xml_path, pos_only=True, render_mode=None,
             control_freq=250, k_substeps=5, max_steps=200,
             tol_pos=1e-2, tol_ang=1e-1):
    env = FlexPALSB3Env(
        xml_path=xml_path,
        action_mode="absolute",
        render_mode=render_mode,
        pos_only_ctrl=pos_only,
        control_freq=control_freq,
        K_substeps=k_substeps,
        max_episode_steps=max_steps,
        tol_pos=tol_pos,
        tol_ang=tol_ang,
    )
    return env

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml",  default="/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml")
    ap.add_argument("--phi",  default="/home/yinan/Documents/FlexPAL_Mjx/phi.pt")
    ap.add_argument("--total-steps", type=int, default=300_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta-start", type=float, default=0.2)
    ap.add_argument("--beta-end", type=float, default=0.6)
    ap.add_argument("--beta-warmup-frac", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--shape-clip", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--save-dir", type=str, default="runs_phi")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_random_seed(args.seed)

    # --- base env ---
    env = make_env(args.xml, pos_only=True, render_mode=None)

    # --- wrap with potential shaping ---
    env = PhiRewardWrapper(
        env,
        phi_ckpt=args.phi,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        warmup_frac=args.beta_warmup_frac,
        gamma=args.gamma,
        shape_clip=args.shape_clip,
        device=args.device,
    )
    # 告诉 wrapper 总步数（用于 β 退火）
    if hasattr(env, "set_total_steps"):
        env.set_total_steps(args.total_steps)

    # --- PPO config ---
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    # --- callbacks & logging ---
    cb = MetricsCallback()
    print(f"[train] total_timesteps={args.total_steps}, seed={args.seed}")
    model.learn(total_timesteps=args.total_steps, callback=cb, progress_bar=True)

    # --- save ---
    save_path = os.path.join(args.save_dir, "ppo_phi.zip")
    model.save(save_path)
    print(f"[saved] {save_path}")

if __name__ == "__main__":
    main()
