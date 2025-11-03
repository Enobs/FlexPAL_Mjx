import os
import re
import time
import torch
import gymnasium as gym
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import flexpal.mujoco_cpu

import os, torch, multiprocessing as mp
# 锁死每个子进程的BLAS/OMP线程，避免256进程互抢核
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Linux上更稳：forkserver，避免GL/OMP状态被fork污染
try:
    mp.set_start_method("forkserver")
except RuntimeError:
    pass

# 训练期禁用渲染
os.environ.pop("MUJOCO_GL", None)

# ========= 全局配置 =========
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "log/"
MODEL_PREFIX = "PPO_SOFT_ROBOT"
TB_DIR = os.path.join(LOG_DIR)  # SB3 自带 tensorboard_log

# ========= 回调：定期保存 =========
class TensorboardCheckpointCallback(BaseCallback):
    """
    每隔 save_freq 环境步保存一次模型到 log_dir/model_saved/PPO/...
    """
    def __init__(self, log_dir: str, algo_tag: str = "PPO",
                 save_freq: int = 51200, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.algo_tag = algo_tag
        self.save_freq = save_freq
        os.makedirs(os.path.join(self.log_dir, "model_saved", self.algo_tag), exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.log_dir, "model_saved", self.algo_tag,
                f"{MODEL_PREFIX}_{self.n_calls}"
            )
            self.model.save(path)
            if self.verbose:
                print(f"[Checkpoint] Saved to: {path}")
        return True

# ========= 工具：并行环境 =========
def make_env():
    def _init():
        env = gym.make("Gym_softmujoco-v0")
        env = Monitor(env)
        env.reset()
        return env
    return _init

def make_vec_envs(n_envs: int, use_subproc: bool = True):
    Vec = SubprocVecEnv if (use_subproc and n_envs > 1) else DummyVecEnv
    return Vec([make_env() for _ in range(n_envs)])

# ========= 工具：查找最新版本 =========
def find_latest_version(file_prefix: str, directory: str):
    """
    查找目录下 file_prefix_数字 的最大版本号，返回 (最新文件名, 下一个版本号)
    """
    highest = -1
    latest_name = ""
    pat = re.compile(fr"^{re.escape(file_prefix)}_(\d+)$")
    for fn in os.listdir(directory):
        m = pat.match(fn)
        if m:
            v = int(m.group(1))
            if v > highest:
                highest = v
                latest_name = fn
    return latest_name, highest + 1

def latest_model_path(prefix: str, directory: str) -> Optional[str]:
    latest, _ = find_latest_version(prefix, directory)
    return os.path.join(directory, latest) if latest else None

# ========= 训练 =========
def train(total_steps: int = int(1e5),
          n_envs: int = 10,
          n_steps: int = 2048,
          batch_size: int = 256,
          ent_coef: float = 5e-3,
          gamma: float = 0.99,
          seed: int = 0,
          save_freq: int = 51200,
          resume: bool = False,
          n_epochs: int = 8):
    """
    PPO 训练主程序
    """
    env = make_vec_envs(n_envs=n_envs, use_subproc=True)

    policy_kwargs = dict(net_arch=[128, 128])  # 你原来 512×6 太大易不稳；PPO 默认 256×2 常用
    model: PPO

    # 断点续训（如果需要）
    if resume:
        os.makedirs(".", exist_ok=True)
        last_path = latest_model_path(MODEL_PREFIX, ".")
        if last_path and os.path.exists(last_path + ".zip"):
            print(f"[Resume] Loading {last_path}")
            model = PPO.load(last_path, env=env, device=DEVICE)
        else:
            print("[Resume] No previous model found, starting fresh.")
            model = PPO(
                policy="MultiInputPolicy",
                env=env,
                device=DEVICE,
                n_steps=n_steps,
                batch_size=batch_size,
                ent_coef=ent_coef,
                gamma=gamma,
                learning_rate = 1e-4,
                target_kl=0.02,          
                max_grad_norm=0.3,
                n_epochs=n_epochs,
                gae_lambda=0.95,
                tensorboard_log=TB_DIR,
                seed=seed,
                verbose=1,
                policy_kwargs=policy_kwargs,
            )
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            device=DEVICE,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            gamma=gamma,
            learning_rate = 1e-4,
            target_kl=0.02,          
            max_grad_norm=0.3,
            n_epochs=n_epochs,
            gae_lambda=0.95,
            tensorboard_log=TB_DIR,
            seed=seed,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )

    # 回调：定期保存
    callback = TensorboardCheckpointCallback(
        log_dir=LOG_DIR, algo_tag="PPO", save_freq=save_freq, verbose=1
    )

    # 预热（可选）
    warmup = max(n_envs * 2, 2048)
    print(f"[Warmup] {warmup} steps (not counted)")
    model.learn(total_timesteps=warmup, reset_num_timesteps=False, progress_bar=False)

    # 计时
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=callback,
                reset_num_timesteps=False, progress_bar=False)
    t1 = time.time()

    elapsed = t1 - t0
    sps = total_steps / max(elapsed, 1e-9)
    sps_per_env = sps / n_envs
    print("=" * 60)
    print(f"[PPO] total_steps={total_steps:,} | n_envs={n_envs} | device={DEVICE}")
    print(f"Elapsed: {elapsed:.2f}s | Throughput: {sps:,.0f} steps/s (~{sps_per_env:,.0f}/env/s)")
    print("=" * 60)

    # 按递增版本保存
    _, next_ver = find_latest_version(MODEL_PREFIX, ".")
    final_path = f"{MODEL_PREFIX}_{next_ver}"
    model.save(final_path)
    print(f"[Save] Final model -> {final_path}")

    env.close()

# ========= 测试 =========
def test_model(model_path: Optional[str] = None, deterministic: bool = True, episodes: int = 5):

    env = gym.make("Gym_softmujoco-v0")
    if model_path is None:
        model_path = latest_model_path(MODEL_PREFIX, ".")
    assert model_path and os.path.exists(model_path + ".zip"), f"Model not found: {model_path}"
    print(f"[Test] Loading {model_path}")
    model = PPO.load(model_path, env=env, device=DEVICE)

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_ret, ep_len = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
            # 如需可视化再启用：env.render()
            if terminated or truncated:
                print(f"[Episode {ep+1}] return={ep_ret:.2f} len={ep_len}")
                break
    env.close()

# ========= 入口 =========
if __name__ == "__main__":
    # train(
    #     total_steps=int(1e7),
    #     n_envs=128,
    #     n_steps=128,
    #     batch_size=1024,
    #     gamma=0.96,
    #     ent_coef = 1e-3,
    #     seed=0,
    #     save_freq=51200,
    #     resume=False,    
    #     n_epochs=8,
    # )
    test_model("/home/yinan/Documents/FlexPAL_Mjx/PPO_SOFT_ROBOT_new")
