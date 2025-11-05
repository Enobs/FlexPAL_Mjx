import os
import re
import time
import torch
import gymnasium as gym
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
import flexpal.mujoco_cpu
import multiprocessing as mp

# ======= 基础加速与隔离 =======
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
try:
    mp.set_start_method("forkserver")
except RuntimeError:
    pass
os.environ.pop("MUJOCO_GL", None)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "log"
SAVE_DIR = os.path.join(LOG_DIR, "model_saved", "PPO")
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PREFIX = "PPO_SOFT_ROBOT"
TB_DIR = LOG_DIR  # SB3 自带 tensorboard_log 写到 LOG_DIR

# ======= 工具函数 =======
def _extract_ver_from_name(path_or_name: str) -> Optional[int]:
    base = os.path.basename(path_or_name)
    m = re.search(r'_(\d+)(?:\.zip)?$', base)
    return int(m.group(1)) if m else None

def _vecnorm_path_from_ver(ver: int) -> str:
    return os.path.join(SAVE_DIR, f"vecnormalize_{ver}.pkl")

def find_latest_version(file_prefix: str, directory: str):
    highest = -1
    latest_name = ""
    pat = re.compile(fr"^{re.escape(file_prefix)}_(\d+)\.zip$")
    if not os.path.isdir(directory):
        return latest_name, 0
    for fn in os.listdir(directory):
        m = pat.match(fn)
        if m:
            v = int(m.group(1))
            if v > highest:
                highest = v
                latest_name = fn
    return latest_name, highest + 1

def latest_model_path(prefix: str, directory: str) -> Optional[str]:
    name, _ = find_latest_version(prefix, directory)
    return os.path.join(directory, name) if name else None

# ======= 回调：定期保存（模型 + 对应 VecNormalize）=======
class TensorboardCheckpointCallback(BaseCallback):
    def __init__(self, save_dir: str, algo_tag: str = "PPO",
                 save_freq: int = 51200, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.algo_tag = algo_tag
        self.save_freq = save_freq
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_name = f"{MODEL_PREFIX}_{self.n_calls}"
            model_path = os.path.join(self.save_dir, model_name)
            self.model.save(model_path)
            vn_path = None
            env = self.model.get_env()
            if isinstance(env, VecNormalize):
                vn_path = os.path.join(self.save_dir, f"vecnormalize_{self.n_calls}.pkl")
                env.save(vn_path)
            if self.verbose:
                print(f"[Checkpoint] Saved model: {model_path} ; vecnorm: {vn_path or '(none)'}")
        return True

# ======= 并行环境 =======
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

# ======= 训练 =======
def train(total_steps: int = int(1e6),
          n_envs: int = 16,
          n_steps: int = 128,
          batch_size: int = 1024,
          ent_coef: float = 3e-3,
          gamma: float = 0.99,
          seed: int = 0,
          save_freq: int = 51200,
          resume: bool = False,
          n_epochs: int = 8):
    policy_kwargs = dict(net_arch=[128, 128])

    # 构造 env / model（统一放到 resume 分支中处理）
    last_path = latest_model_path(MODEL_PREFIX, SAVE_DIR) if resume else None
    if resume and last_path and os.path.exists(last_path):
        print(f"[Resume] Loading {last_path}")
        envs = make_vec_envs(n_envs=n_envs, use_subproc=True)
        ver = _extract_ver_from_name(last_path)
        vn_file = _vecnorm_path_from_ver(ver) if ver is not None else None
        if vn_file and os.path.exists(vn_file):
            env = VecNormalize.load(vn_file, envs)
            print(f"[Resume] Loaded VecNormalize from {vn_file}")
        else:
            env = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=gamma)
            print("[Resume] VecNormalize not found; started fresh statistics")
        env.training = True
        env.norm_reward = True
        model = PPO.load(last_path, env=env, device=DEVICE)
    else:
        if resume:
            print("[Resume] No previous model found, starting fresh.")
        envs = make_vec_envs(n_envs=n_envs, use_subproc=True)
        env = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=gamma)
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            device=DEVICE,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            gamma=gamma,
            learning_rate=1e-4,
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
    callback = TensorboardCheckpointCallback(save_dir=SAVE_DIR, save_freq=save_freq, verbose=1)

    # Warmup（resume 时不扰动统计）
    warmup = 0 if resume else max(n_envs * 2, 2048)
    if warmup > 0:
        print(f"[Warmup] {warmup} steps (not counted)")
        model.learn(total_timesteps=warmup, reset_num_timesteps=False, progress_bar=False)

    # 计时
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=callback, reset_num_timesteps=False, progress_bar=False)
    t1 = time.time()

    elapsed = t1 - t0
    sps = total_steps / max(elapsed, 1e-9)
    sps_per_env = sps / n_envs
    print("=" * 60)
    print(f"[PPO] total_steps={total_steps:,} | n_envs={n_envs} | device={DEVICE}")
    print(f"Elapsed: {elapsed:.2f}s | Throughput: {sps:,.0f} steps/s (~{sps_per_env:,.0f}/env/s)")
    print("=" * 60)

    # 末尾保存（与 SAVE_DIR 统一）
    _, next_ver = find_latest_version(MODEL_PREFIX, SAVE_DIR)
    final_path = os.path.join(SAVE_DIR, f"{MODEL_PREFIX}_{next_ver}")
    model.save(final_path)
    print(f"[Save] Final model -> {final_path}")

    if isinstance(env, VecNormalize):
        vn_path = _vecnorm_path_from_ver(next_ver)
        env.save(vn_path)
        print(f"[Save] VecNormalize -> {vn_path}")

    env.close()

# ======= 测试 =======
def test_model(model_path: Optional[str] = None, deterministic: bool = True, episodes: int = 5):
    if model_path is None:
        model_path = latest_model_path(MODEL_PREFIX, SAVE_DIR)
    assert model_path and os.path.exists(model_path), f"Model not found: {model_path}"
    print(f"[Test] Loading {model_path}")

    eval_venv = make_vec_envs(n_envs=1, use_subproc=False)
    ver = _extract_ver_from_name(model_path)
    vn_file = _vecnorm_path_from_ver(ver) if ver is not None else None
    if vn_file and os.path.exists(vn_file):
        eval_env = VecNormalize.load(vn_file, eval_venv)
        print(f"[Test] Loaded VecNormalize from {vn_file}")
    else:
        eval_env = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
        print("[Test] WARNING: VecNormalize stats not found. Using fresh stats (sanity check only).")

    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env, device=DEVICE)

    obs = eval_env.reset()
    for ep in range(episodes):
        ep_ret, ep_len = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = eval_env.step(action)
            ep_ret += float(reward[0])
            ep_len += 1
            if done[0]:
                print(f"[Episode {ep+1}] return={ep_ret:.2f} len={ep_len}")
                obs = eval_env.reset()
                break
    eval_env.close()

# ======= 入口 =======
if __name__ == "__main__":
    train(
        total_steps=int(1e4),
        n_envs=16,          # 训练时建议多环境
        n_steps=64,
        batch_size=512,
        gamma=0.99,
        ent_coef=1e-3,
        seed=0,
        save_freq=51200,
        resume=True,
        n_epochs=8,
    )
    # test_model()  # 训练完单独跑评估时打开
