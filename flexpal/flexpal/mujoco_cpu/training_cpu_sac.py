# train_her_sac.py
import os, re, time, csv
import numpy as np
import torch, gymnasium as gym
import multiprocessing as mp
from typing import Optional, Dict, Any
from collections import deque
import flexpal.mujoco_cpu
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

# 基础设置
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1); torch.set_num_interop_threads(1)
try: mp.set_start_method("forkserver")
except RuntimeError: pass
os.environ.pop("MUJOCO_GL", None)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "log"
SAVE_DIR = os.path.join(LOG_DIR, "model_saved", "HER_SAC")
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PREFIX = "HER_SAC_SOFT_ROBOT"
TB_DIR = LOG_DIR

# 统一环境参数（训练/测试一致）
ENV_KW: Dict[str, Any] = dict(
    control_freq=250,
    K_substeps=5,
    tol_pos=0.015,
    tol_ang=0.1,
    hold_steps=2,
    dL_max=0.10,
    reward_mode="exp",
    r_bonus=20.0,
    near_gain=1.0,
    scale=0.03,
    prog_coef=0.5,
    du_coef_far=1e-3,
    du_stop_near=5.0,
    step_penalty=0.002,
    pos_only_ctrl=True,
    curriculum_precision_prob=0.2,
    curriculum_r_small=0.03,
    curriculum_r_large=0.12,
)

def _extract_ver_from_name(path_or_name: str) -> Optional[int]:
    base = os.path.basename(path_or_name)
    m = re.search(r'_(\d+)(?:\.zip)?$', base)
    return int(m.group(1)) if m else None

def _vecnorm_path_from_ver(ver: int) -> str:
    return os.path.join(SAVE_DIR, f"vecnormalize_{ver}.pkl")

def find_latest_version(file_prefix: str, directory: str):
    highest = -1; latest_name = ""
    pat = re.compile(fr"^{re.escape(file_prefix)}_(\d+)\.zip$")
    if not os.path.isdir(directory): return latest_name, 0
    for fn in os.listdir(directory):
        m = pat.match(fn)
        if m:
            v = int(m.group(1))
            if v > highest: highest = v; latest_name = fn
    return latest_name, highest + 1

def latest_model_path(prefix: str, directory: str) -> Optional[str]:
    name, _ = find_latest_version(prefix, directory)
    return os.path.join(directory, name) if name else None

# 向量化环境
def make_env():
    def _init():
        env = gym.make("Gym_softmujoco-v0_SAC", **ENV_KW)
        env = Monitor(env)
        env.reset()
        return env
    return _init

def make_vec_envs(n_envs: int, use_subproc: bool = True):
    Vec = SubprocVecEnv if (use_subproc and n_envs > 1) else DummyVecEnv
    if Vec is SubprocVecEnv:
        return Vec([make_env() for _ in range(n_envs)], start_method="forkserver")
    return Vec([make_env() for _ in range(n_envs)])

def _clone_vecnorm_stats(src_vn: VecNormalize, dst_vn: VecNormalize):
    def _copy_rms(src_rms, dst_rms):
        if isinstance(src_rms, dict) and isinstance(dst_rms, dict):
            for k in src_rms.keys():
                if k not in dst_rms: continue
                dst_rms[k].mean  = src_rms[k].mean.copy()
                dst_rms[k].var   = src_rms[k].var.copy()
                dst_rms[k].count = float(src_rms[k].count)
        else:
            dst_rms.mean  = src_rms.mean.copy()
            dst_rms.var   = src_rms.var.copy()
            dst_rms.count = float(src_rms.count)
    _copy_rms(src_vn.obs_rms, dst_vn.obs_rms)
    if src_vn.ret_rms is not None and dst_vn.ret_rms is not None:
        _copy_rms(src_vn.ret_rms, dst_vn.ret_rms)
    dst_vn.clip_obs    = src_vn.clip_obs
    dst_vn.clip_reward = src_vn.clip_reward
    dst_vn.gamma       = src_vn.gamma
    dst_vn.epsilon     = src_vn.epsilon
    dst_vn.training    = src_vn.training
    dst_vn.norm_reward = src_vn.norm_reward

def _rebuild_env_and_swap(model, n_envs: int, gamma: float = 0.98):
    old_env = model.get_env()
    assert isinstance(old_env, VecNormalize), "最外层应为 VecNormalize"
    new_vec = make_vec_envs(n_envs=n_envs, use_subproc=True)
    new_env = VecNormalize(new_vec, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=gamma)
    new_env.training = True; new_env.norm_reward = False
    _clone_vecnorm_stats(old_env, new_env)
    model.set_env(new_env)
    try: old_env.close()
    except Exception: pass
    return new_env

# train_her_sac.py
def her_compute_reward(achieved, desired, info):
    ag = np.asarray(achieved)[..., :3]
    dg = np.asarray(desired)[..., :3]
    dist = np.linalg.norm(ag - dg, axis=-1)
    reward = -dist
    reward = np.where(dist < 0.015, reward + 20.0, reward)  # 与 env 的 tol_pos/r_bonus 对齐
    return reward


# 成功率与精度指标回调（TensorBoard 里看）
class SuccessMetricsCallback(BaseCallback):
    def __init__(self, window=2000, verbose=0):
        super().__init__(verbose)
        self.window = int(window)
        self.pos_err_buf = deque(maxlen=self.window)
        self.success_buf = deque(maxlen=self.window)

    def _on_step(self) -> bool:
        # vec-env 的 infos 是 list
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict): continue
            if "pos_err" in info:
                self.pos_err_buf.append(float(info["pos_err"]))
            if "success" in info:
                self.success_buf.append(1.0 if info["success"] else 0.0)

        if self.n_calls % 1000 == 0 and len(self.pos_err_buf) > 0:
            pos = np.array(self.pos_err_buf, dtype=np.float32)
            suc = np.array(self.success_buf, dtype=np.float32) if len(self.success_buf)>0 else np.zeros_like(pos)
            s10 = (pos < 0.010).mean()
            s05 = (pos < 0.005).mean()
            p95 = float(np.percentile(pos, 95))
            self.logger.record("eval/success@10mm", s10)
            self.logger.record("eval/success@5mm",  s05)
            self.logger.record("eval/pos_err_mean", float(pos.mean()))
            self.logger.record("eval/pos_err_p95",  p95)
        return True

class TensorboardCheckpointCallback(BaseCallback):
    def __init__(self, save_dir: str, save_freq: int = 51200, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir; self.save_freq = save_freq
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_name = f"{MODEL_PREFIX}_{self.n_calls}"
            model_path = os.path.join(self.save_dir, model_name)
            self.model.save(model_path)
            env = self.model.get_env()
            vn_path = None
            if isinstance(env, VecNormalize):
                vn_path = os.path.join(self.save_dir, f"vecnormalize_{self.n_calls}.pkl")
                env.save(vn_path)
            if self.verbose:
                print(f"[Checkpoint] Saved: {model_path} ; vecnorm: {vn_path or '(none)'}")
        return True

# 训练
def train(total_steps: int = int(2e7),
          n_envs: int = 8,
          gamma: float = 0.98,
          seed: int = 0,
          save_freq: int = 51200,
          resume: bool = False,
          steps_per_cycle: int = int(5e6)):
    envs = make_vec_envs(n_envs=n_envs, use_subproc=True)
    env = VecNormalize(envs, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=gamma)
    env.training = True; env.norm_reward = False
    
    replay_buffer_kwargs = dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",  # "future" | "final" | "episode"
    )

    sac_kwargs = dict(
        policy="MultiInputPolicy",
        learning_rate=3e-4,
        buffer_size=int(2e6),
        learning_starts=10000,
        batch_size=512,
        tau=0.02,
        gamma=gamma,
        train_freq=256,
        gradient_steps=256,
        ent_coef="auto",
        verbose=1,
        device=DEVICE,
        tensorboard_log=TB_DIR,
        policy_kwargs=dict(net_arch=[512, 512, 512]),
        seed=seed,

        # ★ 关键：接入 HER 回放
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=replay_buffer_kwargs,
    )

    last_path = latest_model_path(MODEL_PREFIX, SAVE_DIR) if resume else None
    if resume and last_path and os.path.exists(last_path):
        print(f"[Resume] Loading {last_path}")
        model = SAC.load(last_path, env=env, device=DEVICE)  # ← 用 SAC.load
        # VecNormalize 另行加载迁移（保持你现有逻辑）
    else:
        if resume:
            print("[Resume] No previous model found, starting fresh.")
        model = SAC(env=env, **sac_kwargs)

    cb = [TensorboardCheckpointCallback(save_dir=SAVE_DIR, save_freq=save_freq, verbose=1),
          SuccessMetricsCallback(window=2000)]

    done = 0; t0 = time.time()
    while done < total_steps:
        run = min(steps_per_cycle, total_steps - done)
        print(f"\n=== Cycle: training {run:,} steps (done {done:,}/{total_steps:,}) ===")
        model.learn(total_timesteps=run, reset_num_timesteps=False, callback=cb, progress_bar=False)
        done += run
        if done < total_steps:
            _rebuild_env_and_swap(model, n_envs=n_envs, gamma=gamma)
            print("[Cycle] Rebuilt SubprocVecEnv and migrated VecNormalize stats.")

    t1 = time.time()
    print("=" * 60)
    print(f"[HER+SAC] total_steps={total_steps:,} | n_envs={n_envs} | device={DEVICE}")
    print(f"Elapsed: {t1 - t0:.2f}s")
    print("=" * 60)

    _, next_ver = find_latest_version(MODEL_PREFIX, SAVE_DIR)
    final_path = os.path.join(SAVE_DIR, f"{MODEL_PREFIX}_{next_ver}")
    model.save(final_path); print(f"[Save] Final model -> {final_path}")
    vec = model.get_env()
    if isinstance(vec, VecNormalize):
        vn_path = _vecnorm_path_from_ver(next_ver); vec.save(vn_path)
        print(f"[Save] VecNormalize -> {vn_path}")
    vec.close()

# 测试（含 success@10mm/5mm / p95 ）
def test_model(model_path: Optional[str] = None, deterministic: bool = True, episodes: int = 50):
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
        print("[Test] WARNING: VecNormalize stats not found.")

    eval_env.training = False; eval_env.norm_reward = False
    model = SAC.load(model_path, env=eval_env, device=DEVICE)

    succ = 0; pos_list = []
    for ep in range(episodes):
        obs = eval_env.reset(); ep_ret = 0.0; ep_len = 0
        done = [False]
        while not done[0]:
            action, _ = model.predict(obs, deterministic=deterministic)
            next_obs, reward, done, info = eval_env.step(action)
            ep_ret += float(reward[0]); ep_len += 1
            if done[0]:
                term = info[0].get("terminal_observation") or info[0].get("final_observation")
                if term is not None and hasattr(eval_env, "unnormalize_obs"):
                    term_orig = eval_env.unnormalize_obs(term)
                    ag = np.asarray(term_orig["achieved_goal"][:3], dtype=np.float32)
                    dg = np.asarray(term_orig["desired_goal"][:3], dtype=np.float32)
                    dist = float(np.linalg.norm(ag - dg))
                else:
                    dist = float(info[0].get("pos_err", 0.0))
                pos_list.append(dist)
                s = bool(info[0].get("success", False)); succ += int(s)
                print(f"[Ep {ep+1}] len={ep_len} success={s} pos_err={dist:.4f} ret={ep_ret:.2f}")
                print(f"[achieved_goal {term_orig["achieved_goal"][:3]}] [term_orig {term_orig["desired_goal"][:3]}]")
                obs = eval_env.reset()
            else:
                obs = next_obs

    pos = np.array(pos_list, dtype=np.float32)
    s10 = float((pos < 0.010).mean()); s05 = float((pos < 0.005).mean())
    print(f"[Eval] episodes={episodes} | success@10mm={s10:.3f} | success@5mm={s05:.3f} | "
          f"avg_pos_err={float(pos.mean()):.4f} | p95_pos_err={float(np.percentile(pos,95)):.4f} | "
          f"deterministic={deterministic}")

# 逐步记录轨迹 CSV（调试精度用）
def save_eval_traj(model_path: Optional[str] = None, episodes: int = 10,
                   deterministic: bool = True, out_csv="eval_traj.csv"):
    if model_path is None:
        model_path = latest_model_path(MODEL_PREFIX, SAVE_DIR)
    eval_venv = make_vec_envs(n_envs=1, use_subproc=False)
    ver = _extract_ver_from_name(model_path)
    vn_file = _vecnorm_path_from_ver(ver) if ver is not None else None
    if vn_file and os.path.exists(vn_file):
        eval_env = VecNormalize.load(vn_file, eval_venv)
    else:
        eval_env = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False; eval_env.norm_reward = False
    model = SAC.load(model_path, env=eval_env, device=DEVICE)

    rows = []
    for ep in range(episodes):
        obs = eval_env.reset(); t = 0
        done = [False]
        while not done[0]:
            if hasattr(eval_env, "unnormalize_obs"):
                obs_un = eval_env.unnormalize_obs(obs)
            else:
                obs_un = obs
            ag = np.asarray(obs_un["achieved_goal"][0][:3], dtype=np.float32)
            dg = np.asarray(obs_un["desired_goal"][0][:3], dtype=np.float32)
            pos_err = float(np.linalg.norm(ag - dg))
            action, _ = model.predict(obs, deterministic=deterministic)
            next_obs, reward, done, info = eval_env.step(action)
            rows.append({
                "episode": ep+1, "t": t,
                "ach_x": float(ag[0]), "ach_y": float(ag[1]), "ach_z": float(ag[2]),
                "des_x": float(dg[0]), "des_y": float(dg[1]), "des_z": float(dg[2]),
                "pos_err": pos_err
            })
            t += 1; obs = next_obs
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[EvalTraj] Saved {len(rows)} rows -> {out_csv}")

if __name__ == "__main__":
    # 训练：achievedesired_goald_goal
    # train(total_steps=int(1e5), n_envs=1, gamma=0.98, seed=0,
    #       save_freq=51200, resume=False, steps_per_cycle=int(1e5))

    # 测试：
    test_model(deterministic=True, episodes=300)

    # 导出轨迹：
    # save_eval_traj(episodes=10, deterministic=True, out_csv="eval_traj.csv")
    pass
