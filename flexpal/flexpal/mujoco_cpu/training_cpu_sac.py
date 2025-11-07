import os
import re
import time
import torch
import gymnasium as gym
from typing import Optional
from stable_baselines3 import SAC
from stable_baselines3.her import HER
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
import flexpal.mujoco_cpu
import multiprocessing as mp
import numpy as np

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
SAVE_DIR = os.path.join(LOG_DIR, "model_saved", "HER_SAC")
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PREFIX = "HER_SAC_SOFT_ROBOT"
TB_DIR = LOG_DIR  # SB3 自带 tensorboard_log 写到 LOG_DIR

# ======= 工具函数 =======
def _extract_ver_from_name(path_or_name: str) -> Optional[int]:
    base = os.path.basename(path_or_name)
    m = re.search(r'_(\d+)(?:\.zip)?$', base)
    return int(m.group(1)) if m else None

def _vecnorm_path_from_ver(ver: int) -> str:
    return os.path.join(SAVE_DIR, f"vecnormalize_{ver}.pkl")

def _clone_vecnorm_stats(src_vn: VecNormalize, dst_vn: VecNormalize):
    """复制 obs_rms / ret_rms 统计量（VecNormalize 新旧兼容）"""
    def _copy_rms(src_rms, dst_rms):
        if isinstance(src_rms, dict) and isinstance(dst_rms, dict):
            for key in src_rms.keys():
                if key not in dst_rms:
                    continue
                dst_rms[key].mean  = src_rms[key].mean.copy()
                dst_rms[key].var   = src_rms[key].var.copy()
                dst_rms[key].count = float(src_rms[key].count)
        else:
            dst_rms.mean  = src_rms.mean.copy()
            dst_rms.var   = src_rms.var.copy()
            dst_rms.count = float(src_rms.count)
    _copy_rms(src_vn.obs_rms, dst_vn.obs_rms)
    if src_vn.ret_rms is not None and dst_vn.ret_rms is not None:
        _copy_rms(src_vn.ret_rms, dst_vn.ret_rms)
    dst_vn.clip_obs     = src_vn.clip_obs
    dst_vn.clip_reward  = src_vn.clip_reward
    dst_vn.gamma        = src_vn.gamma
    dst_vn.epsilon      = src_vn.epsilon
    dst_vn.training     = src_vn.training
    dst_vn.norm_reward  = src_vn.norm_reward

def _rebuild_env_and_swap(model, n_envs: int, gamma: float = 0.99):
    """
    关闭旧的 SubprocVecEnv，重建新的子进程，并把 VecNormalize 的统计量拷过去。
    兼容 HER(SAC)：off-policy 也可以 set_env。
    """
    old_env = model.get_env()
    assert isinstance(old_env, VecNormalize), "train() 里请确保最外层是 VecNormalize"

    new_vec = make_vec_envs(n_envs=n_envs, use_subproc=True)
    # HER/SAC 下建议 norm_reward=False（避免稀疏奖励被缩放）
    new_env = VecNormalize(new_vec, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=gamma)
    new_env.training = True
    new_env.norm_reward = False

    _clone_vecnorm_stats(old_env, new_env)
    model.set_env(new_env)

    try:
        old_env.close()
    except Exception:
        pass
    return new_env

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
    def __init__(self, save_dir: str, algo_tag: str = "HER_SAC",
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
        # 建议将你的 ENV_KW 显式传入 gym.make("Gym_softmujoco-v0", **ENV_KW)
        env = gym.make("Gym_softmujoco-v0")
        env = Monitor(env)
        env.reset()
        return env
    return _init

def make_vec_envs(n_envs: int, use_subproc: bool = True):
    Vec = SubprocVecEnv if (use_subproc and n_envs > 1) else DummyVecEnv
    if Vec is SubprocVecEnv:
        return Vec([make_env() for _ in range(n_envs)], start_method="forkserver")
    return Vec([make_env() for _ in range(n_envs)])

# ======= HER 的 reward 函数（dense + bonus，兼容 3D 位置目标）=======
def her_compute_reward(achieved_goal, desired_goal, info):
    """
    achieved_goal, desired_goal: (..., D)
    这里按位置前三维计算距离；阈值与奖励可按需调整。
    """
    ag = np.asarray(achieved_goal)[..., :3]
    dg = np.asarray(desired_goal)[..., :3]
    dist = np.linalg.norm(ag - dg, axis=-1)

    # dense 负距离 + 命中奖励（更接近“成功率”）
    reward = -dist
    reward = np.where(dist < 0.015, reward + 20.0, reward)  # tol≈1.5cm 时 +20
    return reward

# ======= 训练（SAC + HER）=======
def train(total_steps: int = int(2e7),
          n_envs: int = 8,
          gamma: float = 0.98,
          seed: int = 0,
          save_freq: int = 51200,
          resume: bool = False,
          steps_per_cycle: int = int(5e6)):   # 每个周期后重启子进程
    # === VecEnv + VecNormalize ===
    envs = make_vec_envs(n_envs=n_envs, use_subproc=True)
    # HER/SAC：norm_obs=True, norm_reward=False
    env = VecNormalize(envs, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=gamma)
    env.training = True
    env.norm_reward = False

    # === SAC 的超参 ===
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
    )

    last_path = latest_model_path(MODEL_PREFIX, SAVE_DIR) if resume else None
    if resume and last_path and os.path.exists(last_path):
        print(f"[Resume] Loading {last_path}")
        # 载入 HER 包装的模型
        model = HER.load(last_path, env=env, device=DEVICE)
        # 注意：HER.load 会自动恢复 SAC 的内部状态；VecNormalize 统计需你自己加载（见下）
        # 如需同时恢复 VecNormalize 统计，请在下面加载（本脚本的 checkpoint 已会保存 vecnorm）
        ver = _extract_ver_from_name(last_path)
        vn_file = _vecnorm_path_from_ver(ver) if ver is not None else None
        if vn_file and os.path.exists(vn_file):
            env2 = VecNormalize.load(vn_file, envs)  # 用新的子进程包上统计
            env2.training = True
            env2.norm_reward = False
            _clone_vecnorm_stats(env, env2)
            model.set_env(env2)
            env = model.get_env()
            print(f"[Resume] Loaded VecNormalize from {vn_file}")
        else:
            print("[Resume] VecNormalize not found; keep current stats")
    else:
        if resume:
            print("[Resume] No previous model found, starting fresh.")
        # === 创建 HER(SAC) 模型 ===
        model = HER(
            policy="MultiInputPolicy",
            env=env,
            model_class=SAC,
            n_sampled_goal=4,
            goal_selection_strategy="future",
            online_sampling=True,
            max_episode_length=100,            # 和 env 的每局上限一致
            compute_reward=her_compute_reward, # 使用上面的 dense+bonus
            **sac_kwargs,
        )

    callback = TensorboardCheckpointCallback(save_dir=SAVE_DIR, save_freq=save_freq, verbose=1)

    # === 训练（分周期 + 重启子进程，迁移 VecNormalize 统计）===
    done = 0
    t0 = time.time()
    while done < total_steps:
        run = min(steps_per_cycle, total_steps - done)
        print(f"\n=== Cycle: training {run:,} steps (done {done:,}/{total_steps:,}) ===")
        model.learn(total_timesteps=run, reset_num_timesteps=False, callback=callback, progress_bar=False)
        done += run

        if done < total_steps:
            _rebuild_env_and_swap(model, n_envs=n_envs, gamma=gamma)
            print("[Cycle] Rebuilt SubprocVecEnv and migrated VecNormalize stats.")

    t1 = time.time()
    print("=" * 60)
    print(f"[HER+SAC] total_steps={total_steps:,} | n_envs={n_envs} | device={DEVICE}")
    print(f"Elapsed: {t1 - t0:.2f}s")
    print("=" * 60)

    # 末尾保存
    _, next_ver = find_latest_version(MODEL_PREFIX, SAVE_DIR)
    final_path = os.path.join(SAVE_DIR, f"{MODEL_PREFIX}_{next_ver}")
    model.save(final_path)
    print(f"[Save] Final model -> {final_path}")

    vec = model.get_env()
    if isinstance(vec, VecNormalize):
        vn_path = _vecnorm_path_from_ver(next_ver)
        vec.save(vn_path)
        print(f"[Save] VecNormalize -> {vn_path}")

    vec.close()

# ======= 测试（评估成功率/距离；默认 deterministic=True）=======
def test_model(model_path: Optional[str] = None, deterministic: bool = True, episodes: int = 20):
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

    # 加载 HER 模型
    model = HER.load(model_path, env=eval_env, device=DEVICE)

    succ_cnt, dist_sum, ret_sum = 0, 0.0, 0.0
    for ep in range(episodes):
        obs = eval_env.reset()
        ep_ret, ep_len = 0.0, 0
        done = [False]
        while not done[0]:
            action, _ = model.predict(obs, deterministic=deterministic)
            next_obs, reward, done, info = eval_env.step(action)
            ep_ret += float(reward[0]); ep_len += 1
            if done[0]:
                term = info[0].get("terminal_observation") or info[0].get("final_observation")
                if term is not None and hasattr(eval_env, "unnormalize_obs"):
                    term_orig = eval_env.unnormalize_obs(term)
                    ag = term_orig["achieved_goal"]; dg = term_orig["desired_goal"]
                    print(f"[Episode {ep+1}] terminal achieved_goal={ag} desired_goal={dg}")

                if info[0].get("time_limit", False):  # 修正键名
                    print(f"[Episode {ep+1}] truncated by time limit")

                succ = bool(info[0].get("success", False))
                # 如果 env 的 info 没有 pos_err，就自己算一个（反归一化后的）
                if term is not None and hasattr(eval_env, "unnormalize_obs"):
                    agp = np.asarray(ag[0][:3], dtype=np.float32)
                    dgp = np.asarray(dg[0][:3], dtype=np.float32)
                    dist = float(np.linalg.norm(agp - dgp))
                else:
                    dist = float(info[0].get("pos_err", 0.0))

                succ_cnt += int(succ); dist_sum += dist; ret_sum += ep_ret
                print(f"[Episode {ep+1}] return={ep_ret:.2f} len={ep_len} success={succ} pos_err={dist:.4f}")
                obs = eval_env.reset()
            else:
                obs = next_obs

    print(f"[Eval] episodes={episodes} | success_rate={succ_cnt/episodes:.3f} | "
          f"avg_pos_err={dist_sum/episodes:.4f} | avg_return_raw={ret_sum/episodes:.2f} | "
          f"deterministic={deterministic}")
    eval_env.close()

# ======= 入口 =======
if __name__ == "__main__":
    # 训练：
    # train(
    #     total_steps=int(2e7),
    #     n_envs=8,
    #     gamma=0.98,
    #     seed=0,
    #     save_freq=51200,
    #     resume=False,
    #     steps_per_cycle=int(5e6),
    # )
    # 测试：
    test_model(deterministic=True, episodes=20)
