# collect_lpos_from_actions.py
import itertools
import numpy as np
import os
from tqdm import tqdm

import mujoco
from flexpal.mujoco_cpu.flexpal_env_cpu import FlexPALSB3Env
from flexpal.mujoco_cpu import sensors

def generate_action_grid(nu, levels=(-1.0, 0.0, 1.0), max_combos=None, seed=0):
    """离散动作网格（每维取 levels），可选随机子采样以防 3^nu 太大。"""
    grid = list(itertools.product(*([levels] * nu)))
    if (max_combos is not None) and (len(grid) > max_combos):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(grid), size=max_combos, replace=False)
        grid = [grid[i] for i in idx]
    actions = np.array(grid, dtype=np.float32)
    return actions  # (M, nu)

def rollout_one_action(env, action, T=200, warmup_steps=0):
    """
    对单个绝对动作向量 action 连续施加 T 步。
    - env.action_mode='absolute' 会把 [-1,1] 映射到 ctrlrange，并带你设的限速/平滑；
    - warmup_steps>0 时，前 warmup 步不记录，只用于让系统靠近目标。
    返回: (L_seq, X_seq) 分别是 (T-warmup, nu) 和 (T-warmup, 3)
    """
    # 每个动作序列从同一初态开始，保证可比
    env.reset()
    L_list, X_list = [], []
    a = np.asarray(action, dtype=np.float32)

    for t in range(T):
        # 施加恒定绝对动作（你的 env 会处理到 ctrl 的映射与 slew-rate）
        obs, rew, term, trunc, info = env.step(a)

        # 记录 ctrl(tendon length) 与 末端位置
        L_t = env.get_lengths().copy()                       # (nu,)
        x_t = sensors.site_pos(env.data, env.tip_sid).copy() # (3,)
        if t >= warmup_steps:
            L_list.append(L_t.astype(np.float32))
            X_list.append(x_t.astype(np.float32))

        # 不强制提前终止：我们要完整 200 步的时间序列覆盖
        # 如果你希望更快，只记录最后 N 步，也可以在这里判断 term 后 break

    return np.stack(L_list, axis=0), np.stack(X_list, axis=0)

def main():
    # 1) 初始化 env（禁用渲染；只做位置目标）
    env = FlexPALSB3Env(
        xml_path="/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml",
        action_mode="absolute",
        render_mode=None,
        pos_only_ctrl=True,
        K_substeps=5,           # 采样期适当降低保真换速度
        control_freq=120,
        max_episode_steps=1000,
    )
    env.reset()

    nu = env.nu
    print(f"nu={nu}, ctrl range per dim ~ [{env.L_min.min():.3f}, {env.L_max.max():.3f}]")

    # 2) 构造离散动作集合（注意 3^9=19683，可以先限制 max_combos 做子采样）
    levels = (-1.0, )      # 可以换成 (-1, -0.5, 0, 0.5, 1) 更密
    actions = generate_action_grid(nu, levels=levels, max_combos=None, seed=0)
    print(f"total action combos = {len(actions)}")

    # 3) 对每个动作做 T 步 rollout，收集 (L, x)
    T = 10
    warmup_steps = 0               # 如果想让系统先靠近目标再记录，可设 20~50
    L_all, X_all, A_all, S_all = [], [], [], []  # ctrl, pos, action, (combo_idx, step_idx)

    for combo_idx, a in enumerate(tqdm(actions, desc="Sampling actions")):
        L_seq, X_seq = rollout_one_action(env, a, T=T, warmup_steps=warmup_steps)
        steps = L_seq.shape[0]
        L_all.append(L_seq)                               # (steps, nu)
        X_all.append(X_seq)                               # (steps, 3)
        A_all.append(np.repeat(a.reshape(1, -1), steps, axis=0))  # (steps, nu)
        S_all.append(np.column_stack([                     # (steps, 2)
            np.full((steps, 1), combo_idx, dtype=np.int32),
            np.arange(steps, dtype=np.int32).reshape(-1, 1)
        ]))

    # 4) 拼接保存
    L_all = np.concatenate(L_all, axis=0)  # (N, nu)
    X_all = np.concatenate(X_all, axis=0)  # (N, 3)
    A_all = np.concatenate(A_all, axis=0)  # (N, nu)
    S_all = np.concatenate(S_all, axis=0)  # (N, 2): [combo_idx, step_idx]

    os.makedirs("reach_data", exist_ok=True)
    np.savez_compressed(
        "reach_data/lpos_samples_abs_actions.npz",
        L=L_all,            # 每步 “实际 ctrl/tendon length”
        X=X_all,            # 对应末端位置
        A=A_all,            # 施加的绝对动作向量（归一化到[-1,1]）
        S=S_all,            # [combo_idx, step_idx]
        levels=np.array(levels, dtype=np.float32),
        T=np.int32(T),
        warmup=np.int32(warmup_steps),
    )
    print(f"[saved] N={len(X_all)} samples  ->  reach_data/lpos_samples_abs_actions.npz")

    env.close()

if __name__ == "__main__":
    main()
