# collect_lpos_from_actions_mp.py
# -*- coding: utf-8 -*-

# ---- 先限制 BLAS 线程，必须在 import numpy 之前 ----
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import itertools
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

def generate_action_grid(nu, levels=(-1.0, 0.0, 1.0), max_combos=None, seed=0):
    grid = list(itertools.product(*([levels] * nu)))
    if (max_combos is not None) and (len(grid) > max_combos):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(grid), size=max_combos, replace=False)
        grid = [grid[i] for i in idx]
    actions = np.array(grid, dtype=np.float32)
    return actions  # (M, nu)
def _worker_rollout_chunk(actions_chunk, env_kwargs, T, warmup_steps, *args):
    import os                     # ✅ 为了打印 PID
    import mujoco
    from flexpal.mujoco_cpu.flexpal_env_cpu import FlexPALSB3Env
    from flexpal.mujoco_cpu import sensors

    env = FlexPALSB3Env(**env_kwargs)
    env.reset()

    L_list, X_list, A_list, S_list = [], [], [], []

    for combo_idx, a in enumerate(actions_chunk):
        # ✅ 每 100 个动作打印一次该 worker 的进度
        if combo_idx % 100 == 0:
            print(f"[PID {os.getpid()}] {combo_idx}/{len(actions_chunk)} actions done", flush=True)

        env.reset()                          # 每个动作从同一初态开始
        a = np.asarray(a, dtype=np.float32)
        step_idx = 0

        for t in range(int(T)):
            obs, rew, term, trunc, info = env.step(a)
            L_t = env.get_lengths().copy()
            x_t = sensors.site_pos(env.data, env.tip_sid).copy()
            if t >= warmup_steps:
                L_list.append(L_t.astype(np.float32))
                X_list.append(x_t.astype(np.float32))
                A_list.append(a.copy())
                S_list.append(np.array([combo_idx, step_idx], dtype=np.int32))
                step_idx += 1

    env.close()

    if len(X_list) == 0:
        nu = env_kwargs.get("nu", 0)
        return (np.empty((0, nu), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, nu), dtype=np.float32),
                np.empty((0, 2), dtype=np.int32))

    L_arr = np.stack(L_list, axis=0)
    X_arr = np.stack(X_list, axis=0)
    A_arr = np.stack(A_list, axis=0)
    S_arr = np.stack(S_list, axis=0)
    return L_arr, X_arr, A_arr, S_arr


def _worker_star(args):
    return _worker_rollout_chunk(*args)

def main():
    import argparse, time, joblib
    from sklearn.neighbors import KDTree  # 可能后续要用，这里没用上

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str,
                        default="/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml")
    parser.add_argument("--levels", type=str, default="-1,0,1")   # 逗号分隔
    parser.add_argument("--T", type=int, default=90)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-combos", type=int, default=None)   # 子采样 action 组合数量
    parser.add_argument("--n-proc", type=int, default=None)       # 不填则用 CPU 个数
    parser.add_argument("--chunks", type=int, default=None)       # 手动设块数；默认按 n_proc 均分
    parser.add_argument("--out", type=str, default="reach_data/lpos_samples_abs_actions_mp.npz")
    args = parser.parse_args()

    # ---- Env 参数（禁用渲染，只做位置）----
    env_kwargs = dict(
        xml_path=args.xml,
        action_mode="absolute",
        render_mode=None,
        pos_only_ctrl=True,
        K_substeps=5,
        control_freq=250,
        max_episode_steps=max(100, args.T + 10),
    )

    # 先创建一个 env 只为获取 nu（在主进程里创建，随后销毁）
    from flexpal.mujoco_cpu.flexpal_env_cpu import FlexPALSB3Env
    tmp_env = FlexPALSB3Env(**env_kwargs)
    nu = tmp_env.nu
    tmp_env.close()

    # ---- 构造 actions ----
    levels = tuple(float(s) for s in args.levels.split(","))
    actions = generate_action_grid(nu, levels=levels, max_combos=args.max_combos, seed=0)
    M = len(actions)
    print(f"[config] nu={nu}, levels={levels}, total action combos={M}, "
          f"T={args.T}, warmup={args.warmup}")

    # ---- 切块 ----
    n_proc = args.n_proc or mp.cpu_count()
    if args.chunks is None:
        n_chunks = n_proc
    else:
        n_chunks = max(1, int(args.chunks))
    chunks = np.array_split(actions, n_chunks)

    # ---- 并行执行（显示总 action 进度）----
    ctx = mp.get_context("fork")  # Linux 推荐；若在 macOS/Win 要用 'spawn'
    t0 = time.time()
    L_all, X_all, A_all, S_all = [], [], [], []
    with ctx.Pool(processes=n_proc) as pool:
        tasks = [(c, env_kwargs, args.T, args.warmup) for c in chunks]
        with tqdm(total=M, desc="Sampling actions") as pbar:
            for L_arr, X_arr, A_arr, S_arr in pool.imap_unordered(_worker_star, tasks):
                if L_arr.size:
                    L_all.append(L_arr)
                    X_all.append(X_arr)
                    A_all.append(A_arr)
                    S_all.append(S_arr)
                    # 进度按“本块包含的 actions 数量”累加
                    # 每个动作产生 (T - warmup) 行样本，这里只统计 action 组合完成数
                    pbar.update(S_arr[:,0].max() + 1 if S_arr.size else 0)

    # ---- 汇总保存 ----
    if len(X_all) == 0:
        print("[warn] no samples collected.")
        return
    L_all = np.concatenate(L_all, axis=0)
    X_all = np.concatenate(X_all, axis=0)
    A_all = np.concatenate(A_all, axis=0)
    S_all = np.concatenate(S_all, axis=0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        L=L_all, X=X_all, A=A_all, S=S_all,
        levels=np.array(levels, dtype=np.float32),
        T=np.int32(args.T),
        warmup=np.int32(args.warmup),
    )
    print(f"[saved] N={len(X_all)} samples -> {args.out}")
    print(f"[time] elapsed {time.time()-t0:.1f}s; "
          f"rows per action ≈ {max(0, args.T-args.warmup)}; "
          f"total rows ≈ {len(actions)*(max(0,args.T-args.warmup))}")

if __name__ == "__main__":
    main()
