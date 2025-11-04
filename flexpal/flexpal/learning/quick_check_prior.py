# quickcheck_prior.py
import os
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree

# === 你的 env & 传感器 ===
import mujoco
from flexpal.mujoco_cpu.flexpal_env_cpu import FlexPALSB3Env
from flexpal.mujoco_cpu import sensors

# ==== 1) 加载先验库（你采好的 L, X） ====
PRIOR_PATH = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/lpos_samples_abs_actions_mp_59.npz"   
XML_PATH   = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml"

# 评测参数
TOL_POS     = 0.02    # 1 cm 判定可达
N_GOALS     = 1000    # 随机目标数量（用于投影测试 + 方向测试）
N_STEPS_DIR = 100      # 方向性测试：每个目标执行的步数（只用先验动作）
CONTROL_FREQ= 250
K_SUBSTEPS  = 5

def load_prior(path):
    D = np.load(path)
    L = D["L"]  # (N, nu) 实际 ctrl/腱长
    X = D["X"]  # (N, 3)  末端位置
    # 轻量去重（可选）：避免 KDTree 里大量近重复点
    vox = np.floor(X / 1e-3).astype(np.int32)
    _, keep = np.unique(vox, axis=0, return_index=True)
    L = L[keep]; X = X[keep]
    tree = KDTree(X, leaf_size=40)
    return L.astype(np.float32), X.astype(np.float32), tree

def sample_workspace_goals(n, workspace):
    xs = np.random.uniform(workspace['xmin'], workspace['xmax'], size=(n,1))
    ys = np.random.uniform(workspace['ymin'], workspace['ymax'], size=(n,1))
    zs = np.random.uniform(workspace['zmin'], workspace['zmax'], size=(n,1))
    return np.concatenate([xs,ys,zs], axis=1).astype(np.float32)

def check_projection(X, tree, tol, n_goals=1000, inset=0.05):
    mins, maxs = X.min(0), X.max(0)
    # inset>0 表示向内收缩（例如 0.01 -> 四周缩 1 cm）
    mins = mins + inset
    maxs = maxs - inset
    G = sample_workspace_goals(n_goals, dict(
        xmin=mins[0], xmax=maxs[0], ymin=mins[1], ymax=maxs[1], zmin=mins[2], zmax=maxs[2]
    ))
    dist, idx = tree.query(G, k=1)
    dist = dist.ravel()
    return {
        "coverage": float((dist <= tol).mean()),
        "dist_mean": float(dist.mean()),
        "dist_p90": float(np.percentile(dist, 90)),
    }, G, idx.reshape(-1), dist

def L_to_absolute(L, L_min, L_max):
    span = (L_max - L_min)
    return np.clip(2.0*(L - L_min)/(span + 1e-12) - 1.0, -1.0, 1.0)

def check_direction(xml_path, L_set, X_set, tree, tol, n_goals=200,
                    n_steps=10, control_freq=120, k_substeps=5):
    env = FlexPALSB3Env(xml_path=xml_path, action_mode="absolute",
                        render_mode=None, pos_only_ctrl=True,
                        control_freq=control_freq, K_substeps=k_substeps,
                        max_episode_steps=1000)
    env.reset()

    # 定义评测工作空间（用库边界）
    mins = X_set.min(axis=0); maxs = X_set.max(axis=0)
    pad  = 0.02
    workspace = dict(xmin=mins[0]-pad, xmax=maxs[0]+pad,
                     ymin=mins[1]-pad, ymax=maxs[1]+pad,
                     zmin=max(0.0, mins[2]-pad), zmax=maxs[2]+pad)
    goals = sample_workspace_goals(n_goals, workspace)

    # 统计
    single_step_improve = []
    multi_step_improve  = []
    success_by_prior    = 0

    for g in tqdm(goals, desc="Direction test"):
        # 把目标设给 env（pos-only）
        env.set_goal(g)
        # 用库投影得到先验 L_prior
        d, idx = tree.query(g.reshape(1,-1), k=8)
        W = 1.0/(d[0] + 1e-6); W /= W.sum()
        Lk = L_set[idx[0]]
        L_prior = (W[:,None] * Lk).sum(0).astype(np.float32)

        # 计算先验对应的 absolute 动作（一次固定；不加 RL 残差）
        a_prior = L_to_absolute(L_prior, env.L_min, env.L_max)

        # reset + （可选）沉降到当前 ctrl（这里直接用默认重置，不做 settle）
        env.reset()
        x = sensors.site_pos(env.data, env.tip_sid).astype(np.float32)
        d_prev = float(np.linalg.norm(x - g))

        # 执行 1 步：看单步改进
        obs, rew, term, trunc, info = env.step(a_prior)
        x1 = sensors.site_pos(env.data, env.tip_sid).astype(np.float32)
        d_curr = float(np.linalg.norm(x1 - g))
        single_step_improve.append(d_prev - d_curr)  # >0 则朝目标前进

        # 继续执行 n_steps-1 步：看多步平均改进
        d_last = d_curr
        for _ in range(max(0, n_steps-1)):
            obs, rew, term, trunc, info = env.step(a_prior)
            xk = sensors.site_pos(env.data, env.tip_sid).astype(np.float32)
            dk = float(np.linalg.norm(xk - g))
            d_last = dk
        multi_step_improve.append(d_prev - d_last)

        if d_last <= tol:
            success_by_prior += 1

    env.close()
    dir_stats = dict(
        single_step_improve_mean=float(np.mean(single_step_improve)),
        single_step_improve_p10=float(np.percentile(single_step_improve, 10)),
        single_step_improve_p90=float(np.percentile(single_step_improve, 90)),
        multi_step_improve_mean=float(np.mean(multi_step_improve)),
        multi_step_improve_p10=float(np.percentile(multi_step_improve, 10)),
        multi_step_improve_p90=float(np.percentile(multi_step_improve, 90)),
        success_rate_prior=float(success_by_prior/len(goals))
    )
    return dir_stats

def main():
    print("[load] prior dataset:", PRIOR_PATH)
    L, X, tree = load_prior(PRIOR_PATH)
    print(f"[prior] L:{L.shape}, X:{X.shape}")

    # 1) 投影有效性
    proj_stats, G, idx, dist = check_projection(X, tree, tol=TOL_POS, n_goals=N_GOALS)
    print("\n[Projection]")
    for k, v in proj_stats.items():
        print(f"  {k}: {v:.4f}")
    # 预期：coverage 高（比如 >0.8~0.9），dist_mean 小

    # 2) 方向有效性（只用先验动作，不加 RL 残差）
    dir_stats = check_direction(XML_PATH, L_set=L, X_set=X, tree=tree,
                                tol=TOL_POS, n_goals=min(N_GOALS, 200),
                                n_steps=N_STEPS_DIR, control_freq=CONTROL_FREQ,
                                k_substeps=K_SUBSTEPS)
    print("\n[Direction]")
    for k, v in dir_stats.items():
        print(f"  {k}: {v:.4f}")

    print("\n[Interpretation]")
    print("  - Projection: coverage 越接近 1 越好；dist_mean/p90 越小越好。")
    print("  - Direction: single/multi step improve 的均值 > 0 表示先验动作给了正确方向；")
    print("               success_rate_prior 表示只靠先验能到阈值的比例（通常不要求很高）。")

if __name__ == "__main__":
    main()
