# reach_sampling.py
import numpy as np
from tqdm import trange
from sklearn.neighbors import KDTree
import joblib
import os
from stable_baselines3.common.vec_env import SubprocVecEnv


# ====== 基控器：任务空间 P 控制 + 数值雅可比 + 速率限 ======
class RateLimiter:
    def __init__(self, max_step: float):
        self.max_step = float(max_step)
    def __call__(self, u: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(u, ord=np.inf)
        if n <= self.max_step: return u
        return u * (self.max_step / (n + 1e-12))

def finite_diff_jac(env, l, site_id=None, eps=1e-3):
    x0 = env.get_ee_pos(site_id)
    J_cols = []
    for i in range(len(l)):
        l_pert = l.copy(); l_pert[i] += eps
        env.set_lengths(l_pert)        # 只改内部缓存，不推进仿真
        x1 = env.get_ee_pos(site_id)
        J_cols.append((x1 - x0) / eps)
    env.set_lengths(l)                 # 复原
    return np.stack(J_cols, axis=1)    # 形状 (3, nu)

class BaselineController:
    def __init__(self, site_id=None, kp=2.0, dl_rate=0.005, dl_limit=0.01, eps_jac=1e-3):
        self.site_id = site_id
        self.kp = kp
        self.dl_rate = dl_rate
        self.limiter = RateLimiter(dl_limit)
        self.eps_jac = eps_jac

    def step(self, env, x_goal):
        x = env.get_ee_pos(self.site_id)
        ex = x_goal - x                    # 任务空间误差
        v_des = self.kp * ex               # 期望末端“速度”
        l = env.get_lengths()              # 腔长
        J = finite_diff_jac(env, l, self.site_id, eps=self.eps_jac)   # (3, nu)
        # 最小二乘伪逆
        dl, *_ = np.linalg.lstsq(J, v_des, rcond=None)
        dl = self.limiter(dl * self.dl_rate)
        l_min, l_max = env.get_l_bounds()
        l_next = np.clip(l + dl, l_min, l_max)
        return l_next, float(np.linalg.norm(ex)), dl

# ====== 目标采样（均匀 + 边界偏置）======
def sample_goals(n, workspace):
    # workspace: dict {'xmin','xmax','ymin','ymax','zmin','zmax'}
    xs = np.random.uniform(workspace['xmin'], workspace['xmax'], size=(n,1))
    ys = np.random.uniform(workspace['ymin'], workspace['ymax'], size=(n,1))
    zs = np.random.uniform(workspace['zmin'], workspace['zmax'], size=(n,1))
    xyz = np.concatenate([xs, ys, zs], axis=1)

    # 一半推到外壳附近，刺激边界
    m = n // 2
    center = np.array([(workspace['xmin']+workspace['xmax'])/2,
                       (workspace['ymin']+workspace['ymax'])/2,
                       (workspace['zmin']+workspace['zmax'])/2], dtype=float)
    extents = np.array([workspace['xmax']-center[0],
                        workspace['ymax']-center[1],
                        workspace['zmax']-center[2]], dtype=float)
    shell = xyz[:m]
    vec = shell - center
    r = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9
    shell = center + vec / r * (np.random.uniform(0.8, 1.0, size=(m,1)) * np.linalg.norm(extents))
    xyz[:m] = shell
    return xyz.astype(np.float32)

# ====== VecEnv 工厂 ======
def make_env_fn(env_cls, seed, site_id=None, **env_kwargs):
    def _thunk():
        env = env_cls(site_id=site_id, **env_kwargs)
        env.seed(seed)
        return env
    return _thunk

# ====== 批量 rollouts，生成数据集 ======
def rollout_batch(vec_env, goals, controller: BaselineController,
                  T_max=300, eps_pos=0.01, sat_thresh=0.2):
    B = vec_env.num_envs
    data = []
    for i in trange(0, len(goals), B, desc="Sampling"):
        batch = goals[i:i+B]
        obs = vec_env.reset()
        for e, g in enumerate(batch):
            vec_env.env_method('set_goal', g, indices=e)

        succ = np.zeros(len(batch), dtype=bool)
        reach_t = np.full(len(batch), T_max, dtype=int)
        min_dist = np.full(len(batch), np.inf, dtype=float)
        sat_count = np.zeros(len(batch), dtype=int)

        for t in range(T_max):
            ls = []
            ds = []
            for e in range(len(batch)):
                # 在环境子进程里调用 baseline 控制一步，返回下一步腔长、距离
                l_next, dist, _ = vec_env.env_method('baseline_step', controller, batch[e], indices=e)[0]
                ls.append(l_next)
                ds.append(dist)
                min_dist[e] = min(min_dist[e], dist)

                lmin, lmax = vec_env.env_method('get_l_bounds', indices=e)[0]
                if np.any(np.isclose(l_next, lmin, atol=1e-9)) or np.any(np.isclose(l_next, lmax, atol=1e-9)):
                    sat_count[e] += 1

            actions = np.stack(ls, axis=0)
            obs, _, done, _ = vec_env.step(actions)

            arrived = (np.array(ds) <= eps_pos) & (~succ)
            succ[arrived] = True
            reach_t[arrived] = t
            if succ.all(): break

        sat_ratio = sat_count / float(T_max)
        for e, g in enumerate(batch):
            l0 = vec_env.env_method('get_start_lengths', indices=e)[0]
            data.append({
                'x_goal': g.astype(np.float32),
                'context_l': l0.astype(np.float32),
                'success': bool(succ[e]),
                'reach_time': int(reach_t[e]),
                'min_dist': float(min_dist[e]),
                'sat_ratio': float(sat_ratio[e]),
            })
    return data

# ====== 保存数据集 + 成功库 KDTree ======
def save_dataset(dataset, save_prefix):
    X_goal = np.stack([d['x_goal'] for d in dataset], axis=0)
    L_ctx  = np.stack([d['context_l'] for d in dataset], axis=0)
    y      = np.array([d['success'] for d in dataset], dtype=np.int8)
    t_reach= np.array([d['reach_time'] for d in dataset], dtype=np.int32)
    d_min  = np.array([d['min_dist'] for d in dataset], dtype=np.float32)
    sat    = np.array([d['sat_ratio'] for d in dataset], dtype=np.float32)

    np.savez_compressed(f'{save_prefix}_reach_dataset.npz',
        x_goal=X_goal, context_l=L_ctx, success=y,
        reach_time=t_reach, min_dist=d_min, sat_ratio=sat)

    S = X_goal[y==1]
    tree = KDTree(S, leaf_size=40) if len(S) > 0 else None
    joblib.dump({'S': S, 'kdtree': tree}, f'{save_prefix}_success_kdtree.pkl')
    print(f'[saved] {save_prefix}_reach_dataset.npz  | successes={len(S)}/{len(y)}')
    if tree is not None:
        print(f'[saved] {save_prefix}_success_kdtree.pkl  | dim={S.shape[1]}  N={S.shape[0]}')

# ====== 一键运行 ======
def main():
    # 1) 配置你的环境和工作空间
    from your_env_module import YourMuJoCoEnv  # TODO: 替换成你的环境
    site_id = None     # 如果有末端 site，可填 int
    n_envs = 16
    n_goals = 10000    # 先 1e4 起步，够训练一个 R(x) 了
    T_max = 300
    eps_pos = 0.01     # 1 cm 成功阈值
    workspace = dict(xmin=0.05, xmax=0.35, ymin=-0.15, ymax=0.15, zmin=0.05, zmax=0.30)

    # 2) VecEnv
    env_fns = [make_env_fn(YourMuJoCoEnv, seed=1000+i, site_id=site_id) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # 3) 控制器参数（按你的系统实际改）
    controller = BaselineController(site_id=site_id, kp=2.0, dl_rate=0.005, dl_limit=0.01, eps_jac=1e-3)

    # 4) 采目标 & rollout
    goals = sample_goals(n_goals, workspace)
    dataset = rollout_batch(vec_env, goals, controller, T_max=T_max, eps_pos=eps_pos, sat_thresh=0.2)

    # 5) 存档
    os.makedirs('reach_data', exist_ok=True)
    save_dataset(dataset, save_prefix='reach_data/mjx')

    vec_env.close()

if __name__ == "__main__":
    main()
