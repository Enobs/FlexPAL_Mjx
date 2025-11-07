# flexpal_sb3_env.py
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from typing import Tuple, Dict, Optional

from flexpal.mujoco_cpu.renderers import MjRenderer
import flexpal.mujoco_cpu.idbuild as idbuild
from flexpal.mujoco_cpu import sensors


# ---------- utils ----------
def quat_geodesic_angle_np(q1: np.ndarray, q2: np.ndarray, eps: float = 1e-8) -> float:
    q1 = q1 / (np.linalg.norm(q1) + eps)
    q2 = q2 / (np.linalg.norm(q2) + eps)
    c = np.clip(abs(float(np.dot(q1, q2))), 0.0, 1.0)
    return 2.0 * math.acos(c)


def reaching_reward_np(
    data: mujoco.MjData,
    goal: np.ndarray,
    goal_dim: int,
    tip_site_id: int,
    w_pos: float = 1.0,
    w_ori: float = 0.2,
    clip_range: float = 50.0,
) -> Tuple[float, float, float, float]:
    ee_pos  = sensors.site_pos(data, tip_site_id)
    ee_quat = sensors.site_quat_world(data, tip_site_id)
    ee_quat = ee_quat / (np.linalg.norm(ee_quat) + 1e-8)

    pos_err = float(np.linalg.norm(ee_pos - goal[:3]))
    if goal_dim == 3:
        base_r  = -(w_pos * pos_err)
        ang_err = 0.0
    else:
        ang_err = quat_geodesic_angle_np(ee_quat, goal[3:])
        base_r  = -(w_pos * pos_err + w_ori * ang_err)

    reward = float(np.clip(base_r, -clip_range, clip_range))
    return reward, pos_err, ang_err,


class FlexPALSB3Env(gym.Env):
    """
    SB3 可用环境（CPU MuJoCo）：
    - 动作: [-1,1]^nu -> actuator ctrlrange（限步 + 可选平滑）
    - 物理: 每 step 执行 n_frames * K_substeps 次 mj_step
    - 目标: desired_goal = [pos(3), quat(4)]，achieved_goal 同格式（7D）
    - 终止: 连续 hold_steps 命中阈值 => terminated=True；超步数 => truncated=True
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        xml_path: str,
        control_freq: float = 250.0,
        tol_pos: float = 0.015,           # 目标精度：1.5 cm
        tol_ang: float = 1e-1,
        hold_steps: int = 2,
        alpha_smooth: float = 0.0,
        action_mode: str = "absolute",
        dL_max: float = 0.10,             # 步长放大，便于“冲线”（0.08~0.15 可调）
        w_pos: float = 1.0,
        w_ori: float = 0.2,
        # ---- 新增：直观奖励参数 ----
        reward_mode: str = "exp",         # ["exp", "piecewise", "l2"]
        r_bonus: float = 20.0,            # 命中爆发奖励
        near_gain: float = 1.0,           # 近端进步增益
        scale: float = 0.03,              # 奖励尺度 ~3cm（近端陡峭）
        prog_coef: float = 0.5,           # 进步奖励系数
        du_coef_far: float = 1e-3,        # 远端动作惩罚系数
        du_stop_near: float = 5.0,        # <5×tol_pos 关闭动作惩罚
        # --------------------------------
        w_du: float = 1e-2,               # 兼容旧参数（已不直接使用）
        K_substeps: int = 5,              # 内步减少（提升每步位移）
        max_episode_steps: int = 100,
        render_mode: Optional[str] = None,
        render_width: int = 960,
        render_height: int = 720,
        render_camera: Optional[str] = None,
        step_penalty: float = 0.001,     
        use_time_scale: bool = False,
        pos_only_ctrl: bool = True,
        goal_library_path: Optional[str] = None,
        goal_voxel: Optional[float] = 0.005,
        # 课程采样（精度圈）
        curriculum_precision_prob: float = 0.2,  # 20% 采样到近端小幅目标
        curriculum_r_small: float = 0.03,        # 3 cm 精度圈
        curriculum_r_large: float = 0.12,        # 12 cm 常规圈，后续可放宽
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        # name -> id
        act_names  = idbuild.gen_actuator_names()
        ten_names  = idbuild.gen_tendon_names()
        site_names = idbuild.gen_site_names()
        self.cids, self.ids = idbuild.build_ids(
            self.model, sites=site_names, tendons=ten_names, actuators=act_names
        )
        self.ids_act  = self.ids.actuator
        self.ids_ten  = self.ids.tendon
        self.ids_site = self.ids.site
        self.tip_sid  = int(self.ids_site[-1])

        # control discretization
        model_dt      = float(self.model.opt.timestep)
        self.n_frames = max(1, int(round((1.0 / control_freq) / model_dt)))
        self.K_substeps = int(K_substeps)
        self.control_period = self.n_frames * model_dt

        # ranges / sizes
        ctrl_rng      = self.model.actuator_ctrlrange.astype(np.float32)
        self.ctrl_min = ctrl_rng[:, 0]
        self.ctrl_max = ctrl_rng[:, 1]
        self.L_min    = self.ctrl_min.copy()
        self.L_max    = self.ctrl_max.copy()
        self.nu       = int(self.model.nu)

        # action mode
        self.action_mode = str(action_mode).lower()
        assert self.action_mode in ("relative", "absolute")

        # reward/episode params
        self.tol_pos = float(tol_pos)
        self.tol_ang = float(tol_ang)
        self.hold_steps = int(hold_steps)
        self.alpha  = float(alpha_smooth)
        self.dL_max = float(dL_max)
        self.w_pos = float(w_pos)
        self.w_ori = float(w_ori)
        self.gamma_rl = 0.98
        self.pbk = 50.0   

        self.reward_mode  = str(reward_mode)
        self.r_bonus      = float(r_bonus)
        self.near_gain    = float(near_gain)
        self.scale        = float(scale)
        self.prog_coef    = float(prog_coef)
        self.du_coef_far  = float(du_coef_far)
        self.du_stop_near = float(du_stop_near)

        self.max_episode_steps = int(max_episode_steps)
        self.step_penalty = float(step_penalty)
        self.use_time_scale = bool(use_time_scale)

        # precision curriculum
        self.curriculum_precision_prob = float(curriculum_precision_prob)
        self.curriculum_r_small = float(curriculum_r_small)
        self.curriculum_r_large = float(curriculum_r_large)

        # state trackers
        self._prev_pos_err = None
        self._timestep = 0
        self._success_streak = 0

        if pos_only_ctrl:
            g = np.array([-0.201, 0.149, 0.840], dtype=np.float32)
            self.goal = g
            self.goal_dim = 3
        else:
            g = np.array([-0.201, 0.149, 0.840, 0.657, 0.242, 0.683, 0.204], dtype=np.float32)
            g[3:] /= (np.linalg.norm(g[3:]) + 1e-8)
            self.goal = g
            self.goal_dim = 7

        self._goal_lib: Optional[np.ndarray] = None
        if goal_library_path is not None:
            try:
                D = np.load(goal_library_path)
                X = D["X"].astype(np.float32)
                if goal_voxel is not None:
                    vox = np.floor(X / float(goal_voxel)).astype(np.int32)
                    _, keep = np.unique(vox, axis=0, return_index=True)
                    X = X[keep]
                self._goal_lib = X
                mins, maxs = X.min(axis=0), X.max(axis=0)
                pad = 0.005
                self.workspace = dict(
                    xmin=float(mins[0]-pad), xmax=float(maxs[0]+pad),
                    ymin=float(mins[1]-pad), ymax=float(maxs[1]+pad),
                    zmin=float(max(0.0, mins[2]-pad)), zmax=float(maxs[2]+pad),
                )
                print(f"[Env] goal library loaded: {len(X)} points; workspace={self.workspace}")
            except Exception as e:
                print(f"[Env] failed to load goal library: {e}")
                self._goal_lib = None
                self.workspace = None
        else:
            self.workspace = None

        # spaces
        imu_len = int(len(self.ids_ten) * 6)
        obs_dim = len(self.ids_ten) + imu_len + self.goal_dim
        self._imu_len = imu_len
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(self.goal_dim,), dtype=np.float32),
            "desired_goal":  spaces.Box(low=-np.inf, high=np.inf, shape=(self.goal_dim,), dtype=np.float32),
        })

        # rendering
        self.render_mode   = render_mode
        self._render_w     = int(render_width)
        self._render_h     = int(render_height)
        self._render_cam   = render_camera
        self._renderer: Optional[MjRenderer] = None
        self._last_rgb: Optional[np.ndarray] = None

        # init ctrl
        self._set_ctrl(np.ones(self.nu, dtype=np.float32) * 0.29)

        if self.render_mode == "human":
            self._renderer = MjRenderer(self.model, self.data, mode="human",
                                        width=self._render_w, height=self._render_h, camera=self._render_cam)
        elif self.render_mode == "rgb_array":
            self._renderer = MjRenderer(self.model, self.data, mode="rgb_array",
                                        width=self._render_w, height=self._render_h, camera=self._render_cam)

    # ---------- helpers ----------
    def _set_ctrl(self, ctrl_vec: np.ndarray):
        self.data.ctrl[:self.nu] = ctrl_vec[:self.nu]

    def _step_physics(self):
        for _ in range(self.n_frames):
            for _ in range(self.K_substeps):
                mujoco.mj_step(self.model, self.data)

    def _obs_from_data(self) -> Dict[str, np.ndarray]:
        tendon = sensors.tendon_state(self.data, self.ids_ten).astype(np.float32)
        sens   = np.asarray(self.data.sensordata, dtype=np.float32)
        if sens.size == 0:
            sens = np.zeros((self._imu_len,), dtype=np.float32)
        elif sens.size < self._imu_len:
            sens = np.pad(sens, (0, self._imu_len - sens.size))

        obsv = np.concatenate([tendon, sens, self.goal]).astype(np.float32)

        ee_pos  = sensors.site_pos(self.data, self.tip_sid).astype(np.float32)
        ee_quat = sensors.site_quat_world(self.data, self.tip_sid).astype(np.float32)

        if self.goal_dim == 7:
            achieved = np.concatenate([ee_pos, ee_quat]).astype(np.float32)
            desired  = self.goal.astype(np.float32)
        else:
            achieved = ee_pos.astype(np.float32)
            desired  = self.goal.astype(np.float32)

        return {"observation": obsv, "achieved_goal": achieved, "desired_goal": desired}

    # —— 新增：直观奖励 —— #
    def _compute_reward(self, pos_err: float, ang_err: float, delta: float,
                        u_new: np.ndarray, u_prev: np.ndarray) -> float:
        # 形状项：指数靠近（近端陡峭）
        if self.reward_mode == "exp":
            r_shape = math.exp(-pos_err / max(1e-6, self.scale))
        elif self.reward_mode == "piecewise":
            s = pos_err / max(1e-6, self.scale)
            r_shape = 1.0 / (1.0 + s * s)  # 近端更陡
        else:
            r_shape = 1.0 / (1.0 + pos_err / max(1e-6, self.scale))

        # 进步项：靠近就加分，越近权重越大
        near = max(0.0, 1.0 - pos_err / (2.0 * self.tol_pos))
        r_prog = self.prog_coef * max(min(delta, self.tol_pos), -self.tol_pos)
        r_prog *= (1.0 + self.near_gain * near)

        # 动作变化惩罚：远端惩罚，近端放开
        du_pen = float(np.mean((u_new - u_prev) ** 2))
        r_du = - self.du_coef_far * du_pen if pos_err > self.du_stop_near * self.tol_pos else 0.0

        return r_shape + r_prog + r_du

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._success_streak = 0
        self._timestep = 0
        self._prev_pos_err = None
        self._set_ctrl(np.ones(self.nu, dtype=np.float32) * 0.29)
        self.l0 = self.data.ctrl[:self.nu].copy()
        self._step_physics()

        # 课程采样（优先近端精度练习）
        if self._goal_lib is not None:
            x0 = sensors.site_pos(self.data, self.tip_sid)
            if self.np_random.random() < self.curriculum_precision_prob:
                R = self.curriculum_r_small
            else:
                R = self.curriculum_r_large
            d  = np.linalg.norm(self._goal_lib - x0[None, :], axis=1)
            cand = np.flatnonzero(d < R)
            if cand.size > 0:
                idx = int(self.np_random.integers(cand.size))
                goal_pos = self._goal_lib[cand[idx]]
            else:
                idx = int(self.np_random.integers(self._goal_lib.shape[0]))
                goal_pos = self._goal_lib[idx]
            if self.goal_dim == 3:
                self.set_goal(goal_pos.astype(np.float32))
            else:
                g7 = np.empty(7, dtype=np.float32)
                g7[:3] = goal_pos
                g7[3:] = np.array([1, 0, 0, 0], np.float32)
                self.set_goal(g7)

        if self._renderer is not None:
            self._renderer.update(self.model, self.data)
            if self.render_mode == "rgb_array":
                self._last_rgb = self._renderer.render()

        obs = self._obs_from_data()
        assert isinstance(obs, dict)
        assert obs["achieved_goal"].shape == (self.goal_dim,)
        assert obs["desired_goal"].shape  == (self.goal_dim,)
        return obs, {}

    def step(self, action: np.ndarray):
        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0)
        self._timestep += 1

        L_prev = self.data.ctrl[:self.nu].copy()
        span   = (self.L_max - self.L_min).astype(np.float32)
        dL_cap = self.dL_max * span

        if self.action_mode == "absolute":
            L_abs = self.L_min + 0.5 * (a + 1.0) * span
            room_up   = self.L_max - L_prev
            room_down = L_prev   - self.L_min
            # 放宽到 1.0 * room_up/down，便于“冲线”
            step_pos  = np.minimum(np.maximum(L_abs - L_prev, 0.0), 1.0 * room_up)
            step_neg  = -np.minimum(np.maximum(L_prev - L_abs, 0.0), 1.0 * room_down)
            dL        = np.clip(step_pos + step_neg, -dL_cap, dL_cap)
            L_target  = np.clip(L_prev + dL, self.L_min, self.L_max)
        else:
            dL = a * dL_cap
            room_up   = self.L_max - L_prev
            room_down = L_prev   - self.L_min
            dL_pos    = np.minimum(np.maximum(dL, 0.0), 1.0 * room_up)
            dL_neg    = -np.minimum(np.maximum(-dL, 0.0), 1.0 * room_down)
            L_target  = np.clip(L_prev + dL_pos + dL_neg, self.L_min, self.L_max)

        u_new = self.alpha * L_prev + (1.0 - self.alpha) * L_target
        self._set_ctrl(u_new)
        self._step_physics()

        # 误差
        _, pos_err, ang_err = reaching_reward_np(
            self.data, self.goal, self.goal_dim, self.tip_sid,
            w_pos=self.w_pos, w_ori=self.w_ori)

        prev_err = self._prev_pos_err if self._prev_pos_err is not None else pos_err
        # 势能：Phi(s) = -k * dist(s)
        phi_prev = - self.pbk * prev_err
        phi_curr = - self.pbk * pos_err
        r_potential = self.gamma_rl * phi_curr - phi_prev

        self._prev_pos_err = pos_err

        reward = r_potential
        reward -= self.step_penalty * (self.control_period if self.use_time_scale else 1.0)

        if self.goal_dim == 3:
            ok = (pos_err < self.tol_pos)
        else:
            ok = (pos_err < self.tol_pos) and (ang_err < self.tol_ang)

        self._success_streak = self._success_streak + 1 if ok else 0
        terminated = bool(self._success_streak >= self.hold_steps)
        if terminated:
            reward += self.r_bonus

        truncated = bool(self._timestep >= self.max_episode_steps)

        obs = self._obs_from_data()
        success = bool(self._success_streak >= self.hold_steps)  # 与 truncated 无关
        info = {
            "pos_err": float(pos_err),
            "ang_err": float(ang_err),
            "time_limit": truncated,
            "success": success,
        }

        if self._renderer is not None:
            self._renderer.update(self.model, self.data)
            if self.render_mode == "human":
                self._renderer.render()
            elif self.render_mode == "rgb_array":
                self._last_rgb = self._renderer.render()

        return obs, float(reward), bool(terminated), bool(truncated), info

    # ---------- external helpers ----------
    def set_goal(self, x_goal: np.ndarray):
        x_goal = np.asarray(x_goal, dtype=float)
        if x_goal.shape[0] == 3:
            g = np.empty(3, dtype=float); g[:3] = x_goal; self.goal = g
        elif x_goal.shape[0] == 7:
            g = x_goal.copy()
            if not np.isnan(g[3:]).all():
                q = g[3:]; g[3:] = q / (np.linalg.norm(q) + 1e-8)
            self.goal = g
        else:
            raise ValueError("x_goal must be shape (3,) or (7,)")

    def get_l_bounds(self):
        return self.L_min, self.L_max

    def close(self):
        try:
            if self._renderer is not None:
                self._renderer.close()
        except Exception:
            pass
        self._renderer = None
        
    def compute_reward(self, achieved_goal, desired_goal, info):
        ag = np.asarray(achieved_goal)[..., :3]
        dg = np.asarray(desired_goal)[..., :3]
        dist = np.linalg.norm(ag - dg, axis=-1)
        reward = -dist
        reward = np.where(dist < self.tol_pos, reward + self.r_bonus, reward)
        return reward



if __name__ == "__main__":
    env = FlexPALSB3Env(
        xml_path="/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml",
        action_mode="absolute",
        render_mode="human",
        K_substeps=5,
        max_episode_steps=300,
    )
    obs, info = env.reset()
    a = [0, 0]
    action = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float32)[:env.nu]
    while True:
        obs, rew, term, trunc, info = env.step(action)
        a[1] += rew; a[0] += 1
        if term or trunc:
            break
    print("done:", term, "truncated:", trunc, "reward:", rew, "iter:", a[0], "rew_sum=", a[1])
