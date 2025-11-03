# flexpal_sb3_env.py
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from typing import Tuple, Dict, Optional

from renderers import MjRenderer 

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
    model: mujoco.MjModel,
    goal: np.ndarray,
    goal_dim: int,        
    tip_site_id: int,
    w_pos: float = 1.0,
    w_ori: float = 0.2,
    prev_data: Optional[mujoco.MjData] = None,
    tol_pos: float = 5e-3,
    tol_ang: float = 5e-2,
    w_improve: float = 0.5,
    w_soft: float = 0.5,
    ang_improve_scale: float = 0.3,
    clip_range: float = 50.0,
) -> Tuple[float, float, float, float]:
    ee_pos  = sensors.site_pos(data, tip_site_id)
    ee_quat = sensors.site_quat_world(data, tip_site_id)
    ee_quat = ee_quat / (np.linalg.norm(ee_quat) + 1e-8)

    pos_err = float(np.linalg.norm(ee_pos - goal[:3]))
    if goal_dim == 3:
        base_r  = -(w_pos * pos_err)
        ang_err = 0
    else:
        ang_err = quat_geodesic_angle_np(ee_quat, goal[3:])
        base_r  = -(w_pos * pos_err + w_ori * ang_err)

    improve = 0.0
    if prev_data is not None:
        p0  = sensors.site_pos(prev_data, tip_site_id)
        q0  = sensors.site_quat_world(prev_data, tip_site_id)
        q0  = q0 / (np.linalg.norm(q0) + 1e-8)
        pos0 = float(np.linalg.norm(p0 - goal[:3]))
        if goal_dim == 7:
            ang0 = quat_geodesic_angle_np(q0, goal[3:])
        # improve = (pos0 - pos_err) + ang_improve_scale * (ang0 - ang_err)

    def _sigmoid_stable(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        ez = np.exp(z)
        return ez / (1.0 + ez)

    def smooth_success(e, tol):
        x = (tol - e) / (tol + 1e-6)
        return _sigmoid_stable(6.0 * x)

    # soft = 0.7 * smooth_success(pos_err, tol_pos) + 0.3 * smooth_success(ang_err, tol_ang)
    soft = 0
    reward = float(np.clip(base_r + w_improve * improve + w_soft * soft, -clip_range, clip_range))
    return reward, pos_err, ang_err, soft


class FlexPALSB3Env(gym.Env):
    """
    SB3 可用环境（CPU MuJoCo）：
    - 动作: [-1,1]^nu -> 映射到 actuator ctrlrange，带限步 + 一阶平滑
    - 物理: 每个 step 执行 n_frames * K_substeps 次 mj_step
    - 目标: desired_goal = [pos(3), quat(4)]，achieved_goal 同格式（7D）
    - 终止: 连续 hold_steps 满足阈值 => terminated=True；超步数 => truncated=True
    - 渲染: human/rgb_array，一次性创建、重复复用
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        xml_path: str,
        control_freq: float = 250.0,
        tol_pos: float = 1e-2,
        tol_ang: float = 1e-1,
        hold_steps: int = 3,
        alpha_smooth: float = 0.0,
        action_mode: str = "absolute",
        dL_max: float = 0.06,
        w_pos: float = 1.0,
        w_ori: float = 0.2,
        w_du: float = 1e-2,
        r_bonus: float = 8.0,
        K_substeps: int = 10,
        max_episode_steps: int = 100,
        render_mode: Optional[str] = None,
        render_width: int = 960,
        render_height: int = 720,
        render_camera: Optional[str] = None,
        step_penalty: float = 0.01,   
        use_time_scale: bool = False, 
        pos_only_ctrl = True,
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
        self.w_du  = float(w_du)
        self.r_bonus = float(r_bonus)
        self.max_episode_steps = int(max_episode_steps)

        # step penalty
        self.step_penalty = float(step_penalty)
        self.use_time_scale = bool(use_time_scale)

        if pos_only_ctrl: 
            g = np.array([-0.201, 0.149, 0.840], dtype=np.float32)
            self.goal = g
            self.goal_dim = 3
        else:
            g = np.array([-0.201, 0.149, 0.840, 0.657, 0.242, 0.683, 0.204], dtype=np.float32)
            g[3:] /= (np.linalg.norm(g[3:]) + 1e-8)
            self.goal = g
            self.goal_dim = 7

        # spaces
        imu_len = int(len(self.ids_ten) * 6)
        obs_dim = len(self.ids_ten) + imu_len + self.goal_dim
        self._imu_len = imu_len
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(self.goal_dim,),    dtype=np.float32),
            "desired_goal":  spaces.Box(low=-np.inf, high=np.inf, shape=(self.goal_dim,),    dtype=np.float32),
        })

        # rendering
        self.render_mode   = render_mode
        self._render_w     = int(render_width)
        self._render_h     = int(render_height)
        self._render_cam   = render_camera
        self._renderer: Optional[MjRenderer] = None
        self._last_rgb: Optional[np.ndarray] = None

        # counters
        self._success_streak = 0
        self._timestep = 0

        # init ctrl
        self._set_ctrl(np.ones(self.nu, dtype=np.float32) * 0.29)

        # optional create viewer/renderer once
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

    def _clone_core_state(self) -> mujoco.MjData:
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.data.qpos
        d.qvel[:] = self.data.qvel
        if self.nu > 0:
            d.ctrl[:self.nu] = self.data.ctrl[:self.nu]
        mujoco.mj_forward(self.model, d)
        return d

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
            achieved = np.concatenate([ee_pos, ee_quat]).astype(np.float32)     # 7
            desired  = self.goal.astype(np.float32)                              # 7
        else:
            achieved = ee_pos.astype(np.float32)                                 # 3
            desired  = self.goal.astype(np.float32)                              # 3

        return {"observation": obsv, "achieved_goal": achieved, "desired_goal": desired}
    
    def estimate_jacobian_fd(self, eps: float = 1e-3, K: int = 5) -> np.ndarray:

        nu = self.nu
        J = np.zeros((3, nu), dtype=np.float32)

        base = mujoco.MjData(self.model)
        base.qpos[:] = self.data.qpos
        base.qvel[:] = self.data.qvel
        base.act[:]  = self.data.act
        base.ctrl[:] = self.data.ctrl
        mujoco.mj_forward(self.model, base)
        for _ in range(self.K_substeps if K is None else K):
            mujoco.mj_step(self.model, base)
        x0 = sensors.site_pos(base, self.tip_sid).astype(np.float32)

        # 2) 对每个通道做 +eps 微扰，并同样推进 K 步
        for i in range(nu):
            dtmp = mujoco.MjData(self.model)
            dtmp.qpos[:] = self.data.qpos
            dtmp.qvel[:] = self.data.qvel
            dtmp.act[:]  = self.data.act
            dtmp.ctrl[:] = self.data.ctrl
            mujoco.mj_forward(self.model, dtmp)
            ctrl = dtmp.ctrl[:nu].copy()
            # eps 建议按 ctrlrange 的比例来取，避免过小/过大
            rng = (self.L_max[i] - self.L_min[i])
            de  = eps if eps is not None else 1e-3
            de  = max(de, 1e-4 * rng)  # 至少是范围的 0.01%
            ctrl[i] = np.clip(ctrl[i] + de, self.L_min[i], self.L_max[i])
            dtmp.ctrl[:nu] = ctrl

            for _ in range(self.K_substeps if K is None else K):
                mujoco.mj_step(self.model, dtmp)

            x1 = sensors.site_pos(dtmp, self.tip_sid).astype(np.float32)
            J[:, i] = (x1 - x0) / de

        return J


    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._success_streak = 0
        self._timestep = 0
        self._set_ctrl(np.ones(self.nu, dtype=np.float32) * 0.29)
        self.l0 = self.data.ctrl[:self.nu].copy()
        self._step_physics()
        if self._renderer is not None:
            self._renderer.update(self.model, self.data)
            if self.render_mode == "rgb_array":
                self._last_rgb = self._renderer.render()

        obs = self._obs_from_data()
        info = {}
        return obs, info

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
            step_pos  = np.minimum(np.maximum(L_abs - L_prev, 0.0), 0.5 * room_up)
            step_neg  = -np.minimum(np.maximum(L_prev - L_abs, 0.0), 0.5 * room_down)
            dL        = np.clip(step_pos + step_neg, -dL_cap, dL_cap)
            L_target  = np.clip(L_prev + dL, self.L_min, self.L_max)
        else:
            dL = a * dL_cap
            room_up   = self.L_max - L_prev
            room_down = L_prev   - self.L_min
            dL_pos    = np.minimum(np.maximum(dL, 0.0), 0.5 * room_up)
            dL_neg    = -np.minimum(np.maximum(-dL, 0.0), 0.5 * room_down)
            L_target  = np.clip(L_prev + dL_pos + dL_neg, self.L_min, self.L_max)

        u_new = self.alpha * L_prev + (1.0 - self.alpha) * L_target

        prev_snapshot = self._clone_core_state() 
        self._set_ctrl(u_new)
        self._step_physics()

        r_shape, pos_err, ang_err, soft_succ = reaching_reward_np(
            self.data, self.model, self.goal, self.goal_dim, self.tip_sid,
            w_pos=self.w_pos, w_ori=self.w_ori,
            prev_data=prev_snapshot, tol_pos=self.tol_pos, tol_ang=self.tol_ang,
            w_improve=0.5, w_soft=0.5, ang_improve_scale=0.3
        )

        du_pen = float(np.mean((u_new - L_prev) ** 2))
        reward = r_shape - self.w_du * du_pen

        if self.use_time_scale:
            reward -= self.step_penalty * self.control_period
        else:
            reward -= self.step_penalty
            
        if self.goal_dim ==3:
            ok = (pos_err < self.tol_pos)
        else:
            ok = (pos_err < self.tol_pos) and (ang_err < self.tol_ang)
        self._success_streak = self._success_streak + 1 if ok else 0
        terminated = bool(self._success_streak >= self.hold_steps)
        if terminated:
            reward += self.r_bonus

        truncated = bool(self._timestep >= self.max_episode_steps)

        obs = self._obs_from_data()
        info = {
            "pos_err": pos_err,
            "ang_err": ang_err,
            "soft_succ": soft_succ,
            "time_limit": truncated,
        }

        if self._renderer is not None:
            self._renderer.update(self.model, self.data)
            if self.render_mode == "human":
                self._renderer.render()
            elif self.render_mode == "rgb_array":
                self._last_rgb = self._renderer.render()

        return obs, reward, terminated, truncated, info

    # ---------- Render API ----------
    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                self._renderer = MjRenderer(self.model, self.data, mode="human",
                                            width=self._render_w, height=self._render_h, camera=self._render_cam)
            self._renderer.update(self.model, self.data)
            self._renderer.render()
            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = MjRenderer(self.model, self.data, mode="rgb_array",
                                            width=self._render_w, height=self._render_h, camera=self._render_cam)
            self._renderer.update(self.model, self.data)
            self._last_rgb = self._renderer.render()
            return None if self._last_rgb is None else self._last_rgb.copy()
        return None

    def set_goal(self, x_goal: np.ndarray):
        x_goal = np.asarray(x_goal, dtype=float)
        if x_goal.shape[0] == 3:
            g = np.empty(3, dtype=float)
            g[:3] = x_goal
            self.goal = g
        elif x_goal.shape[0] == 7:
            g = x_goal.copy()
            if not np.isnan(g[3:]).all():
                q = g[3:]
                g[3:] = q / (np.linalg.norm(q) + 1e-8)
            self.goal = g
        else:
            raise ValueError("x_goal must be shape (3,) or (7,)")


    def get_start_lengths(self) -> np.ndarray:
        return self.l0.copy()   

    def baseline_step(self, controller, x_goal):
        l_next, dist, dl = controller.step(self, x_goal)
        self.set_lengths(l_next)   
        return l_next, dist, dl

    def get_ee_pos(self, id = None)-> np.ndarray:
        return sensors.site_pos(self.data, self.tip_sid).astype(np.float32)
    
    def get_lengths(self) -> np.ndarray:
        return self.data.ctrl[:self.nu].astype(np.float32)

    def set_lengths(self, l: np.ndarray):
        l = np.asarray(l, dtype=np.float32)
        l = np.clip(l, self.L_min, self.L_max)
        self._set_ctrl(l)

    def set_length(self, ctrl_vec: np.ndarray):
        self.set_lengths(ctrl_vec)
    
    def get_l_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.L_min, self.L_max
    
    def close(self):
        try:
            if self._renderer is not None:
                self._renderer.close()
        except Exception:
            pass
        self._renderer = None



if __name__ == "__main__":
    env = FlexPALSB3Env(
        xml_path="/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml",
        action_mode="absolute",
        render_mode="human",           # 调试 human；训练 None 或 rgb_array
        render_camera=None,
        K_substeps=10,
        max_episode_steps=1000,
    )
    obs, info = env.reset()
    a = 0
    action = np.array([-1, 1, -1, 1, -1, 1, -0.95, 1, -1], dtype=np.float32)[:env.nu]
    for i in range(100):
        obs, rew, term, trunc, info = env.step(action)
        a += rew
        if term or trunc:
            break
    print("done:", term, "truncated:", trunc, "reward:", rew, "iter:", i ,"rew_sum=", a)
