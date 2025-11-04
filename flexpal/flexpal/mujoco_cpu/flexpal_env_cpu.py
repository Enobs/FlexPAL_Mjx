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
        ang_err = 0
    else:
        ang_err = quat_geodesic_angle_np(ee_quat, goal[3:])
        base_r  = -(w_pos * pos_err + w_ori * ang_err)

    reward = float(np.clip(base_r , -clip_range, clip_range))
    return reward, pos_err, ang_err, 


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
        max_episode_steps: int = 200,
        render_mode: Optional[str] = None,
        render_width: int = 960,
        render_height: int = 720,
        render_camera: Optional[str] = None,
        step_penalty: float = 0.01,   
        use_time_scale: bool = False, 
        pos_only_ctrl = True,
        goal_library_path: Optional[str] = None,
        goal_voxel: Optional[float] = 0.002,
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
        
        self._prev_pos_err = None
        self._prog_hist = []
        self.stall_win = 15       # 连续多少步观察进步
        self.stall_eps = 1e-4     # 平均进步阈值（米）
        self.sat_frac_th = 0.3    # 动作饱和占比阈值
        self.sat_eps = 0.98       # |a|>0.98 视为饱和
        self.beta_prog = 0.3      # 进步奖励系数（0.2~0.6 可调）

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
                # 用库决定 workspace（轻微外扩便于可视化/渲染）
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
    
    def estimate_jacobian_fd(self, eps: float = 1e-3, K: int = 10) -> np.ndarray:

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
        self._prev_pos_err = None
        self._prog_hist.clear()
        self._set_ctrl(np.ones(self.nu, dtype=np.float32) * 0.29)
        self.l0 = self.data.ctrl[:self.nu].copy()
        self._step_physics()
        if self._goal_lib is not None:
            idx = np.random.randint(0, self._goal_lib.shape[0])
            goal_pos = self._goal_lib[idx]
            if self.goal_dim == 3:
                self.set_goal(goal_pos.astype(np.float32))  # (3,)
            else:
                # 如需姿态目标，你可以拼接一个默认四元数或从数据里取；这里只给位置
                g7 = np.empty(7, dtype=np.float32); g7[:3] = goal_pos; g7[3:] = np.array([1,0,0,0], np.float32)
                self.set_goal(g7)
        
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

        r_shape, pos_err, ang_err = reaching_reward_np(
            self.data, self.goal, self.goal_dim, self.tip_sid,
            w_pos=self.w_pos, w_ori=self.w_ori)

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
            "time_limit": truncated,
            "success": bool(terminated and not truncated),
        }

        if self._renderer is not None:
            self._renderer.update(self.model, self.data)
            if self.render_mode == "human":
                self._renderer.render()
            elif self.render_mode == "rgb_array":
                self._last_rgb = self._renderer.render()

        return obs, reward, terminated, truncated, info




        # --------- 进步塑形 (policy-invariant) ----------
        if not hasattr(self, "_prev_pos_err") or self._prev_pos_err is None:
            self._prev_pos_err = float(pos_err)

        if not hasattr(self, "beta_prog"):
            self.beta_prog = 0.3     # 可在 __init__ 固定配置
        if not hasattr(self, "_prog_hist"):
            self._prog_hist = []
            self.stall_win    = 15   # 连续观察步数
            self.stall_eps    = 1e-4 # 平均进步阈值(米)
            self.sat_frac_th  = 0.30 # 动作饱和占比阈值
            self.sat_eps      = 0.98 # |a|>=sat_eps 视为饱和

        delta = float(self._prev_pos_err - pos_err)  # >0 表示更接近
        r_prog = self.beta_prog * max(min(delta, float(self.tol_pos)), -float(self.tol_pos))
        # 近端加权：越靠近目标，进步奖励略放大
        denom = max(1e-8, 2.0 * float(self.tol_pos))
        near  = max(0.0, 1.0 - float(pos_en) / denom) if denom > 0 else 0.0
        r_prog *= (1.0 + 0.5 * near)
        reward += r_prog
        self._prev_pos_err = float(pos_err)

        # --------- 成功检测（保持你的 ok/hold_steps 逻辑） ----------
        if self.goal_dim == 3:
            ok = (pos_err < self.tol_pos)
        else:
            ok = (pos_err < self.tol_pos) and (ang_err < self.tol_ang)

        self._success_streak = self._success_streak + 1 if ok else 0
        terminated = bool(self._success_streak >= self.hold_steps)
        if terminated:
            reward += self.r_bonus

        # --------- 早停：无进步 + 动作长期饱和 ----------
        # 把最近的进步记录进窗口
        self._prog_hist.append(delta)
        if len(self._prog_hist) > self.stall_win:
            self._prog_hist.pop(0)
        sat_frac = float(np.mean(np.abs(a) >= self.sat_eps))

        if (not terminated) and len(self._prog_hist) == self.stall_win:
            avg_prog = float(np.mean(self._prog_hist))
            if (avg_prog < self.stall_eps) and (sat_frac > self.sat_frac_th):
                # 轻微负奖，避免长时间原地推不动
                reward -= 0.2
                terminated = True

        truncated = bool(self._timestep >= self.max_episode_steps)

        obs = self._obs_from_data()
        info = {
            "pos_err": float(pos_err),
            "ang_err": float(ang_err),
            "time_limit": truncated,
            "success": bool(terminated and not truncated),
            "r_prog": float(r_prog),
        }

        if self._renderer is not None:
            self._renderer.update(self.model, self.data)
            if self.render_mode == "human":
                self._renderer.render()
            elif self._render_w is not None:
                self._last_rgb = self._renderer.render()

        return obs, float(reward), bool(terminated), bool(truncated), info


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
        return sensors.tendon_state(self.data,self.ids_ten)

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
    action = np.array([-1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)[:env.nu]
    for i in range(100):
        obs, rew, term, trunc, info = env.step(action)
        a += rew
        if term or trunc:
            break
    print("done:", term, "truncated:", trunc, "reward:", rew, "iter:", i ,"rew_sum=", a)
