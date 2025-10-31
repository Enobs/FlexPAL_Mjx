# flex_env.py
import jax
import jax.numpy as jnp
import mujoco
from typing import Tuple, Any
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as brax_mjcf

import flexpal.mujoco.idbuild as idbuild
import flexpal.mujoco.core as core
from flexpal.mujoco import sensors
from flax.core import FrozenDict


def quat_geodesic_angle(q_current: jnp.ndarray, q_goal: jnp.ndarray, eps=1e-8) -> jnp.ndarray:
    q1 = q_current / (jnp.linalg.norm(q_current) + eps)
    q2 = q_goal    / (jnp.linalg.norm(q_goal)    + eps)
    c = jnp.clip(jnp.abs(jnp.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * jnp.arccos(c)

import jax
import jax.numpy as jnp
import mujoco

def reaching_reward(
    data: mujoco.mjx.Data,
    goal_pos: jnp.ndarray,        # (3,)
    goal_quat: jnp.ndarray,       # (4,) unit
    tip_site_id: int,             # 建议用 int
    w_pos: float = 1.0,
    w_ori: float = 0.2,
    *,
    prev_data: mujoco.mjx.Data | None = None,   # 上一帧 data（可选）
    tol_pos: float = 5e-3,
    tol_ang: float = 5e-2,
    w_improve: float = 0.5,       # 进步项权重
    w_soft: float = 0.5,          # 软成功权重
    ang_improve_scale: float = 0.3,  # 角度进步的相对权重
    clip_range: float = 50.0
):
    # 当前误差
    ee_pos  = sensors.site_pos(data, tip_site_id)        # (3,)
    ee_quat = sensors.site_quat_world(data, tip_site_id) # (4,)
    ee_quat = ee_quat / (jnp.linalg.norm(ee_quat) + 1e-8)

    pos_err = jnp.linalg.norm(ee_pos - goal_pos)

    # 四元数夹角（单位化后用 |dot| 避免双解）
    c = jnp.clip(jnp.abs(jnp.dot(ee_quat, goal_quat)), 0.0, 1.0)
    ang_err = 2.0 * jnp.arccos(c)

    # 基础密集奖励：负误差
    base_r = -(w_pos * pos_err + w_ori * ang_err)

    # 进步项（如果提供了上一帧 prev_data）
    def _err_from(d):
        p  = sensors.site_pos(d, tip_site_id)
        q  = sensors.site_quat_world(d, tip_site_id)
        q  = q / (jnp.linalg.norm(q) + 1e-8)
        pe = jnp.linalg.norm(p - goal_pos)
        ce = jnp.clip(jnp.abs(jnp.dot(q, goal_quat)), 0.0, 1.0)
        ae = 2.0 * jnp.arccos(ce)
        return pe, ae

    if prev_data is not None:
        pos_err0, ang_err0 = _err_from(prev_data)
        improve = (pos_err0 - pos_err) + ang_improve_scale * (ang_err0 - ang_err)
    else:
        improve = 0.0

    # 软成功：把硬阈值变为连续增益（0~1）
    def smooth_success(e, tol):
        x = (tol - e) / (tol + 1e-6)   # >0 表示进入阈值
        return jax.nn.sigmoid(6.0 * x) # 6.0 可调，越大越“硬”

    soft_succ = 0.7 * smooth_success(pos_err, tol_pos) + 0.3 * smooth_success(ang_err, tol_ang)

    # 总奖励
    reward = base_r + w_improve * improve + w_soft * soft_succ
    reward = jnp.clip(reward, -clip_range, clip_range)

    return reward, pos_err, ang_err, soft_succ



class FlexPALEnv(PipelineEnv):
    """
    - 每个 env.step() 推进一个控制周期（p.substeps 个物理步）。
    - 动作 -> (相对/绝对) 腱长目标 -> 一阶低通 -> 写入 actuator 槽位 -> 推进一步。
    - 误差低于阈值连续 hold_steps 步则成功（+bonus 且 done=True）。
    - 不在类内使用 @jax.jit；让 Brax 训练器在外层 wrap+jit。
    """

    def __init__(self,
                 xml_path: str = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml",
                 control_freq: float = 250.0,
                 tol_pos: float = 5e-3,          
                 tol_ang: float = 5e-2,          
                 hold_steps: int = 3,            
                 alpha_smooth: float = 0,      
                 action_mode: str = "absolute",  
                 dL_max: float = 0.06,          
                 w_pos: float = 1.0,
                 w_ori: float = 0.2,
                 w_du: float  = 1e-2,            
                 r_bonus: float = 1.0,           
                 ):
        # 1) 载入模型（MJCF -> Sys），并用 MJX 后端
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        try:
            sys = brax_mjcf.load_model(mj_model)
        except Exception:
            sys = brax_mjcf.load(xml_path)
        
        self.actuator_index = idbuild.gen_actuator_names()
        self.site_index     = idbuild.gen_site_names()
        self.tendon_index   = idbuild.gen_tendon_names()

        self.p = core.core_build_params(
            mj_model, control_freq=control_freq,
            sites=self.site_index, tendons=self.tendon_index,
            actuators=self.actuator_index)

        ctrl_rng = mj_model.actuator_ctrlrange  
        self.ctrl_min = jnp.asarray(ctrl_rng[:, 0], dtype=jnp.float32)
        self.ctrl_max = jnp.asarray(ctrl_rng[:, 1], dtype=jnp.float32)

        self.L_min = self.ctrl_min
        self.L_max = self.ctrl_max

        model_dt = float(mj_model.opt.timestep)
        n_frames = int(round((1.0 / control_freq) / model_dt))
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self.nt = len(self.p.ids.tendon)
        self.dL_max = float(dL_max)
        self.alpha  = float(alpha_smooth)

        self.tol_pos = float(tol_pos)
        self.tol_ang = float(tol_ang)
        self.hold_steps = int(hold_steps)

        self.action_mode = str(action_mode).lower()
        assert self.action_mode in ("relative", "absolute")

        g = jnp.array([-0.187, 0.142, 0.835,
                        0.671, 0.234, 0.673, 0.201], dtype=jnp.float32)
        self.goal = g.at[3:].set(g[3:] / (jnp.linalg.norm(g[3:]) + 1e-8))


        self.w_pos = float(w_pos)
        self.w_ori = float(w_ori)
        self.w_du  = float(w_du)
        self.r_bonus = float(r_bonus)
        self._act_id = jnp.asarray(self.p.ids.actuator, dtype=jnp.int32)
        self._ten_id = jnp.asarray(self.p.ids.tendon,   dtype=jnp.int32)
        self._tip_id = jnp.asarray([int(self.p.ids.site[-1])], dtype=jnp.int32)
        self.K_substeps = 10

    # ====== Brax Env API ======

    @property
    def action_size(self) -> int:
        return int(self.nt)

    def reset(self, rng):
        
        data = self.pipeline_init(self.sys.qpos0, jnp.zeros(self.sys.nv))
        init_ctrl = jnp.ones((self.nt,), jnp.float32) * 0.29
        data = data.replace(ctrl=data.ctrl.at[self._act_id].set(init_ctrl))

        obs = self._get_obs_from_data(data)

        zero = jnp.array(0., jnp.float32)
        metrics = dict(
            inner_steps=zero,
            success_count=zero
        )
        return State(data, obs, zero, zero, metrics)
    
    def _k_substeps(self, data, ctrl):
        def body(_, d):
            return self.pipeline_step(d, ctrl)
        return jax.lax.fori_loop(0, self.K_substeps, body, data)

    def step(self, state: State, action: jnp.ndarray) -> State:
        a  = jnp.clip(action, -1.0, 1.0)

        # 上一时刻腱长（或 ctrl，同一物理量）
        L_prev = state.pipeline_state.ctrl[self._act_id]

        span   = self.L_max - self.L_min                # shape (nu,)
        dL_cap = self.dL_max * span                     # shape (nu,)

        # absolute：映射到各自 [L_min, L_max]，再做“限步逼近”
        def abs_branch(_):
            L_abs = self.L_min + 0.5 * (a + 1.0) * span
            room_up   = self.L_max - L_prev
            room_down = L_prev   - self.L_min
            step_pos  = jnp.minimum(jnp.maximum(L_abs - L_prev, 0.0), 0.5 * room_up)
            step_neg  = -jnp.minimum(jnp.maximum(L_prev - L_abs, 0.0), 0.5 * room_down)
            dL        = jnp.clip(step_pos + step_neg, -dL_cap, dL_cap)
            return jnp.clip(L_prev + dL, self.L_min, self.L_max)

        # relative：按每路步幅增量
        def rel_branch(_):
            dL = a * dL_cap
            room_up   = self.L_max - L_prev
            room_down = L_prev   - self.L_min
            dL_pos    = jnp.minimum(jnp.maximum(dL, 0.0), 0.5 * room_up)
            dL_neg    = -jnp.minimum(jnp.maximum(-dL, 0.0), 0.5 * room_down)
            return jnp.clip(L_prev + dL_pos + dL_neg, self.L_min, self.L_max)

        L_target = jax.lax.cond(self.action_mode == "absolute", abs_branch, rel_branch, operand=None)


        u_new  = self.alpha * L_prev + (1.0 - self.alpha) * L_target

        # data1 = self._k_substeps(state.pipeline_state, u_new)
        data1 = self.pipeline_step(state.pipeline_state, u_new)
        

        shaped_r, pos_err, ang_err, soft_succ  = reaching_reward(
            data1, self.goal[:3], self.goal[3:], self._tip_id,
            w_pos=self.w_pos, w_ori=self.w_ori,
            prev_data=state.pipeline_state,       # 有上一帧就传
            tol_pos=self.tol_pos, tol_ang=self.tol_ang,
            w_improve=0.5, w_soft=0.5, ang_improve_scale=0.3
        )

        du_pen = jnp.mean((u_new - L_prev) ** 2)
        reward  = shaped_r - self.w_du * du_pen



        ok = jnp.logical_and(pos_err < self.tol_pos, ang_err < self.tol_ang)
        success_count = jnp.where(ok, state.metrics["success_count"] + 1.0, jnp.array(0.0, jnp.float32))
        done = jnp.array(success_count >= float(self.hold_steps), jnp.float32)

        reward = jnp.where(done > 0, reward + self.r_bonus, reward)

        obs = self._get_obs_from_data(data1)
        metrics = dict(state.metrics)
        metrics["success_count"] = success_count
        metrics["inner_steps"] = jnp.array(1.0, jnp.float32)  
        return state.replace(pipeline_state=data1, obs=obs, reward=reward, done=done, metrics=metrics)

    def _get_obs_from_data(self, data: mujoco.mjx.Data) -> jnp.ndarray:
        # 一次性 tendon 状态（传入已经是 device 上的 _ten_id）
        tendon = sensors.tendon_state(data, self._ten_id)
        imu    = data.sensordata
        imu    = jnp.where(imu.size > 0, imu, jnp.zeros((self.nt * 6,), jnp.float32))
        return jnp.concatenate([tendon, imu, self.goal])



if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    env= FlexPALEnv(action_mode="absolute")
    # env= FlexPALEnv()
    key   = jax.random.PRNGKey(0)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    state = jit_reset(key)
    warm_step = jit_step(state, (jnp.ones(9,)*0.29))
    action = jnp.array(([-1,1,-1,1,-1,1,-1,1,-1]),dtype=jnp.float32)
    t1 = time.perf_counter()
    duration = t1 - t0
    print(f"Total time taken: {duration:.4f} seconds")
    t0 = time.perf_counter()
    for i in range(150):
        state = jit_step(state , action)
        print(f"current {i}-step reward rew={state.reward}")
    p = env.p
    t1 = time.perf_counter()
    duration = t1 - t0
    print(f"Total time taken: {duration:.4f} seconds")
    print(f"\n--- Final Performance Report ---")
    print(f"current tendon length: {sensors.tendon_state(state.pipeline_state, p.ids.tendon)}")
    print(f"current actuator ctrl: {state.pipeline_state.ctrl}")
    print(f"tip site pos : {sensors.site_pos(state.pipeline_state, p.ids.site[-1])}")
    print(f"tip site quat: {sensors.site_quat_world(state.pipeline_state, p.ids.site[-1])}")
    print(f"current sensordata len={len(state.pipeline_state.sensordata)}")
    print(f"current reward rew={state.reward}")
    
