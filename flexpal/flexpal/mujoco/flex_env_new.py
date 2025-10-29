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



def quat_geodesic_angle(q_current: jnp.ndarray, q_goal: jnp.ndarray, eps=1e-8) -> jnp.ndarray:
    q1 = q_current / (jnp.linalg.norm(q_current) + eps)
    q2 = q_goal    / (jnp.linalg.norm(q_goal)    + eps)
    c = jnp.clip(jnp.abs(jnp.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * jnp.arccos(c)

def reaching_reward(s: "core.CoreState",
                    goal_pos: jnp.ndarray,     # (3,)
                    goal_quat: jnp.ndarray,    # (4,) (w, x, y, z)
                    tip_site_id: int,
                    w_pos: float,
                    w_ori: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """返回 (reward, pos_err, ang_err)。奖励是负误差（还未加成功奖励/控制惩罚）。"""
    ee_pos  = sensors.site_pos(s, tip_site_id)          # (3,)
    ee_quat = sensors.site_quat_world(s, tip_site_id)   # (4,)
    ee_quat = ee_quat / (jnp.linalg.norm(ee_quat) + 1e-8)
    pos_err = jnp.linalg.norm(ee_pos - goal_pos)
    ang_err = quat_geodesic_angle(ee_quat, goal_quat)
    reward  = -(w_pos * pos_err + w_ori * ang_err)
    return reward, pos_err, ang_err


class FlexPALEnv(PipelineEnv):
    """
    - 每个 env.step() 推进一个控制周期（p.substeps 个物理步）。
    - 动作 -> (相对/绝对) 腱长目标 -> 一阶低通 -> 写入 actuator 槽位 -> 推进一步。
    - 误差低于阈值连续 hold_steps 步则成功（+bonus 且 done=True）。
    - 不在类内使用 @jax.jit；让 Brax 训练器在外层 wrap+jit。
    """

    def __init__(self,
                 xml_path: str = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml",
                 control_freq: float = 20.0,
                 tol_pos: float = 5e-3,          
                 tol_ang: float = 5e-2,          
                 hold_steps: int = 3,            
                 alpha_smooth: float = 0,      
                 action_mode: str = "absolute",  
                 dL_max: float = 0.005,          
                 L_min_val: float = 0.15,
                 L_max_val: float = 0.33,
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

        # 2) 建 ID、MJX Params
        self.actuator_index = idbuild.gen_actuator_names()
        self.site_index     = idbuild.gen_site_names()
        self.tendon_index   = idbuild.gen_tendon_names()

        self.p = core.core_build_params(
            mj_model, control_freq=control_freq,
            sites=self.site_index, tendons=self.tendon_index,
            actuators=self.actuator_index)

        model_dt = float(mj_model.opt.timestep)
        n_frames = int(round((1.0 / control_freq) / model_dt))
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self.nt = len(self.p.ids.tendon)
        self.L_min  = jnp.ones((self.nt,), jnp.float32) * L_min_val
        self.L_max  = jnp.ones((self.nt,), jnp.float32) * L_max_val
        self.dL_max = float(dL_max)
        self.alpha  = float(alpha_smooth)

        self.tol_pos = float(tol_pos)
        self.tol_ang = float(tol_ang)
        self.hold_steps = int(hold_steps)

        self.action_mode = str(action_mode).lower()
        assert self.action_mode in ("relative", "absolute")

        g = jnp.array([-0.41326547,0.01179456,0.75909054,
                        0.9773609 ,0.12534763,-0.03064912, 0.1676719], dtype=jnp.float32)
        self.goal = g.at[3:].set(g[3:] / (jnp.linalg.norm(g[3:]) + 1e-8))

        self.tip_site_id = int(self.p.ids.site[-1])

        self.w_pos = float(w_pos)
        self.w_ori = float(w_ori)
        self.w_du  = float(w_du)
        self.r_bonus = float(r_bonus)

    # ====== Brax Env API ======

    @property
    def action_size(self) -> int:
        return int(self.nt)

    def reset(self, rng):
        
        data = self.pipeline_init(self.sys.qpos0, jnp.zeros(self.sys.nv))

        init_ctrl = jnp.ones((self.nt,), jnp.float32) * 0.29
        data = data.replace(ctrl=data.ctrl.at[self.p.ids.actuator].set(init_ctrl))

        s0 = core.CoreState(data=data, t=jnp.array(0, jnp.int32))
        obs = self._get_obs(s0)

        zero = jnp.array(0., jnp.float32)
        metrics = dict(
            inner_steps=zero,
            success_count=zero
        )
        return State(data, obs, zero, zero, metrics)

    def step(self, state: State, action: jnp.ndarray) -> State:
        """每个环境步：动作->目标->低通->推进一步->奖励/终止。"""
        s0 = core.CoreState(data=state.pipeline_state, t=jnp.array(0, jnp.int32))
        u_prev = s0.data.ctrl[self.p.ids.actuator]
        target =jax.lax.cond(
            self.action_mode == "relative",
            lambda _ : jnp.clip(sensors.tendon_state(s0, self.p.ids.tendon) + jnp.clip(action, -1.0, 1.0) * self.dL_max, self.L_min, self.L_max),
            lambda _ : self.L_min + 0.5 * (jnp.clip(action, -1.0, 1.0) + 1.0) * (self.L_max - self.L_min),
            operand = None)

        u_new  = self.alpha * u_prev + (1.0 - self.alpha) * target

        data1 = self.pipeline_step(s0.data, u_new)
        s1 = core.CoreState(data=data1, t=s0.t + 1)

        base_r, pos_err, ang_err = reaching_reward(
            s1, self.goal[:3], self.goal[3:], self.tip_site_id, self.w_pos, self.w_ori
        )
        du_pen = jnp.mean((u_new - u_prev) ** 2)
        reward = base_r - self.w_du * du_pen


        ok = jnp.logical_and(pos_err < self.tol_pos, ang_err < self.tol_ang)
        success_count = jnp.where(ok, state.metrics["success_count"] + 1.0, jnp.array(0.0, jnp.float32))
        done = jnp.array(success_count >= float(self.hold_steps), jnp.float32)

        reward = jnp.where(done > 0, reward + self.r_bonus, reward)

        obs = self._get_obs(s1)
        metrics = dict(state.metrics)
        metrics["success_count"] = success_count
        metrics["inner_steps"] = jnp.array(1.0, jnp.float32)  

        return state.replace(pipeline_state=s1.data, obs=obs, reward=reward, done=done, metrics=metrics)

    def _get_obs(self, s: "core.CoreState") -> jnp.ndarray:
        tendon = sensors.tendon_state(s, self.p.ids.tendon)  
        imu = s.data.sensordata
        imu = jnp.where(imu.size > 0, imu, jnp.zeros((self.nt * 6,), jnp.float32))
        return jnp.concatenate([tendon, imu, self.goal])



if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    env= FlexPALEnv(action_mode="absolute")
    key   = jax.random.PRNGKey(0)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    state = jit_reset(key)
    warm_step = jit_step(state, (jnp.ones(9,)*0.29))
    action = jnp.array(([-0.49,0.1,-0.5,0.0,-0.5,-0.5,-0.5,-0.5,0.0]),dtype=jnp.float32)
    t1 = time.perf_counter()
    duration = t1 - t0
    print(f"Total time taken: {duration:.4f} seconds")
    t0 = time.perf_counter()
    for _ in range(40):
        state = jit_step(state , action)
    s_current = core.CoreState(data=state.pipeline_state, t=jnp.array(0, jnp.int32))
    p = env.p
    t1 = time.perf_counter()
    duration = t1 - t0
    print(f"Total time taken: {duration:.4f} seconds")
    print(f"\n--- Final Performance Report ---")
    print(f"current tendon length: {sensors.tendon_state(s_current, p.ids.tendon)}")
    print(f"current actuator ctrl: {s_current.data.ctrl}")
    print(f"tip site pos : {sensors.site_pos(s_current, p.ids.site[-1])}")
    print(f"tip site quat: {sensors.site_quat_world(s_current, p.ids.site[-1])}")
    print(f"current sensordata len={len(s_current.data.sensordata)}")
    print(f"current reward rew={state.reward}")
    


# --- Final Performance Report ---
# current tendon length: [0.20139477 0.24779536 0.20266755 0.24372745 0.19784747 0.20033537
#  0.19892767 0.19889532 0.24373154]
# current actuator ctrl: [0.19590001 0.24900001 0.19500001 0.24000001 0.19500001 0.19500001
#  0.19500001 0.19500001 0.24000001]
# tip site pos : [-0.41337612  0.01178646  0.75916535]
# tip site quat: [ 0.97735983  0.12537758 -0.03053388  0.16767731]
# current sensordata len=54

# --- Final Performance Report ---
# current tendon length: [0.20139505 0.24779561 0.20266783 0.2437269  0.19784687 0.2003347
#  0.1989274  0.1988951  0.24373138]
# current actuator ctrl: [0.19590001 0.24900001 0.19500001 0.24000001 0.19500001 0.19500001
#  0.19500001 0.19500001 0.24000001]
# tip site pos : [-0.41337886  0.01178995  0.7591661 ]
# tip site quat: [ 0.9773599   0.12537761 -0.03053319  0.16767834]

# --- Final Performance Report ---
# current tendon length: [0.20139529 0.24779706 0.20266668 0.24372736 0.19784774 0.2003347
#  0.19892734 0.19889508 0.24373123]
# current actuator ctrl: [0.19590001 0.24900001 0.19500001 0.24000001 0.19500001 0.19500001
#  0.19500001 0.19500001 0.24000001]
# tip site pos : [-0.41341022  0.01177989  0.75918734]
# tip site quat: [ 0.97736     0.12538095 -0.03050144  0.16768108]