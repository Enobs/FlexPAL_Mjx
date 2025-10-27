# flex_env.py
import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as brax_mjcf
import flexpal.mujoco.idbuild as idbuild
import flexpal.mujoco.core as core
from typing import Tuple, Any
from flexpal.mujoco.control import SensorPIDParams  
from flexpal.mujoco import sensors, control
from functools import partial



@jax.jit
def quat_geodesic_angle(q_current: jnp.ndarray, q_goal: jnp.ndarray, eps=1e-8) -> jnp.ndarray:
    q1 = q_current / (jnp.linalg.norm(q_current) + eps)
    q2 = q_goal    / (jnp.linalg.norm(q_goal)    + eps)
    c = jnp.clip(jnp.abs(jnp.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * jnp.arccos(c)



@jax.jit
def compute_reward(s: core.CoreState,
                   goal_pos: jnp.ndarray,      # shape (3,)
                   goal_quat: jnp.ndarray,     # shape (4,), (w,x,y,z)
                   tip_site_id: int,
                   w_pos: float = 1.0,
                   w_ori: float = 0.2) -> jnp.ndarray:
    ee_pos  = sensors.site_pos(s, tip_site_id)                 # (3,)
    ee_quat = sensors.site_quat_world(s, tip_site_id)          # (4,)
    ee_quat = ee_quat/ (jnp.linalg.norm(ee_quat) + 1e-8)
    pos_err = jnp.linalg.norm(ee_pos - goal_pos)               # 米
    ang_err = quat_geodesic_angle(ee_quat, goal_quat)          # 弧度

    reward = -(w_pos * pos_err + w_ori * ang_err)
    return reward

class FlexPALEnv(PipelineEnv):

    def __init__(self,
                 xml_path: str,
                 control_freq: float = 25,
                 kp: float = 2.0,
                 ki: float = 0.3,
                 tol: float = 1e-3,
                 max_inner_steps: int = 400,
                 ):

        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        # mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        # mj_model.opt.iterations = 6
        # mj_model.opt.ls_iterations = 6

        try:
            sys = brax_mjcf.load_model(mj_model)
        except Exception:
            sys = brax_mjcf.load(xml_path)
        self.actuator_index =idbuild.gen_actuator_names()
        self.site_index=idbuild.gen_site_names()
        self.tendon_index = idbuild.gen_tendon_names()
        self.p = core.core_build_params(
          mj_model, control_freq=control_freq, 
          sites=self.site_index, 
          tendons=self.tendon_index,
          actuators=self.actuator_index)
        self.ctrl_param = core.core_build_pid_param()
        model_dt = float(mj_model.opt.timestep)
        n_frames = int(round((1.0 / control_freq) / model_dt))
        
        super().__init__(sys, backend="mjx", n_frames=n_frames)
        self.continuous_mode = False 

        self.kp = float(kp)
        self.ki = float(ki)
        self.tol = float(tol)
        self.max_inner = int(max_inner_steps)
        self.nt = len(self.p.ids.tendon)
        self.goal = jnp.zeros((7,), dtype= jnp.float32)
        self.dL_max = 0.005  
        self.L_min  = jnp.ones((self.nt,), jnp.float32)*0.24      
        self.L_max  = jnp.ones((self.nt,),  jnp.float32)*0.33  
        g = jnp.array([-0.08655875,0.02789154,0.8624209,0.53883886,-0.4978265,0.40249392,0.54755914], dtype=jnp.float32) 
        self.goal = g.at[3:].set(g[3:] / (jnp.linalg.norm(g[3:]) + 1e-8))
        
    @property
    def action_size(self) -> int:
        return int(self.nt)

    def _action_to_target_absolute(self, action: jnp.ndarray):
        # action ∈ [-1, 1]
        a01 = 0.5 * (action + 1.0)             # [-1,1] → [0,1]
        L_target = self.L_min + a01 * (self.L_max - self.L_min)
        return L_target
        
    def _action_to_tendon_target(self, s: core.CoreState, agent_action: jnp.ndarray):
        # agent_action in [-1, 1]^nt
        L  = sensors.tendon_state(s, self.p.ids.tendon)                 
        dL = jnp.clip(agent_action, -1.0, 1.0) * self.dL_max            
        L_target = jnp.clip(L + dL, self.L_min, self.L_max)             
        return L_target

    def reset(self, rng):
        data = self.pipeline_init(self.sys.qpos0, jnp.zeros(self.sys.nv))
        s0 = core.CoreState(data=data, t=jnp.array(0, jnp.int32))
        # sensor pos[-0.08655875  0.02789154  0.8624209 ]
        # sensor quat[ 0.53883886 -0.4978265   0.40249392  0.54755914]

        obs = self._get_obs(s0)
        zero = jnp.array(0., jnp.float32)
        metrics = dict(
            inner_steps=jnp.array(0, jnp.int32)
        )
        return State(data, obs, zero, zero, metrics)
    
    @partial(jax.jit, static_argnums=(0,))  
    def _step_controller_brax(self,
                              p: core.CoreParams,
                              s: core.CoreState,
                              ss_g: jax.Array,
                              pid_param: SensorPIDParams,
                              ctrl_param: core.PIDPiecewise
                              ) -> Tuple[core.CoreState, jax.Array, SensorPIDParams]:
        tendon_full = sensors.tendon_state(s, p.ids.tendon)

        delta_u, new_integral = control.v_pid_sensor(ss_g, tendon_full, pid_param, p.ctrl_dt)
        u_raw  = s.data.ctrl + delta_u
        u_ctrl = jnp.clip(u_raw, -1.0, 1.0)

        s_next, _ = core.inner_step(p, s, u_ctrl, ctrl_param)
        dT = self.pipeline_step(s.data, s_next.data.ctrl)   # <-- 直接用 self.pipeline_step
        s_current = s.replace(data=dT, t=s.t + 1)

        sd = s_current.data.sensordata
        n_total = sd.shape[0]
        n_imu = n_total // 6

        def has_imu_true(_):
            gyro_vals = sd.reshape((n_imu, 6))[:, 0:3]
            gyro_norm = jnp.linalg.norm(gyro_vals, axis=1)
            return jnp.all(gyro_norm < 0.05)

        def has_imu_false(_):
            return jnp.array(False, dtype=jnp.bool_)

        gyro_stable = jax.lax.cond(n_imu > 0, has_imu_true, has_imu_false, operand=None)

        tendon_after = sensors.tendon_state(s_current, p.ids.tendon)
        pose_reach = jnp.all(jnp.abs(tendon_after - ss_g) < pid_param.tol)

        done = jnp.logical_or(gyro_stable, pose_reach)

        new_pid = pid_param.replace(integral=new_integral)
        return s_current, done, new_pid

    @partial(jax.jit, static_argnums=(0,))  # self 静态；max_steps 作为张量/常量也行
    def _loop_until_reach_brax(self, p, s, ss_g, pid_param, ctrl_param, max_steps: int):
        def cond_fun(loop_state):
            _s, done, k, _pid = loop_state
            return (~done) & (k < max_steps)

        def body_fun(loop_state):
            s, _done, k, pidp = loop_state
            s_next, done, pidp_next = self._step_controller_brax(p, s, ss_g, pidp, ctrl_param)
            return (s_next, done, k + 1, pidp_next)

        init = (s, jnp.array(False), jnp.array(0, jnp.int32), pid_param)
        s_f, done_f, steps, pidp_f = jax.lax.while_loop(cond_fun, body_fun, init)
        return s_f, steps, pidp_f


    @partial(jax.jit, static_argnums=0)
    def _jit_step(self, state, action):
        pid_integral = jnp.zeros((len(self.p.ids.tendon),), dtype=jnp.float32)
        pid = SensorPIDParams(kp=self.kp, ki=self.ki, tol=self.tol,
                              integral=pid_integral)
        
        s0 = core.CoreState(data=state.pipeline_state, t=jnp.array(0, jnp.int32))
        tendon_target = jax.lax.cond(
            self.continuous_mode,                                      
            lambda oa: self._action_to_tendon_target(oa[0], oa[1]),    
            lambda oa: self._action_to_target_absolute(oa[1]),         
            operand=(s0, action),                                      
        )
        # tendon_target = action
        s_f, inner_steps, pid_f = self._loop_until_reach_brax(
            self.p, s0, tendon_target, pid, self.ctrl_param, self.max_inner
        )


        data = s_f.data
        obs = self._get_obs(s_f)
        tip_site_id = self.p.ids.site[-1]  # 或 self.tip_site_id
        goal_pos, goal_quat = self.goal[:3], self.goal[3:]
        reward = compute_reward(s_f, goal_pos, goal_quat, tip_site_id)
        state.metrics.update(inner_steps=inner_steps)
        done = jnp.array(1., jnp.float32) if not self.continuous_mode else jnp.array(0., jnp.float32)
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)
    
    def step(self, state, action):
        return self._jit_step(state, action)
    
    def _get_obs(self, s_f: core.CoreState) -> jax.Array:
        tendon = sensors.tendon_state(s_f, self.p.ids.tendon)
        imu = s_f.data.sensordata
        imu = jnp.where(imu.size > 0, imu, jnp.zeros((self.nt * 6,), jnp.float32))
        return jnp.concatenate([tendon, imu, self.goal])

    









if __name__ == "__main__":
    import time
    import numpy as onp
    from jax import random, device_get

    xml_path = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml"

    control_freq = 25.0
    kp, ki = 2.0, 0.3
    tol = 1e-3
    max_inner_steps = 400

    env = FlexPALEnv(
        xml_path=xml_path,
        control_freq=control_freq,
        kp=kp,
        ki=ki,
        tol=tol,
        max_inner_steps=max_inner_steps,
    )


    key = random.PRNGKey(0)
    state = env.reset(key)

    t0 = time.time()


    action = jnp.array([0.30562606, 0.28558427, 0.28487587, 0.20157896, 0.28575578, 0.21900828, 0.14331605, 0.30143574, 0.33560848], dtype=jnp.float32)

    state = env.step(state, action)

    rew = device_get(state.reward)
    done = device_get(state.done)
    inner_steps = device_get(state.metrics['inner_steps'])

    print(f"reward={float(rew):.4f}  inner_steps={int(inner_steps)}  done={float(done):.0f}  ")


