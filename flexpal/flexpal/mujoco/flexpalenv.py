# flex_env.py
import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as brax_mjcf
import idbuild
import core
from typing import Tuple, Any
from control import SensorPIDParams  
import sensors, control

@jax.jit
def step_controller_brax(p: core.CoreParams,
                         s: core.CoreState,
                         ss_g: jax.Array,
                         pid_param: SensorPIDParams,
                         ctrl_param: core.PIDPiecewise,
                         pipeline_step_fn: Any 
                        ) -> Tuple[core.CoreState, jax.Array, SensorPIDParams]:
    tendon_full = sensors.tendon_state(s, p.ids.tendon)
    reach_mask  = jnp.abs(tendon_full - ss_g) < pid_param.tol

    delta_u, new_integral = control.v_pid_sensor(ss_g, tendon_full, pid_param, p.ctrl_dt)
    u_raw  = s.data.ctrl + delta_u
    u_ctrl = jnp.clip(u_raw, -1.0, 1.0)
    
    s_next, _ = core.inner_step(p, s, u_ctrl, ctrl_param)
    dT = pipeline_step_fn(s.data, s_next.data.ctrl)
    s_current = s.replace(data=dT, t=s.t + 1)
    
    
    pose_reach = jnp.all(reach_mask).astype(jnp.int32)
    new_pid = pid_param.replace(integral=new_integral)
    return s_current, pose_reach, new_pid


@jax.jit
def loop_until_reach_brax(p, s, ss_g, pid_param, ctrl_param, pipeline_step_fn, max_steps=1000):
    def cond_fun(loop_state):
        _s, reach, k, _pid = loop_state
        return (reach == 0) & (k < max_steps)

    def body_fun(loop_state):
        s, _reach, k, pidp = loop_state
        s_next, reach, pidp_next = step_controller_brax(
            p, s, ss_g, pidp, ctrl_param, pipeline_step_fn
        )
        return (s_next, reach, k + 1, pidp_next)

    init = (s, jnp.array(0, jnp.int32), jnp.array(0, jnp.int32), pid_param)
    s_f, reach_f, steps, pidp_f = jax.lax.while_loop(cond_fun, body_fun, init)
    return s_f, steps, pidp_f

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
        self.continuous_mode = False  # 或 True

        self.kp = float(kp)
        self.ki = float(ki)
        self.tol = float(tol)
        self.max_inner = int(max_inner_steps)
        self.nt = len(self.p.ids.tendon)

    def reset(self, rng):
        data = self.pipeline_init(self.sys.qpos0, jnp.zeros(self.sys.nv))
        pid_integral = jnp.zeros((len(self.p.ids.tendon),), dtype=jnp.float32)
        obs = self._get_obs(data, jnp.zeros(self.sys.nu))
        zero = jnp.array(0., jnp.float32)
        metrics = dict(
            pid_integral=pid_integral,
            inner_steps=jnp.array(0, jnp.int32)
        )
        return State(data, obs, zero, zero, metrics)


    def step(self, state, tendon_target):
        pid_integral = state.metrics['pid_integral']
        pid = SensorPIDParams(kp=self.kp, ki=self.ki, tol=self.tol,
                              integral=pid_integral)

        s0 = core.CoreState(data=state.pipeline_state, t=jnp.array(0, jnp.int32))
        s_f, inner_steps, pid_f = loop_until_reach_brax(
            self.p, s0, tendon_target, pid, self.ctrl_param,
            pipeline_step_fn=self.pipeline_step,
            max_steps=self.max_inner,
        )

        data = s_f.data
        obs = self._get_obs(data, data.ctrl)
        
        state.metrics.update(pid_integral=pid_f.integral, inner_steps=inner_steps)
        
        done = jnp.array(1., jnp.float32) if not self.continuous_mode else jnp.array(0., jnp.float32)
        reward = jnp.array(0., jnp.float32)

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data, action: jnp.ndarray) -> jnp.ndarray:
        pos = data.qpos
        return jnp.concatenate([
            pos,
            data.qvel,
            data.cinert[1:].ravel(),
            data.cvel[1:].ravel(),
            data.qfrc_actuator,
        ])







if __name__== "__main__":
  pass