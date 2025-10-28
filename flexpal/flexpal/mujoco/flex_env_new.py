# flex_env.py  —— slimmer JIT, no static self, no pipeline_step inside jit
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

# ---------- small pure helpers ----------

@jax.jit
def quat_geodesic_angle(q_current: jnp.ndarray, q_goal: jnp.ndarray, eps=1e-8) -> jnp.ndarray:
    q1 = q_current / (jnp.linalg.norm(q_current) + eps)
    q2 = q_goal    / (jnp.linalg.norm(q_goal)    + eps)
    c = jnp.clip(jnp.abs(jnp.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * jnp.arccos(c)

@jax.jit
def compute_reward(s: core.CoreState,
                   goal_pos: jnp.ndarray,      # (3,)
                   goal_quat: jnp.ndarray,     # (4,) (w,x,y,z)
                   tip_site_id: int,
                   w_pos: float = 1.0,
                   w_ori: float = 0.2) -> jnp.ndarray:
    ee_pos  = sensors.site_pos(s, tip_site_id)
    ee_quat = sensors.site_quat_world(s, tip_site_id)
    ee_quat = ee_quat / (jnp.linalg.norm(ee_quat) + 1e-8)
    pos_err = jnp.linalg.norm(ee_pos - goal_pos)
    ang_err = quat_geodesic_angle(ee_quat, goal_quat)
    return -(w_pos * pos_err + w_ori * ang_err)

# ---------- PURE-JAX control loop (no self captured) ----------

@jax.jit
def _step_controller_jit(p: core.CoreParams,
                         s: core.CoreState,
                         ss_g: jax.Array,
                         pid_param: SensorPIDParams,
                         ctrl_param: core.PIDPiecewise) -> Tuple[core.CoreState, jax.Array, SensorPIDParams]:
    tendon_full = sensors.tendon_state(s, p.ids.tendon)
    delta_u, new_integral = control.v_pid_sensor(ss_g, tendon_full, pid_param, p.ctrl_dt)
    u_ctrl = jnp.clip(s.data.ctrl + delta_u, -1.0, 1.0)

    # inner PID step
    s_next, _ = core.inner_step(p, s, u_ctrl, ctrl_param)
    # integrate physics for one control period
    s_current = core.core_step(p, s_next, s_next.data.ctrl)

    tendon_after = sensors.tendon_state(s_current, p.ids.tendon)
    pose_reach = jnp.all(jnp.abs(tendon_after - ss_g) < pid_param.tol)
    new_pid = pid_param.replace(integral=new_integral)
    return s_current, pose_reach, new_pid

@jax.jit
def _loop_until_reach_jit(p, s, ss_g, pid_param, ctrl_param, max_steps: int):
    def cond_fun(loop_state):
        _s, done, k, _pid = loop_state
        return (~done) & (k < max_steps)

    def body_fun(loop_state):
        s, _done, k, pidp = loop_state
        s_next, done, pidp_next = _step_controller_jit(p, s, ss_g, pidp, ctrl_param)
        return (s_next, done, k + 1, pidp_next)

    init = (s, jnp.array(False), jnp.array(0, jnp.float32), pid_param)
    s_f, done_f, steps, pidp_f = jax.lax.while_loop(cond_fun, body_fun, init)
    return s_f, steps, pidp_f

@jax.jit
def _action_to_target_absolute(action: jnp.ndarray,
                               L_min: jnp.ndarray,
                               L_max: jnp.ndarray) -> jnp.ndarray:
    a01 = 0.5 * (jnp.clip(action, -1.0, 1.0) + 1.0)  # [-1,1] -> [0,1]
    return L_min + a01 * (L_max - L_min)

@jax.jit
def _action_to_tendon_target_rel(s: core.CoreState,
                                 action: jnp.ndarray,
                                 p: core.CoreParams,
                                 dL_max: float,
                                 L_min: jnp.ndarray,
                                 L_max: jnp.ndarray) -> jnp.ndarray:
    L  = sensors.tendon_state(s, p.ids.tendon)
    dL = jnp.clip(action, -1.0, 1.0) * dL_max
    return jnp.clip(L + dL, L_min, L_max)

@jax.jit
def _step_env_jit(state: State,
                  action: jnp.ndarray,
                  p: core.CoreParams,
                  ctrl_param: core.PIDPiecewise,
                  kp: float, ki: float, tol: float,
                  continuous_mode: bool,
                  goal: jnp.ndarray,
                  tip_site_id: int,
                  max_inner: int,
                  dL_max: float,
                  L_min: jnp.ndarray,
                  L_max: jnp.ndarray) -> State:
    # build PID param fresh each step (stateless env)
    pid_integral = jnp.zeros((len(p.ids.tendon),), dtype=jnp.float32)
    pid = SensorPIDParams(kp=kp, ki=ki, tol=tol, integral=pid_integral)

    s0 = core.CoreState(data=state.pipeline_state, t=jnp.array(0, jnp.int32))

    # tendon_target = jax.lax.cond(
    #     continuous_mode,
    #     lambda oa: _action_to_tendon_target_rel(oa[0], oa[1], p, dL_max, L_min, L_max),
    #     lambda oa: _action_to_target_absolute(oa[1], L_min, L_max),
    #     operand=(s0, action)
    # )
    tendon_target = action
    s_f, inner_steps, _ = _loop_until_reach_jit(p, s0, tendon_target, pid, ctrl_param, max_inner)

    # obs: tendon + imu(sensordata) + goal (shape must be static)
    tendon = sensors.tendon_state(s_f, p.ids.tendon)
    imu = s_f.data.sensordata  # length is fixed by model
    obs = jnp.concatenate([tendon, imu, goal])

    reward = compute_reward(s_f, goal[:3], goal[3:], tip_site_id)
    done = jnp.array(1., jnp.float32) * (1.0 - jnp.float32(continuous_mode))

    # update metrics (State.metrics is a dict; keep it, but replace key we care)
    new_metrics = state.metrics.copy()
    new_metrics.update(inner_steps=inner_steps)

    return state.replace(pipeline_state=s_f.data, obs=obs, reward=reward, done=done, metrics=new_metrics)

# ---------- Env class (thin wrapper) ----------

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

        self.ctrl_param = core.core_build_pid_param()

        model_dt = float(mj_model.opt.timestep)
        n_frames = int(round((1.0 / control_freq) / model_dt))

        # still inherit PipelineEnv for wrappers; we'll NOT call pipeline_step in jit
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        # runtime scalars/arrays
        self.continuous_mode = False
        self.kp = float(kp)
        self.ki = float(ki)
        self.tol = float(tol)
        self.max_inner = int(max_inner_steps)

        self.nt = len(self.p.ids.tendon)
        self.dL_max = 0.005
        self.L_min  = jnp.ones((self.nt,), jnp.float32) * 0.24
        self.L_max  = jnp.ones((self.nt,), jnp.float32) * 0.33

        # goal pos+quat (normalize quat part)
        g = jnp.array([-0.08655875, 0.02789154, 0.8624209,
                        0.53883886, -0.4978265, 0.40249392, 0.54755914], dtype=jnp.float32)
        self.goal = g.at[3:].set(g[3:] / (jnp.linalg.norm(g[3:]) + 1e-8))

        # cache scalar ids to avoid Python/container in jit
        self.tip_site_id = int(self.p.ids.site[-1])

    @property
    def action_size(self) -> int:
        return int(self.nt)

    def reset(self, rng):
        data = self.pipeline_init(self.sys.qpos0, jnp.zeros(self.sys.nv))
        s0 = core.CoreState(data=data, t=jnp.array(0, jnp.int32))
        obs = self._get_obs(s0)
        zero = jnp.array(0., jnp.float32)
        metrics = dict(inner_steps=jnp.array(0, jnp.float32))
        return State(data, obs, zero, zero, metrics)

    def step(self, state, action):
        # thin call into pure-jit function
        return _step_env_jit(
            state, action,
            self.p, self.ctrl_param,
            jnp.float32(self.kp), jnp.float32(self.ki), jnp.float32(self.tol),
            bool(self.continuous_mode),
            self.goal, int(self.tip_site_id),
            int(self.max_inner),
            jnp.float32(self.dL_max),
            self.L_min, self.L_max
        )

    def _get_obs(self, s_f: core.CoreState) -> jax.Array:
        tendon = sensors.tendon_state(s_f, self.p.ids.tendon)
        imu = s_f.data.sensordata  # assume fixed length by model
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


    t0 = time.perf_counter()
    key = random.PRNGKey(0)
    state = env.reset(key)
    action = jnp.array([0.30562606, 0.28558427, 0.28487587, 0.20157896, 0.28575578, 0.21900828, 0.14331605, 0.30143574, 0.33560848], dtype=jnp.float32)

    state = env.step(state, action)

    rew = device_get(state.reward)
    done = device_get(state.done)
    inner_steps = device_get(state.metrics['inner_steps'])
    t1 = time.perf_counter()
    duration = t1 - t0

    print(f"reward={float(rew):.4f}  inner_steps={int(inner_steps)}  done={float(done):.0f}  ")
    print(f"Total time taken: {duration:.4f} seconds")
    
    t0 = time.perf_counter()
    key = random.PRNGKey(0)
    state = env.reset(key)
    state = env.step(state, action)
    rew = device_get(state.reward)
    done = device_get(state.done)
    inner_steps = device_get(state.metrics['inner_steps'])
    t1 = time.perf_counter()
    duration = t1 - t0
    print(f"reward={float(rew):.4f}  inner_steps={int(inner_steps)}  done={float(done):.0f}  ")
    print(f"Total time taken: {duration:.4f} seconds")
    


# if __name__ == "__main__":
#     import time
#     import numpy as onp
#     from jax import random, device_get

#     xml_path = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml"

#     control_freq = 25.0
#     kp, ki = 2.0, 0.3
#     tol = 1e-3
#     max_inner_steps = 400

#     env = FlexPALEnv(
#         xml_path=xml_path,
#         control_freq=control_freq,
#         kp=kp,
#         ki=ki,
#         tol=tol,
#         max_inner_steps=max_inner_steps,
#     )


#     t0 = time.perf_counter()
#     key = random.PRNGKey(0)
#     state = env.reset(key)
#     action = jnp.array([0.30562606, 0.28558427, 0.28487587, 0.20157896, 0.28575578, 0.21900828, 0.14331605, 0.30143574, 0.33560848], dtype=jnp.float32)

#     state = env.step(state, action)

#     rew = device_get(state.reward)
#     done = device_get(state.done)
#     inner_steps = device_get(state.metrics['inner_steps'])
#     t1 = time.perf_counter()
#     duration = t1 - t0

#     print(f"reward={float(rew):.4f}  inner_steps={int(inner_steps)}  done={float(done):.0f}  ")
#     print(f"Total time taken: {duration:.4f} seconds")
    
#     t0 = time.perf_counter()
#     key = random.PRNGKey(0)
#     state = env.reset(key)
#     state = env.step(state, action)
#     rew = device_get(state.reward)
#     done = device_get(state.done)
#     inner_steps = device_get(state.metrics['inner_steps'])
#     t1 = time.perf_counter()
#     duration = t1 - t0
#     print(f"reward={float(rew):.4f}  inner_steps={int(inner_steps)}  done={float(done):.0f}  ")
#     print(f"Total time taken: {duration:.4f} seconds")
    
