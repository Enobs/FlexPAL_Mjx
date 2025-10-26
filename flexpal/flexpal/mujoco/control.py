# control.py
from typing import Tuple
from flax import struct
import jax, jax.numpy as jnp
import core
import sensors
@struct.dataclass
class SensorPIDParams:
    kp: float = struct.field(pytree_node=False,default = 2.0)
    ki: float = struct.field(pytree_node=False,default = 0.3)
    kd: float = struct.field(pytree_node=False,default = 0.0)
    tol: float = struct.field(pytree_node=False,default = 0.003)
    integral: jax.Array = struct.field(
        default_factory=lambda: jnp.zeros((9,), dtype=jnp.float32))

def sensor_build_pid_param():
    return SensorPIDParams()

def v_pid_sensor(ss_g: jax.Array,
                 tendon_full: jax.Array,
                 pid_param: SensorPIDParams,
                 dt: float) -> Tuple[jax.Array, jax.Array]:
    def _single(goal, cur, integ):
        err = goal - cur
        err = jnp.where(jnp.abs(err) < pid_param.tol, 0.0, err)
        i_new = integ + pid_param.ki * err * dt
        delta_u = pid_param.kp * err + i_new
        delta_u = jnp.clip(delta_u, -0.2, 0.2)
        return delta_u, i_new

    delta_u, new_i = jax.vmap(_single)(ss_g, tendon_full, pid_param.integral)
    return delta_u, new_i




@jax.jit
def step_controller(p: core.CoreParams,
                    s: core.CoreState,
                    ss_g: jax.Array,
                    pid_param: SensorPIDParams,
                    ctrl_param: core.PIDPiecewise,
                   ) -> Tuple[core.CoreState, jax.Array, SensorPIDParams]:
    tendon_full = sensors.tendon_state(s, p.ids.tendon)
    reach_mask = jnp.abs(tendon_full - ss_g) < pid_param.tol
    delta_u, new_integral = v_pid_sensor(ss_g, tendon_full, pid_param, p.ctrl_dt)
    u_ctrl = jnp.clip(s.data.ctrl + delta_u, -1.0, 1.0)

    s_next, _ = core.inner_step(p, s, u_ctrl, ctrl_param)
    s_current = core.core_step(p, s_next, s_next.data.ctrl)
    pose_reach = jnp.all(reach_mask).astype(jnp.int32)

    new_pid = pid_param.replace(integral=new_integral)
    return s_current, pose_reach, new_pid



@jax.jit
def loop_until_reach(p, s, ss_g, pid_param, ctrl_param, max_steps=1000):
    def cond_fun(loop_state):
        _s, pose_reach, k, _pid = loop_state
        return (pose_reach == 0) & (k < max_steps)

    def body_fun(loop_state):
        s, _reach, k, pidp = loop_state
        s_next, pose_reach, pidp_next = step_controller(p, s, ss_g, pidp, ctrl_param)
        return (s_next, pose_reach, k + 1, pidp_next)

    init = (s, jnp.array(0, jnp.int32), jnp.array(0, jnp.int32), pid_param)
    s_final, pose_reach_final, steps, pidp_final = jax.lax.while_loop(cond_fun, body_fun, init)
    return s_final, steps, pidp_final



if __name__ == '__main__':
    import mujoco
    import idbuild
    import time
    xml_path = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    actuator_index =idbuild.gen_actuator_names()
    site_index=idbuild.gen_site_names()
    tendon_index = idbuild.gen_tendon_names()
    p = core.core_build_params(mj_model, control_freq=25, sites=site_index, tendons=tendon_index,actuators=actuator_index)
    s_init = core.core_reset(p)
    sensor_pid_param = sensor_build_pid_param()
    pid_param = core.core_build_pid_param()
    ctrl_init= jnp.zeros((9,), dtype=jnp.float32)
    ss_g_init = jnp.array([0.29, 0.29, 0.29, 0.29, 0.29,0.29, 0.29 , 0.29 , 0.29], dtype=jnp.float32)
    _, _ = core.inner_step(p, s_init, ctrl_init, pid_param)
    _  = core.core_step(p,s_init, ctrl_init)
    # _, _ , _ = step_controller(p,s_init,ss_g_init,sensor_pid_param,pid_param)
    _, _, _ = loop_until_reach(p,s_init,ss_g_init,sensor_pid_param,pid_param)
    sensor_target = jnp.array([0.30562606, 0.28558427, 0.28487587, 0.20157896, 0.28575578, 0.21900828, 0.14331605, 0.30143574, 0.33560848], dtype=jnp.float32)
    T = 400
    print(f"\n--- Running Timed Simulation for {T} Steps ---")
    
    t0 = time.perf_counter()
    s_current = s_init
    # for _ in range(T):
        # s_current, reach, sensor_pid_param = step_controller(p, s_current, sensor_target, sensor_pid_param, pid_param)
    s_current, reach, sensor_pid_param  = loop_until_reach(p,s_current,sensor_target,sensor_pid_param,pid_param)
    s_current.data.qpos.block_until_ready()
    t1 = time.perf_counter()

    duration = t1 - t0
    physics_sps = (T * p.substeps) / duration if duration > 0 else 0

    print(f"\n--- Final Performance Report ---")
    print(f"Total time taken: {duration:.4f} seconds")
    print(f"Average time per control step: {duration / T * 1000:.4f} ms")
    print(f"Physics Steps per Second: {physics_sps:.1f}  <-- [THE KEY METRIC]")
    print(f"current sensor position: {sensors.tendon_state(s_current, p.ids.tendon)}")
    print(f"current actuator position: {s_current.data.ctrl}")
    print(f"current sensor data: {s_current.data.sensordata}")
    print(f"current step: {reach}")