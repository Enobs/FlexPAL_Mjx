# core.py
# Minimal MJX core (JAX-ready). No rendering, no I/O.
import time
from flax import struct
from typing import Any, Optional, Tuple
import abc
import jax
import mujoco
import jax.numpy as jnp
from jax import lax
from mujoco import mjx
import flexpal.mujoco.idbuild as idbuild
import flexpal.mujoco.sensors as sensors


# =========================
# Params & State (PyTree)
# =========================
@struct.dataclass
class CoreParams:
    mjx_model: Any  = struct.field(pytree_node=False)
    model_dt: float = struct.field(pytree_node=False)
    ctrl_dt:  float = struct.field(pytree_node=False)
    substeps: int   = struct.field(pytree_node=False)
    cids: idbuild.CheckId   = struct.field(pytree_node=False, default_factory=idbuild.CheckId)  
    ids:  idbuild.Ids      = struct.field(default_factory=idbuild.Ids)   


@struct.dataclass
class CoreState:
    data: Any               
    t: jnp.ndarray       

@struct.dataclass
class PIDPiecewise:
    k1: float = struct.field(pytree_node=False, default=5e-2)
    k2: float = struct.field(pytree_node=False, default=1e-1)
    k3: float = struct.field(pytree_node=False, default=2e-1)
    k4: float = struct.field(pytree_node=False, default=3e-1)
    tol: float = struct.field(pytree_node=False, default=1e-2)
    min: float = struct.field(pytree_node=False, default=-3.5e2)
    max: float = struct.field(pytree_node=False, default=2e2)
    


def core_build_params(mj_model, control_freq: float,
                      bodies=(), sites=(), tendons=(), joints=(), actuators=()) -> CoreParams:
    model_dt = float(mj_model.opt.timestep)
    ctrl_dt  = 1.0 / float(control_freq)
    substeps = max(1, int(round(ctrl_dt / model_dt)))

    cids, ids = idbuild.build_ids(
        mj_model, bodies=bodies, sites=sites, tendons=tendons, joints=joints, actuators=actuators
    )

    return CoreParams(
        mjx_model=mjx.put_model(mj_model),
        model_dt=model_dt,
        ctrl_dt=ctrl_dt,
        substeps=substeps,
        cids=cids,
        ids=ids,
    )
    
def core_build_pid_param()->PIDPiecewise:
    return PIDPiecewise()

def core_reset(p: CoreParams,
            init_ctrl: Optional[jnp.ndarray] = None) -> CoreState:
    """Reset MJX state (device-side only)."""
    d = mjx.make_data(p.mjx_model)
    if init_ctrl is not None:
        d = d.replace(ctrl=init_ctrl)
    d = mjx.forward(p.mjx_model, d)
    return CoreState(data=d, t=jnp.array(0, dtype=jnp.int32))

@jax.jit
def core_step(p: CoreParams, s: CoreState, ctrl: jax.Array) -> CoreState:
    """Apply ctrl for one control period (ctrl_dt), integrating substeps."""
    d0 = s.data.replace(ctrl=ctrl)
    
    def substep(_, data):
        return mjx.step(p.mjx_model, data)
    
    dT = lax.fori_loop(0, p.substeps, substep, d0)
    return CoreState(data=dT, t=s.t + 1)

@jax.jit
def pid_step_single(target_j: jax.Array, current_j:jax.Array, param: PIDPiecewise)-> jax.Array:
    err = target_j - current_j
    abs_err = jnp.abs(err)
    conds = [
        abs_err > 0.5,
        (abs_err <= 0.5) & (abs_err > 0.2),
        (abs_err <= 0.2) & (abs_err > 0.1),
    ]
    choices = [
        param.k1 * err + current_j,
        param.k2 * err + current_j,
        param.k3 * err + current_j,
    ]
    out = jnp.select(conds, choices, default=param.k4 * err + current_j)
    return out

v_pid_core = jax.vmap(pid_step_single, in_axes=(0, 0, None))


@jax.jit
def inner_step(p: CoreParams,              
               s: CoreState,               
               action: jax.Array,          
               pid_params : PIDPiecewise,  
               )-> Tuple[CoreState, jax.Array]:
    
    ctrl_full = s.data.ctrl                     # [A]
    aidx = p.ids.actuator                     # [N] int32
    ctrl_sel = ctrl_full[aidx]                  # [N]
    reach_mask = jnp.abs(action - ctrl_sel) < pid_params.tol   

    u = v_pid_core(action, ctrl_sel, pid_params)       
    u = jnp.clip(u, pid_params.min, pid_params.max)                 
    new_sel = jnp.where(reach_mask, ctrl_sel, u)    
    ctrl_full_new = ctrl_full.at[aidx].set(new_sel) 
    s_new = s.replace(data=s.data.replace(ctrl=ctrl_full_new))

    pose_reach = jnp.all(reach_mask).astype(jnp.int32) 

    return s_new, pose_reach



# =========================
# (Optional) CPU mirror for viewer (never JIT this)
# =========================
def mirror_to_cpu_for_view(mj_model: mujoco.MjModel,
                           mj_data: mujoco.MjData,
                           s: CoreState) -> None:
    """Copy MJX device data to CPU mujoco.MjData for rendering/debug.
    Call only outside jitted code.
    """
    mj_data.qpos[:] = jax.device_get(s.data.qpos)
    mj_data.qvel[:] = jax.device_get(s.data.qvel)
    if mj_model.nu > 0:
        mj_data.ctrl[:mj_model.nu] = jax.device_get(s.data.ctrl[:mj_model.nu])
    mujoco.mj_forward(mj_model, mj_data)



if __name__ == '__main__':
    xml_path = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    actuator_index =idbuild.gen_actuator_names()
    site_index=idbuild.gen_site_names()
    tendon_index = idbuild.gen_tendon_names()
    p = core_build_params(mj_model, control_freq=25, sites=site_index, tendons=tendon_index,actuators=actuator_index)
    s_init = core_reset(p)
    pid_param = core_build_pid_param()
    ctrl_init= jnp.zeros((9,), dtype=jnp.float32)
    s_next, _reach = inner_step(p, s_init, ctrl_init, pid_param)
    warmed_up_s  = core_step(p,s_next, ctrl_init)
    ctrl = jnp.array([1,1,1,1,1,1,1,1,1], dtype=jnp.float32)
    T = 300
    print(f"\n--- Running Timed Simulation for {T} Steps ---")
    
    t0 = time.perf_counter()
    s_current = warmed_up_s
    for _ in range(T - 1):
        s_next, _reach = inner_step(p, s_current, ctrl, pid_param)
        s_current  = core_step(p,s_next, s_next.data.ctrl)
    s_current.data.qpos.block_until_ready()
    t1 = time.perf_counter()

    duration = t1 - t0
    physics_sps = (T * p.substeps) / duration if duration > 0 else 0

    print(f"\n--- Final Performance Report ---")
    print(f"Total time taken: {duration:.4f} seconds")
    print(f"Average time per control step: {duration / T * 1000:.4f} ms")
    print(f"Physics Steps per Second: {physics_sps:.1f}  <-- [THE KEY METRIC]")
    print(f"sensor pos{sensors.site_pos(s_current, p.ids.site[-1])}")
    print(f"sensor quat{sensors.site_quat_world(s_current, p.ids.site[-1])}")
    print(f"current sensor position: {sensors.tendon_state(s_current, p.ids.tendon)}")
