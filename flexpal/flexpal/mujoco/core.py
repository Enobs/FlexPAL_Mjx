# core.py
# Minimal MJX core (JAX-ready). No rendering, no I/O.
import time
from flax import struct
from typing import Any, Optional
import abc
import jax
import jax.numpy as jnp
from jax import lax
import mujoco
from mujoco import mjx
from ids import Ids, build_ids


# =========================
# Params & State (PyTree)
# =========================
@struct.dataclass
class CoreParams:
    mjx_model: Any          # mjx.put_model(mj_model)
    model_dt: float         # e.g., mj_model.opt.timestep
    ctrl_dt: float          # 1.0 / control_freq
    substeps: int           # round(ctrl_dt / model_dt)
    ids: Ids = Ids() 


@struct.dataclass
class CoreState:
    data: Any               # mjx.Data
    t: jnp.ndarray          # int32 step counter

@struct.dataclass
class PIDPiecewise:
    k1 = 0.0005
    k2 = 0.001
    k3 = 0.002
    k4 = 0.003
    tol= 1e-2
    
class FlexPALCore(abc.ABC):
    """ Soft robot base class. """

    def __init__(self):
        xml_path = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml"
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)

        self.actuator_index = [
                "L_axial_0" ,
                "L_axial_1" ,
                "L_axial_2" ,
                "LL_axial_0",
                "LL_axial_1",
                "LL_axial_2",
                "LLL_axial_0",
                "LLL_axial_1",
                "LLL_axial_2"]
        self.jnt_num = 9
        
        p = self.core_build_params(mj_model, control_freq=25, sites=("LLLend_effector",), bodies=("Lsec_2",))
        s = self.core_reset(p)
        ctrl = jnp.zeros((mj_model.nu,), dtype=jnp.float32)

        # Warming up JIT compilation 
        s_warmed_up = self.core_step(p, s, ctrl)
        s_warmed_up.data.qpos.block_until_ready()

def core_build_params(mj_model, control_freq: float,
                    bodies=(), sites=(), tendons=(), joints=()) -> CoreParams:
    model_dt = float(mj_model.opt.timestep)
    ctrl_dt  = 1.0 / float(control_freq)
    substeps = max(1, int(round(ctrl_dt / model_dt)))
    return CoreParams(
        mjx_model=mjx.put_model(mj_model),
        model_dt=model_dt,
        ctrl_dt=ctrl_dt,
        substeps=substeps,
        ids=build_ids(mj_model, bodies=bodies, sites=sites, tendons=tendons, joints=joints),
    )

def core_reset(p: CoreParams,
            init_qpos: Optional[jnp.ndarray] = None,
            init_ctrl: Optional[jnp.ndarray] = None) -> CoreState:
    """Reset MJX state (device-side only)."""
    d = mjx.make_data(p.mjx_model)
    if init_qpos is not None:
        d = d.replace(qpos=init_qpos)
    if init_ctrl is not None:
        d = d.replace(ctrl=init_ctrl)
    d = mjx.forward(p.mjx_model, d)
    return CoreState(data=d, t=jnp.array(0, dtype=jnp.int32))

@jax.jit
def core_step(p: CoreParams, s: CoreState, ctrl: jnp.Array) -> CoreState:
    """Apply ctrl for one control period (ctrl_dt), integrating substeps."""
    d0 = s.data.replace(ctrl=ctrl)
    
    def substep(_, data):
        return mjx.step(p.mjx_model, data)
    
    dT = lax.fori_loop(0, p.substeps, substep, d0)
    return CoreState(data=dT, t=s.t + 1)


@jax.jit
def pid_step_single(target_j: jnp.Array, current_j:jnp.Array, param: PIDPiecewise)-> jnp.array_equal:
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

v_pid = jax.vmap(pid_step_single, in_axes=(0, 0, None))

@jax.jit
def inner_step(p,               # CoreParams，里头有 actuator_index、substeps 等
               s,               # CoreState，里头有 data (mjx.Data)
               action,          # [N]  目标（每个关节/执行器）
               pid_params,      # dict，例如 {"Kp": 8.0}
               ):
    
    ctrl_full = s.data.ctrl                     # [A]
    aidx = p.actuator_index                     # [N] int32
    ctrl_sel = ctrl_full[aidx]                  # [N]
    reach_mask = jnp.abs(action - ctrl_sel) < tol   # [N] bool

    u = v_pid(action, ctrl_sel, pid_params)         # [N]
    u = jnp.clip(u, -350.0, 250.0)                  # 饱和

    new_sel = jnp.where(reach_mask, ctrl_sel, u)    # [N]

    ctrl_full_new = ctrl_full.at[aidx].set(new_sel) # [A]
    s_new = s.replace(data=s.data.replace(ctrl=ctrl_full_new))

    pose_reach = jnp.all(reach_mask).astype(jnp.int32)  # 0/1

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
    # 1. 加载模型
    xml_path = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    p = core_build_params(mj_model, control_freq=25, sites=("LLLend_effector",), bodies=("Lsec_2",))
    s = core_reset(p)
    ctrl = jnp.zeros((mj_model.nu,), dtype=jnp.float32)

    # 3. 预热 JIT 编译
    print("--- Warming up JIT compilation ---")
    s_warmed_up = core_step_jit(p, s, ctrl)
    s_warmed_up.data.qpos.block_until_ready()
    print("Warm-up complete.")

    # 4. 运行一个用于计时的循环
    T = 3000
    print(f"\n--- Running Timed Simulation for {T} Steps ---")
    
    t0 = time.perf_counter()
    s_current = s_warmed_up
    for _ in range(T - 1):
        s_current = core_step_jit(p, s_current, ctrl)
    
    s_current.data.qpos.block_until_ready()
    t1 = time.perf_counter()

    # 5. 计算并打印最终的性能指标
    duration = t1 - t0
    physics_sps = (T * p.substeps) / duration if duration > 0 else 0

    print(f"\n--- Final Performance Report ---")
    print(f"Total time taken: {duration:.4f} seconds")
    print(f"Average time per control step: {duration / T * 1000:.4f} ms")
    print(f"Physics Steps per Second: {physics_sps:.1f}  <-- [THE KEY METRIC]")
