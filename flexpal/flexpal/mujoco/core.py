# core.py
# Minimal MJX core (JAX-ready). No rendering, no I/O.
import time
from flax import struct
from typing import Any, Optional

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


# =========================
# One control-period step
# =========================
def core_step(p: CoreParams, s: CoreState, ctrl: jnp.ndarray) -> CoreState:
    """Apply ctrl for one control period (ctrl_dt), integrating substeps."""
    d0 = s.data.replace(ctrl=ctrl)

    def substep(_, data):
        return mjx.step(p.mjx_model, data)

    dT = lax.fori_loop(0, p.substeps, substep, d0)
    return CoreState(data=dT, t=s.t + 1)


# JIT-ed version for training/rollout
core_step_jit = jax.jit(core_step)


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


# if __name__ == '__main__':
    
#     xml_path = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml"
#     mj_model = mujoco.MjModel.from_xml_path(xml_path)
#     mj_data  = mujoco.MjData(mj_model)  
#     p  = core_build_params(mj_model, control_freq=20, sites=("LLLend_effector",), bodies=("Lsec_2",))
#     s = core_reset(p)
#     sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "LLLend_effector")
    

#     ctrl = jnp.zeros((mj_model.nu,), dtype=jnp.float32)   
#     print("Warming up JIT for the single-step function... This will be fast.")
#     s_after_warmup = core_step_jit(p, s, ctrl) 
#     s_after_warmup.data.qpos.block_until_ready()
#     print("Warm-up complete.")

#     T = 300
#     print(f"\nRunning simulation")
#     t0 = time.perf_counter()
    
#     s_current = s_after_warmup
#     for _ in range(T): 
#         s_current = core_step_jit(p, s_current, ctrl)
    
#     s_final = s_current
#     s_final.data.qpos.block_until_ready()
    
#     t1 = time.perf_counter()
#     print(f"Running {T} steps in a loop took {t1 - t0:.4f}s")


if __name__ == '__main__':
    # 1. 加载模型
    xml_path = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    # =====================================================================
    # ==                 这 是 我 们 唯 一 需 要 的 信 息                ==
    # =====================================================================
    print("--- Verifying Model Physics Options ---")
    integrator_code = mj_model.opt.integrator
    integrator_name = mujoco.mjtIntegrator(integrator_code).name
    print(f"Loaded Integrator: {integrator_name} (Code: {integrator_code})")
    print(f"Loaded Timestep: {mj_model.opt.timestep}")
    print(f"Loaded Solver Iterations: {mj_model.opt.iterations}")
    
    if integrator_name != 'mjINT_IMPLICITFAST':
        print("\n[!!! FATAL WARNING !!!] The stable 'implicitfast' integrator was NOT loaded!")
        print("This is the reason for the extreme slowness. Please check your XML file again.\n")
    else:
        print("\n[SUCCESS] The stable 'implicitfast' integrator is correctly loaded. Performance should be high.\n")
    # =====================================================================

    # 2. 构建 MJX 参数
    p = core_build_params(mj_model, control_freq=20, sites=("LLLend_effector",), bodies=("Lsec_2",))
    s = core_reset(p)
    ctrl = jnp.zeros((mj_model.nu,), dtype=jnp.float32)

    # 3. 预热 JIT 编译
    print("--- Warming up JIT compilation ---")
    s_warmed_up = core_step_jit(p, s, ctrl)
    s_warmed_up.data.qpos.block_until_ready()
    print("Warm-up complete.")

    # 4. 运行一个用于计时的循环
    T = 300
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
