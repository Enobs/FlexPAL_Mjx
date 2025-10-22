# control.py
from typing import Tuple
from flax import struct
import jax, jax.numpy as jnp

@struct.dataclass
class SensorPIDParams:
    kp: float
    ki: float
    kd: float
    u_min: float = -350.0
    u_max: float =  250.0
    i_min: float = -1e6
    i_max: float =  1e6

@struct.dataclass
class PIDState:
    integ: jax.Array     # [...]
    prev_err: jax.Array  # [...]

def _pid_single(integ, prev_err, target, current, p: PIDParams, dt: float):
    err = target - current
    integ_new = jnp.clip(integ + err * dt, p.i_min, p.i_max)
    deriv = (err - prev_err) / jnp.maximum(dt, 1e-8)
    u = p.kp * err + p.ki * integ_new + p.kd * deriv
    u = jnp.clip(u, p.u_min, p.u_max)
    return (integ_new, err), u

def pid_step(state: PIDState, target: jax.Array, current: jax.Array,
             params: PIDParams, dt: float) -> Tuple[PIDState, jax.Array]:
    # target/current 同 shape：可以是 [N]、[N,3] 等
    in_axes = (0, 0, 0, 0, None, None) if target.ndim >= 1 else (None, None, None, None, None, None)
    vfun = jax.vmap(_pid_single, in_axes=in_axes)
    (integ_new, prev_err_new), u = vfun(state.integ, state.prev_err, target, current, params, dt)
    return PIDState(integ_new, prev_err_new), u

@struct.dataclass
class SensorOuterParams:
    pid: PIDParams
    tol: float          # 到位阈值（在传感量空间里，例如末端位置误差的范数阈值）

@struct.dataclass
class InnerActParams:
    pid: PIDParams

@struct.dataclass
class ControlParams:
    outer: SensorOuterParams
    inner: InnerActParams

@struct.dataclass
class ControlState:
    pid_outer: PIDState
    pid_inner: PIDState

def control_pipeline_sensor_outer(
    ctrl_p: ControlParams,
    ctrl_s: ControlState,
    # 传感空间目标/当前（例如：末端位置 [K,3]）
    sensor_target: jax.Array,   # [K,3]
    sensor_current: jax.Array,  # [K,3]
    # 执行器当前与索引
    ctrl_full: jax.Array,       # [A]
    actuator_index: jax.Array,   # [N]
    dt: float
):
    # === 外环：传感空间 PID，输出内环参考（映射为每个关节/执行器参考）===
    # 这里简单示例：把外环 u_outer 压缩/投影到 N 维（你可以放自己的雅可比/映射）
    # 例：多末端(K*3)误差 -> 取均值作为全局误差，再分配到各执行器
    pid_outer_new, u_outer = pid_step(ctrl_s.pid_outer, sensor_target, sensor_current,
                                      ctrl_p.outer.pid, dt)     # [K,3]
    # 简单聚合成标量或 N 维：这里演示把 (K,3) 的均值作为全局误差，再复制到 N 维
    u_outer_scalar = jnp.mean(u_outer)                          # ()
    N = actuator_index.shape[0]
    inner_ref = jnp.ones((N,)) * u_outer_scalar                 # [N]
    # === 内环：执行器空间 PID ===
    current_inner = ctrl_full[actuator_index]                   # [N]
    pid_inner_new, u_inner = pid_step(ctrl_s.pid_inner, inner_ref, current_inner,
                                      ctrl_p.inner.pid, dt)     # [N]
    # 到位判定（外环或内环都可以，这里用外环的 L2 误差）
    outer_err_norm = jnp.linalg.norm((sensor_target - sensor_current).reshape(-1))
    reached = (outer_err_norm < ctrl_p.outer.tol).astype(jnp.int32)

    # 写回 ctrl
    ctrl_full_new = ctrl_full.at[actuator_index].set(u_inner)
    ctrl_s_new = ControlState(pid_outer=pid_outer_new, pid_inner=pid_inner_new)
    return ctrl_full_new, ctrl_s_new, reached, dict(outer_err=outer_err_norm)
