# flexpal/cpu/core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import mujoco

from . import idbuild
from . import sensors


@dataclass
class CoreParams:
    mj_model: mujoco.MjModel
    model_dt: float
    ctrl_dt:  float
    substeps: int
    cids: idbuild.CheckId
    ids:  idbuild.Ids

@dataclass
class CoreState:
    data: mujoco.MjData
    t: int

@dataclass
class PIDPiecewise:
    k1: float = 5e-2
    k2: float = 1e-1
    k3: float = 2e-1
    k4: float = 3e-1
    tol: float = 1e-2
    min: float = -3.5e2
    max: float = 2e2


def core_build_params(
    mj_model: mujoco.MjModel,
    control_freq: float,
    bodies=(), sites=(), tendons=(), joints=(), actuators=()
) -> CoreParams:
    model_dt = float(mj_model.opt.timestep)
    ctrl_dt  = 1.0 / float(control_freq)
    substeps = max(1, int(round(ctrl_dt / model_dt)))
    cids, ids = idbuild.build_ids(
        mj_model, bodies=bodies, sites=sites, tendons=tendons, joints=joints, actuators=actuators
    )
    return CoreParams(
        mj_model=mj_model, model_dt=model_dt, ctrl_dt=ctrl_dt, substeps=substeps, cids=cids, ids=ids
    )

def core_build_pid_param() -> PIDPiecewise:
    return PIDPiecewise()

def core_reset(p: CoreParams, init_ctrl: Optional[np.ndarray] = None) -> CoreState:
    d = mujoco.MjData(p.mj_model)
    mujoco.mj_resetData(p.mj_model, d)
    if init_ctrl is not None and p.mj_model.nu > 0:
        n = min(len(init_ctrl), p.mj_model.nu)
        d.ctrl[:n] = init_ctrl[:n]
    mujoco.mj_forward(p.mj_model, d)
    return CoreState(data=d, t=0)

def core_step(p: CoreParams, s: CoreState, ctrl: np.ndarray) -> CoreState:
    """应用 ctrl（完整长度或只给 actuator 索引），推进一个控制周期（substeps 次 mj_step）。"""
    d = s.data
    # 写入控制：假定 ctrl 是 actuator 选择后的长度；如需映射/限幅可在外部处理
    if p.mj_model.nu > 0:
        n = min(len(ctrl), p.mj_model.nu)
        d.ctrl[:n] = ctrl[:n]
    # 推进
    for _ in range(p.substeps):
        mujoco.mj_step(p.mj_model, d)
    return CoreState(data=d, t=s.t + 1)

def pid_step_single(target: float, current: float, param: PIDPiecewise) -> float:
    err = target - current
    aerr = abs(err)
    if aerr > 0.5:
        out = param.k1 * err + current
    elif aerr > 0.2:
        out = param.k2 * err + current
    elif aerr > 0.1:
        out = param.k3 * err + current
    else:
        out = param.k4 * err + current
    return float(np.clip(out, param.min, param.max))

def inner_step(
    p: CoreParams,
    s: CoreState,
    action: np.ndarray,         # 目标 ctrl（与 actuator 数量一致）
    pid_params: PIDPiecewise,
) -> Tuple[CoreState, int]:
    """对 actuator 通道做分段“P控制”逼近，返回新状态 + 是否到达（全部通道 |action - current| < tol）。"""
    d = s.data
    # 取当前选择的 actuator 通道
    aidx = p.ids.actuator
    ctrl_full = d.ctrl.copy()
    current_sel = ctrl_full[aidx]
    reach_mask = np.abs(action - current_sel) < pid_params.tol

    new_sel = current_sel.copy()
    need_update = ~reach_mask
    if np.any(need_update):
        for i in np.where(need_update)[0]:
            new_sel[i] = pid_step_single(action[i], current_sel[i], pid_params)
        ctrl_full[aidx] = new_sel
        d.ctrl[:] = ctrl_full

    pose_reach = int(np.all(reach_mask))
    return CoreState(data=d, t=s.t), pose_reach
