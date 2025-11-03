# flexpal/cpu/sensors.py
from __future__ import annotations
import numpy as np
import mujoco
from typing import Any

# 若已有你自己的旋转到四元数转换，可替换这里：
def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 -> (w, x, y, z) 四元数"""
    # 标准算法，数值稳定性足够
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    t = m00 + m11 + m22
    if t > 0.0:
        S = (t + 1.0) ** 0.5 * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = (1.0 + m00 - m11 - m22) ** 0.5 * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = (1.0 + m11 - m00 - m22) ** 0.5 * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = (1.0 + m22 - m00 - m11) ** 0.5 * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    # 归一化
    q = q / (np.linalg.norm(q) + 1e-8)
    return q

# ---- 读取 MjData 的便捷函数（CPU 版） ----

def site_pos(d: mujoco.MjData, site_id: int) -> np.ndarray:
    """(3,) world position of site"""
    return d.site_xpos[site_id].copy()

def site_quat_world(d: mujoco.MjData, site_id: int) -> np.ndarray:
    """(4,) world orientation of site as quaternion (w,x,y,z)"""
    R = d.site_xmat[site_id].reshape(3, 3)
    return rotmat_to_quat(R)

def tendon_state(d: mujoco.MjData, tendon_ids: np.ndarray) -> np.ndarray:
    return d.ten_length[tendon_ids].copy()

# 如需 body/vel 等，也给出 CPU 版接口
def body_pos(d: mujoco.MjData, body_id: int) -> np.ndarray:
    # MuJoCo 的 body world pos 在 xipos
    return d.xipos[body_id].copy()

def body_rotm(d: mujoco.MjData, body_id: int) -> np.ndarray:
    return d.xmat[body_id].reshape(3, 3).copy()

def body_quat(d: mujoco.MjData, body_id: int) -> np.ndarray:
    R = body_rotm(d, body_id)
    return rotmat_to_quat(R)

def body_linvel(d: mujoco.MjData, body_id: int) -> np.ndarray:
    # cvel: 6 x nb (ang[0:3], lin[3:6]) or (lin, ang) 取决于构建；这里用官方文档顺序：
    return d.cvel[body_id, :3].copy()

def body_angvel(d: mujoco.MjData, body_id: int) -> np.ndarray:
    return d.cvel[body_id, 3:].copy()
