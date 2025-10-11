# kinematics.py
import jax, jax.numpy as jnp
from mujoco import mjx
from typing import Any

def _forward_site_pos(mjx_model, qpos, site_id: int):
    # 构出一个 data，用给定 qpos 计算该 site 的位置
    # 注意：不要从外部 capture 整个 State（避免意外闭包）；只用传入的 qpos 构造
    d = mjx.make_data(mjx_model).replace(qpos=qpos)
    d = mjx.forward(mjx_model, d)
    return d.site_xpos[site_id]

def site_jacobian(mjx_model, s: Any, site_id: int) -> jnp.ndarray:
    """返回 d(site_pos)/d(qpos)，形状 (3, nq)"""
    q = s.data.qpos
    f = lambda qpos: _forward_site_pos(mjx_model, qpos, site_id)
    jac = jax.jacobian(f)(q)               # (3, nq)
    return jac

def body_pos_from_q(mjx_model, qpos, body_id: int):
    d = mjx.make_data(mjx_model).replace(qpos=qpos)
    d = mjx.forward(mjx_model, d)
    return d.xpos[body_id]

def body_jacobian(mjx_model, s: Any, body_id: int) -> jnp.ndarray:
    """返回 d(body_pos)/d(qpos)，形状 (3, nq)"""
    q = s.data.qpos
    f = lambda qpos: body_pos_from_q(mjx_model, qpos, body_id)
    return jax.jacobian(f)(q)
