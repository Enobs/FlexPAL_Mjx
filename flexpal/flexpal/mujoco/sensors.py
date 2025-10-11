# sensors.py
import jax.numpy as jnp
from typing import Any

# s: CoreState
def qpos(s: Any) -> jnp.ndarray:
    return s.data.qpos                      # (nq,)

def qvel(s: Any) -> jnp.ndarray:
    return s.data.qvel                      # (nv,)

def body_pos(s: Any, body_id: int) -> jnp.ndarray:
    return s.data.xpos[body_id]             # (3,)

def body_quat(s: Any, body_id: int) -> jnp.ndarray:
    return s.data.xquat[body_id]            # (4,)

def body_rotm(s: Any, body_id: int) -> jnp.ndarray:
    return s.data.xmat[body_id].reshape(3, 3)

def body_linvel(s: Any, body_id: int) -> jnp.ndarray:
    return s.data.cvel[body_id, :3]         # (3,)

def body_angvel(s: Any, body_id: int) -> jnp.ndarray:
    return s.data.cvel[body_id, 3:]         # (3,)

def site_pos(s: Any, site_id: int) -> jnp.ndarray:
    return s.data.site_xpos[site_id]        # (3,)

def tendon_length(s: Any, tendon_id: int) -> jnp.ndarray:
    return s.data.ten_length[tendon_id]     # ()
