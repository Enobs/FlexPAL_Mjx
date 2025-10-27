# sensors.py
import jax.numpy as jnp
from typing import Any
import jax
from jax import lax
from flax import struct
from flexpal.common.rotation import rotmat_to_quat

# s: CoreState
def qpos(s: Any) -> jax.Array:
    return s.data.qpos                      # (nq,)

def qvel(s: Any) -> jax.Array:
    return s.data.qvel                      # (nv,)

def body_pos(s: Any, body_id: int) -> jax.Array:
    return s.data.xpos[body_id]             # (3,)

def body_quat(s: Any, body_id: int) -> jax.Array:
    return s.data.xquat[body_id]            # (4,)

def body_rotm(s: Any, body_id: int) -> jax.Array:
    return s.data.xmat[body_id].reshape(3, 3)

def body_linvel(s: Any, body_id: int) -> jax.Array:
    return s.data.cvel[body_id, :3]         # (3,)

def body_angvel(s: Any, body_id: int) -> jax.Array:
    return s.data.cvel[body_id, 3:]         # (3,)

def site_pos(s: Any, site_id: int) -> jax.Array:
    return s.data.site_xpos[site_id]        # (3,)

def tendon_length(s: Any, tendon_id: int) -> jax.Array:
    return s.data.ten_length[tendon_id]     # ()

def tendon_state(s: Any, tendon_ids: jax.Array) -> jax.Array:
    return s.data.ten_length[tendon_ids]

def site_quat_world(s, site_id: int) -> jnp.ndarray:
    R_ws = s.data.site_xmat[site_id].reshape(3,3)   
    q_ws = rotmat_to_quat(R_ws)                     
    return q_ws
