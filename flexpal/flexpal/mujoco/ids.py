# ids.py
from typing import Dict, Iterable
import mujoco
from flax import struct  

@struct.dataclass
class Ids:
    body: Dict[str, int] = struct.field(pytree_node=False, default_factory=dict)
    site: Dict[str, int] = struct.field(pytree_node=False, default_factory=dict)
    tendon: Dict[str, int] = struct.field(pytree_node=False, default_factory=dict)
    joint: Dict[str, int] = struct.field(pytree_node=False, default_factory=dict)

def build_ids(mj_model: mujoco.MjModel,
              bodies: Iterable[str] = (),
              sites: Iterable[str] = (),
              tendons: Iterable[str] = (),
              joints: Iterable[str] = ()) -> Ids:
    b = {n: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in bodies}
    s = {n: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, n) for n in sites}
    t = {n: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_TENDON, n) for n in tendons}
    j = {n: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joints}
    return Ids(body=b, site=s, tendon=t, joint=j)
