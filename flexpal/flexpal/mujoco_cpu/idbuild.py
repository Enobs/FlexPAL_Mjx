# flexpal/cpu/idbuild.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple, List
import numpy as np
import mujoco


@dataclass
class CheckId:
    body:     Dict[str, int] = field(default_factory=dict)
    site:     Dict[str, int] = field(default_factory=dict)
    tendon:   Dict[str, int] = field(default_factory=dict)
    joint:    Dict[str, int] = field(default_factory=dict)
    actuator: Dict[str, int] = field(default_factory=dict)

@dataclass
class Ids:
    body:     np.ndarray
    site:     np.ndarray
    tendon:   np.ndarray
    joint:    np.ndarray
    actuator: np.ndarray


def _name2id(m: mujoco.MjModel, obj_type: int, name: str) -> int:
    return mujoco.mj_name2id(m, obj_type, name)

def _names2ids(m: mujoco.MjModel, obj_type: int, names: Iterable[str]) -> Tuple[Dict[str, int], np.ndarray]:
    names = list(names)
    d: Dict[str, int] = {n: _name2id(m, obj_type, n) for n in names}
    missing = [n for n, i in d.items() if i < 0]
    if missing:
        raise ValueError(f"Names not found for {obj_type}: {missing}")
    arr = np.array([d[n] for n in names], dtype=np.int32) if names else np.zeros((0,), dtype=np.int32)
    return d, arr

def build_ids(
    mj_model: mujoco.MjModel,
    bodies: Iterable[str] = (),
    sites: Iterable[str] = (),
    tendons: Iterable[str] = (),
    joints: Iterable[str] = (),
    actuators: Iterable[str] = (),
) -> Tuple[CheckId, Ids]:
    b_c, b = _names2ids(mj_model, mujoco.mjtObj.mjOBJ_BODY,     bodies)
    s_c, s = _names2ids(mj_model, mujoco.mjtObj.mjOBJ_SITE,     sites)
    t_c, t = _names2ids(mj_model, mujoco.mjtObj.mjOBJ_TENDON,   tendons)
    j_c, j = _names2ids(mj_model, mujoco.mjtObj.mjOBJ_JOINT,    joints)
    a_c, a = _names2ids(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuators)
    return (
        CheckId(body=b_c, site=s_c, tendon=t_c, joint=j_c, actuator=a_c),
        Ids(body=b, site=s, tendon=t, joint=j, actuator=a),
    )

def gen_actuator_names() -> List[str]:
    secs = ["L", "LL", "LLL"]
    return [f"{sec}_axial_{i}" for sec in secs for i in range(3)]

def gen_site_names() -> List[str]:
    names: List[str] = []
    for sec in ["L", "LL", "LLL"]:
        for row in (0, 1, 2):
            for col in range(1, 8):
                names.append(f"{sec}layer{row}{col}")
        if sec == "LLL":
            names.append("LLLend_effector")  # 末端
    return names

def gen_tendon_names() -> List[str]:
    names: List[str] = []
    for sec in ["L", "LL", "LLL"]:
        for row in (0, 1, 2):
            names.append(f"{sec}layer0{row}")
    return names
