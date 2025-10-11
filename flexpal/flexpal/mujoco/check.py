import os; os.environ["JAX_LOG_COMPILES"]="1"; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import jax, jax.numpy as jnp
import mujoco
from mujoco import mjx
from flax import struct
from jax import lax

@struct.dataclass
class P: model:any; substeps:int
@struct.dataclass
class S: d:any; t:jnp.ndarray

def build(model, control_freq=20):
    dt = float(model.opt.timestep)
    return P(mjx.put_model(model), max(1, int(round((1.0/control_freq)/dt))))

def reset(p):
    d = mjx.forward(p.model, mjx.make_data(p.model))
    return S(d, jnp.array(0, jnp.int32))

def step(p, s, u):
    d0 = s.d.replace(ctrl=u)
    def sub(_, d): return mjx.step(p.model, d)
    dT = lax.fori_loop(0, p.substeps, sub, d0)
    return S(dT, s.t+1)

def rollout_T(p, s0, u, T:int):
    def body(s, _): return step(p, s, u), None
    return lax.scan(body, s0, None, length=T)

rollout_T_jit = jax.jit(rollout_T, static_argnames=("T",))
xml_path = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
p  = build(model, control_freq=20)
s0 = reset(p)
u  = jnp.zeros((model.nu,), jnp.float32)

# 预热（应打印一次编译）
_  = rollout_T_jit(p, s0, u, 20)[0].d.qpos.block_until_ready()
# 第二次（不应再打印）
_  = rollout_T_jit(p, s0, u, 20)[0].d.qpos.block_until_ready()
