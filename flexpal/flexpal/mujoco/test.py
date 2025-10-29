import time, jax, jax.numpy as jnp
from flexpal.mujoco.flexpal_env import FlexPALEnv

env = FlexPALEnv(action_mode="relative")
key = jax.random.PRNGKey(0)
state = env.reset(key)
action = jax.random.uniform(key, (env.action_size,), minval=-1.0, maxval=1.0)

from brax.envs.wrappers.training import VmapWrapper
B = 32
benv = VmapWrapper(FlexPALEnv(action_mode="relative"), batch_size=B)
key   = jax.random.PRNGKey(0)
state = benv.reset(key)
acts  = jax.random.uniform(key, (B, benv.action_size), minval=-1.0, maxval=1.0)

# 编译
batched_step = jax.jit(lambda s, a: benv.step(s, a))
compiled = batched_step.lower(state, acts).compile()

# 执行
t0 = time.perf_counter()
state = compiled(state, acts)
jax.block_until_ready(state.reward)
t1 = time.perf_counter()
print(f"[B={B}] mean_reward={float(jnp.mean(state.reward)):.4f}  time={t1 - t0:.3f}s")
