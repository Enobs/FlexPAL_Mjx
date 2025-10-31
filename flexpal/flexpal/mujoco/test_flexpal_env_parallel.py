# test_flexpal_env_parallel.py
import time
import jax
import jax.numpy as jnp
from brax.envs.wrappers.training import VmapWrapper
from flexpal.flexpal.mujoco.flexpal_env_old import FlexPALEnv   # ← 按你的实际路径调整

# ===== 单环境 sanity check =====
env = FlexPALEnv()
key = jax.random.PRNGKey(0)
state = env.reset(key)
action = jax.random.uniform(key, (env.action_size,), minval=-1.0, maxval=1.0)

state = env.step(state, action)
jax.block_until_ready(state.reward)
print(f"single env reward={float(state.reward):.4f}")

# ===== 并行测试 =====
B = 32   # 并行环境数，可以改成 64、128 看变化
benv = VmapWrapper(FlexPALEnv(), batch_size=B)

key = jax.random.PRNGKey(0)
state = benv.reset(key)
acts  = jax.random.uniform(key, (B, benv.action_size), minval=-1.0, maxval=1.0)

# --- 编译 (JIT) ---
print("compiling ...")
batched_step = jax.jit(lambda s, a: benv.step(s, a))
compiled = batched_step.lower(state, acts).compile()

# --- 执行一次 ---
t0 = time.perf_counter()
state = compiled(state, acts)
jax.block_until_ready(state.reward)
t1 = time.perf_counter()

print(f"[B={B}] mean_reward={float(jnp.mean(state.reward)):.4f}  time={t1 - t0:.3f}s")

# ====== 多次执行统计平均 ======
N = 5
times = []
for i in range(N):
    t0 = time.perf_counter()
    state = compiled(state, acts)
    jax.block_until_ready(state.reward)
    t1 = time.perf_counter()
    times.append(t1 - t0)

mean_t = sum(times) / N
print(f"[B={B}] avg_time_per_step = {mean_t:.4f}s   ({B/mean_t:.1f} env/s total)")
