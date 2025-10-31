import jax
from brax import envs
from brax.training.agents.ppo import checkpoint as ppo_ckpt
from flexpal.flexpal.mujoco.flex_env_steps import FlexPALEnv

envs.register_environment('flexpal', FlexPALEnv)
env = envs.get_environment('flexpal')

ckpt_dir = "/tmp/ppo_flexpal_ckpt"

# 直接得到“推理函数”（不用自己构建网络/不用管版本差异）
# 返回的是一个 callable： act, info = inference_fn(obs, rng=None)
inference_fn = ppo_ckpt.load_policy(
    ckpt_dir,
    deterministic=True,   # True=取均值动作；False=按策略分布采样
)

# rollout 示例（JIT 可选）
jit_reset = jax.jit(env.reset)
jit_step  = jax.jit(env.step)
jit_infer = jax.jit(lambda obs, rng: inference_fn(obs, rng))

key = jax.random.PRNGKey(0)
state = jit_reset(key)

for _ in range(200):
    key, sub = jax.random.split(key)
    act, _ = jit_infer(state.obs, sub)   # 大多数版本 rng 可 None；这里给上以防万一
    state = jit_step(state, act)

print("✅ rollout finished")
