import os, functools
import jax, jax.numpy as jnp
from brax import envs
from brax.training.agents.ppo import train as ppo
from orbax import checkpoint as ocp
import os



# os.environ["JAX_ENABLE_X64"] = "1"

from flexpal.mujoco.flex_env_new import FlexPALEnv   

envs.register_environment('flexpal', FlexPALEnv)
env = envs.get_environment('flexpal')

jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
state = jit_reset(jax.random.PRNGKey(0))
act   = jax.random.uniform(jax.random.PRNGKey(1), (env.action_size,), minval=-1., maxval=1.)
state = jit_step(state, act)

NUM_ENVS      = 384       
UNROLL_LENGTH = 5         
BATCH_SIZE    = NUM_ENVS * UNROLL_LENGTH
NUM_MINIBATCH = 15        

train_fn = functools.partial(
    ppo.train,
    num_timesteps=300_000,       
    num_evals=10,
    reward_scaling=1.0,
    episode_length=40,          
    normalize_observations=True,
    action_repeat=1,
    unroll_length=UNROLL_LENGTH,
    num_minibatches=NUM_MINIBATCH,
    num_updates_per_batch=8,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=NUM_ENVS,
    batch_size=BATCH_SIZE,
    seed=0,
)

import time
from datetime import datetime
last = {"t": time.time(), "steps": 0, "t0": time.time()}

def progress(steps, metrics):
  now = time.time()
  dt = now - last["t"]
  dS = steps - last["steps"]

  sps_inst = dS / dt if dt > 0 else float("nan")
  sps_avg = steps / (now - last["t0"]) if (now - last["t0"]) > 0 else float("nan")

  reward = metrics.get("eval/episode_reward", float("nan"))
  reward_std = metrics.get("eval/episode_reward_std", float("nan"))
  loss = metrics.get("loss", float("nan"))

  print(
      f"[{datetime.now().strftime('%H:%M:%S')}] "
      f"steps={steps:>8,d} | "
      f"SPS_inst={sps_inst:>7.0f}  SPS_avg={sps_avg:>7.0f} | "
      f"reward={reward:>8.3f}±{reward_std:<8.3f} | "
      f"loss={loss:<8.3f}"
  )

  last["t"] = now
  last["steps"] = steps

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

# 训练完成后
ckpt_dir = "checkpoints/flexpal"
os.makedirs(ckpt_dir, exist_ok=True)

# 建一个 checkpointer
checkpointer = ocp.PyTreeCheckpointer()

# 保存参数（params 是 brax ppo.train 返回的第二个对象）
save_args = ocp.args.StandardSave(args=ocp.args.PyTreeSave(params))
checkpointer.save(ckpt_dir, params, save_args=save_args)
print(f"✅ saved params to {ckpt_dir}")