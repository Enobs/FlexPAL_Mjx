import os, functools
import jax, jax.numpy as jnp
from brax import envs
from brax.training.agents.ppo import train as ppo
import flax
import msgpack
from brax.training.agents.ppo import checkpoint as ppo_ckpt
from brax.training.agents.ppo import networks as ppo_networks
from flexpal.mujoco.flex_env_steps import FlexPALEnv   

# os.environ["JAX_ENABLE_X64"] = "1" 

envs.register_environment('flexpal', FlexPALEnv)
env = envs.get_environment('flexpal')

jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
state = jit_reset(jax.random.PRNGKey(0))
act   = jax.random.uniform(jax.random.PRNGKey(1), (env.action_size,), minval=-1., maxval=1.)
state = jit_step(state, act)

NUM_ENVS      = 192
UNROLL_LENGTH = 2
BATCH_SIZE    = NUM_ENVS * UNROLL_LENGTH     
NUM_MINIBATCH = 32                          

train_fn = functools.partial(
    ppo.train,
    num_timesteps=30_072_000,    
    num_evals=10,               
    reward_scaling=1.0,
    episode_length=200,
    normalize_observations=False,
    action_repeat=1,
    unroll_length=UNROLL_LENGTH,
    num_minibatches=NUM_MINIBATCH,
    num_updates_per_batch=2,    
    discounting=0.97,
    learning_rate=5e-4,
    entropy_cost=1e-3,
    num_envs=NUM_ENVS,
    batch_size=BATCH_SIZE,
    seed=0,
)

import time
from datetime import datetime
last = {"t": time.time(), "steps": 0, "t0": time.time()}

def progress_fn(step, metrics):
    reward_mean = metrics.get('eval/episode_reward', None)
    reward_std  = metrics.get('eval/episode_reward_std', None)
    ep_len_mean = metrics.get('eval/avg_episode_length', None)
    sps         = metrics.get('eval/sps', None)
    walltime    = metrics.get('eval/walltime', None)

    def fmt(x, n=3):
        return f"{x:.{n}f}" if x is not None else "—"

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"[{step:>10}] "
        f"SPS={fmt(sps)} | "
        f"len={fmt(ep_len_mean,1)} | "
        f"reward={fmt(reward_mean,3)}±{fmt(reward_std,3)} | "
        f"time={fmt(walltime,1)} s"
    )


make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress_fn)

ckpt_dir = os.path.join("/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal", "checkpoints", "ppo_flexpal")
os.makedirs(ckpt_dir, exist_ok=True)

cfg = ppo_ckpt.network_config(
    observation_size=env.observation_size,
    action_size=env.action_size,
    normalize_observations=False,     
    network_factory=ppo_networks.make_ppo_networks,  
)

ppo_ckpt.save(ckpt_dir, step=2000, params=params, config=cfg)
print(f"✅ Saved PPO checkpoint to {ckpt_dir}")