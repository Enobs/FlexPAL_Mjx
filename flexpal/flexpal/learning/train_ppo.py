# flexpal_env/learning/train_ppo.py
from jax import random
from brax.training.agents.ppo import train as ppo_train
import jax

from flexpal.mujoco.flexpal_env import FlexPALEnv
print("Backend:", jax.default_backend(), "Devices:", jax.devices(), flush=True)

XML_PATH = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml"

def progress_fn(step, m):
    loss = float(m.get("training/total_loss", m.get("loss", 0.0)))
    sps  = float(m.get("training/sps", m.get("sps", 0.0)))
    print(f"[step {int(step):7d}]  sps={sps:.1f}  loss={loss:.6f}", flush=True)

def make_env():
    env = FlexPALEnv(
        xml_path=XML_PATH,
        control_freq=25.0,
        kp=2.0,
        ki=0.3,
        tol=1e-3,
        max_inner_steps=400,
    )
    env.continuous_mode = False
    return env

if __name__ == "__main__":
    import time 
    t0 = time.perf_counter()
    from brax.envs.wrappers.training import wrap
    import jax.numpy as jnp
    from jax import random

    env = make_env()

    make_policy, params, metrics = ppo_train.train(
        environment=env,
        wrap_env=True,              
        num_timesteps=2560,
        num_envs=64,                
        episode_length=1,           
        unroll_length=1,
        batch_size=64,              
        num_minibatches=4,
        num_updates_per_batch=2,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.9,             
        gae_lambda=0.95,
        clipping_epsilon=0.2,
        max_grad_norm=0.5,
        normalize_observations=False,
        reward_scaling=0.1,
        log_training_metrics=True,  
        progress_fn = progress_fn,
        run_evals=False,
        seed=0,
    )

    
    key = random.PRNGKey(42)
    policy = make_policy(params)   
    state = env.reset(key)
    act, _ = policy(state.obs, key)
    state = env.step(state, act)
    t1 = time.perf_counter()
    during = t1 - t0
    print("eval reward:", float(state.reward))
    print("time:", float(during))
    