# flexpal/learning/train_ppo.py
from jax import random
from brax.training.agents.ppo import train as ppo_train

# 注意：用包内绝对导入（从项目根目录运行：python -m flexpal.learning.train_ppo）
from flexpal.mujoco.flexpalenv import FlexPALEnv

XML_PATH = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml"

def make_env():
    env = FlexPALEnv(
        xml_path=XML_PATH,
        control_freq=25.0,
        kp=2.0,
        ki=0.3,
        tol=1e-3,
        max_inner_steps=400,
    )
    # 一步到位：每个外层 step 都 done=1
    env.continuous_mode = False

    # 建议：在 _jit_step 里用 self._action_to_target_absolute(action)
    # 让策略输出 ∈[-1,1]，再映射到 [L_min, L_max]，训练更稳
    return env

if __name__ == "__main__":
    env = make_env()

    # 4090 24G，稳健起步；OOM 再下调 num_envs 或 batch_size
    make_policy, params, metrics = ppo_train.train(
        environment=env,           # 传 env 实例（符合你这版源码）
        num_timesteps=600000,     # 总环境步
        wrap_env=True,             # 用 brax 的训练包装器
        madrona_backend=False,
        augment_pixels=False,

        # ===== 向量并行 & 时间长度（一步到位）=====
        num_envs=512,             # 可试 2048；需能被设备数整除（单 4090 就是 1）
        episode_length=1,          # 每 episode 仅 1 步（你的 env 也会 done=1）
        action_repeat=1,

        # ===== PPO 关键超参（按你这份源码的名字）=====
        learning_rate=3e-4,
        entropy_cost=2e-3,         # 单步任务多给点探索抑制塌缩
        discounting=0.0,           # 单步折扣无意义
        unroll_length=1,           # rollout 每次 1 步
        batch_size=8192,           # 一般取 num_envs 的 4–8 倍
        num_minibatches=32,
        num_updates_per_batch=4,   # ≈ 旧版的 num_update_epochs
        normalize_observations=True,
        reward_scaling=1.0,
        clipping_epsilon=0.2,      # 旧叫 clip_epsilon
        gae_lambda=1.0,            # 单步下无实际影响
        max_grad_norm=0.5,         # 旧叫 max_gradient_norm
        # vf_loss_coefficient=0.5,   # 旧叫 value_loss_coef

        # ===== 评估 & 日志（可选）=====
        num_evals=1,               # 训练期间评估次数；1 表示只返回最终 metrics
        deterministic_eval=False,  # 单步任务保持随机策略也 OK
        log_training_metrics=False,
        seed=0,
    )

    # —— 简单单步评测 —— #
    key = random.PRNGKey(42)
    policy = make_policy(params)   # 返回 (obs, key) -> (action, extras)

    state = env.reset(key)
    act, _ = policy(state.obs, key)
    state = env.step(state, act)

    print("eval reward:", float(state.reward))
