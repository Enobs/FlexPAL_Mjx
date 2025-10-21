# core.py
# Minimal MJX core (JAX-ready). Batched + Scan + Micro-batch runner.

import os, time
from flax import struct
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import mujoco
from mujoco import mjx

from ids import Ids, build_ids


@struct.dataclass
class CoreParams:
    mjx_model: Any          # mjx.put_model(mj_model)
    model_dt: float         # e.g., mj_model.opt.timestep
    ctrl_dt: float          # 1.0 / control_freq
    substeps: int           # round(ctrl_dt / model_dt)
    ids: Ids = Ids()


@struct.dataclass
class CoreState:
    data: Any               # mjx.Data (batched或non-batched)
    t: jnp.ndarray          # int32 step counter（batched时为 [B]）


def core_build_params(mj_model: mujoco.MjModel, control_freq: float,
                      bodies=(), sites=(), tendons=(), joints=()) -> CoreParams:
    model_dt = float(mj_model.opt.timestep)
    ctrl_dt  = 1.0 / float(control_freq)
    substeps = max(1, int(round(ctrl_dt / model_dt)))
    return CoreParams(
        mjx_model=mjx.put_model(mj_model),
        model_dt=model_dt,
        ctrl_dt=ctrl_dt,
        substeps=substeps,
        ids=build_ids(mj_model, bodies=bodies, sites=sites, tendons=tendons, joints=joints),
    )


def core_reset(p: CoreParams,
               init_qpos: Optional[jnp.ndarray] = None,
               init_ctrl: Optional[jnp.ndarray] = None) -> CoreState:
    """Reset single-env MJX state (device-side only)."""
    d = mjx.make_data(p.mjx_model)
    if init_qpos is not None:
        d = d.replace(qpos=init_qpos)
    if init_ctrl is not None:
        d = d.replace(ctrl=init_ctrl)
    d = mjx.forward(p.mjx_model, d)
    return CoreState(data=d, t=jnp.array(0, dtype=jnp.int32))



def core_step(p: CoreParams, s: CoreState, ctrl: jnp.ndarray) -> CoreState:
    """Apply ctrl for one control period (ctrl_dt), integrating substeps."""
    d0 = s.data.replace(ctrl=ctrl)

    def substep(_, data):
        return mjx.step(p.mjx_model, data)

    dT = lax.fori_loop(0, p.substeps, substep, d0)
    return CoreState(data=dT, t=s.t + 1)

core_step_jit = jax.jit(core_step)


def mirror_to_cpu_for_view(mj_model: mujoco.MjModel,
                           mj_data: mujoco.MjData,
                           s: CoreState) -> None:
    """Copy MJX device data to CPU mujoco.MjData for rendering/debug."""
    mj_data.qpos[:] = jax.device_get(s.data.qpos)
    mj_data.qvel[:] = jax.device_get(s.data.qvel)
    if mj_model.nu > 0:
        mj_data.ctrl[:mj_model.nu] = jax.device_get(s.data.ctrl[:mj_model.nu])
    mujoco.mj_forward(mj_model, mj_data)


# =========================
# Batched reset / step
# =========================
def _tile_like(x, batch_size: int):
    x = jnp.asarray(x)
    return jnp.broadcast_to(x, (batch_size,) + x.shape)

def core_reset_batch(p: CoreParams,
                     batch_size: int,
                     key: Optional[jax.Array] = None,
                     qpos_noise_std: float = 0.0,
                     init_ctrl: Optional[jnp.ndarray] = None) -> CoreState:
    """复制单环境state为批量，并可对 qpos 加噪声。"""
    s0 = core_reset(p, init_qpos=None, init_ctrl=init_ctrl)  # 单环境
    data_b = jax.tree.map(lambda x: _tile_like(x, batch_size), s0.data)
    t_b = jnp.zeros((batch_size,), dtype=jnp.int32)

    if qpos_noise_std > 0.0:
        assert key is not None, "Provide PRNG key when qpos_noise_std>0"
        noise = qpos_noise_std * jax.random.normal(key, shape=data_b.qpos.shape)
        data_b = data_b.replace(qpos=data_b.qpos + noise)

    return CoreState(data=data_b, t=t_b)

# vmap 并行 step（每个 env 独立，但共享同一个 p）
core_step_batched = jax.jit(jax.vmap(core_step, in_axes=(None, 0, 0), out_axes=0))


# =========================
# Rollout over time (scan)
# =========================
def rollout_fixed_ctrl(p: CoreParams, s_b: CoreState, ctrl_b: jnp.ndarray, T: int) -> Tuple[CoreState, Any]:
    """固定控制信号的 batched rollout.
       s_b: batched CoreState
       ctrl_b: [B, nu]
       T: 时间步数（static 编译更快）
    """
    def body_fn(s_b, _):
        s_b = core_step_batched(p, s_b, ctrl_b)
        return s_b, s_b  

    return lax.scan(body_fn, s_b, xs=None, length=T)

# 可选：时间序列控制（[T, B, nu]）
def rollout_seq_ctrl(p: CoreParams, s_b: CoreState, ctrl_tb: jnp.ndarray, T: int) -> Tuple[CoreState, Any]:
    """时间序列控制信号的 batched rollout.
       ctrl_tb: [T, B, nu]
    """
    def body_fn(s_b, ctrl_b):
        s_b = core_step_batched(p, s_b, ctrl_b)
        return s_b, s_b
    return lax.scan(body_fn, s_b, xs=ctrl_tb, length=T)

# 单环境：时间维 scan + 单步 core_step
def rollout_seq_ctrl_single(p: CoreParams, s0: CoreState, ctrl_seq: jnp.ndarray, T: int):
    """ctrl_seq: [T, nu]  单环境"""
    def body_fn(s, ctrl):
        s_next = core_step(p, s, ctrl)
        return s_next, s_next
    return lax.scan(body_fn, s0, xs=ctrl_seq, length=T)

# 批量：时间维 scan + 空间 vmap(core_step)
def rollout_seq_ctrl_batched(p: CoreParams, s_b: CoreState, ctrl_tb: jnp.ndarray, T: int):
    """ctrl_tb: [T, B, nu]  批量环境"""
    def step_batched(s_b, ctrl_b):  # ctrl_b: [B, nu]
        s_b = jax.vmap(core_step, in_axes=(None, 0, 0))(p, s_b, ctrl_b)
        return s_b, s_b
    return lax.scan(step_batched, s_b, xs=ctrl_tb, length=T)

# jit（把 T 设为静态参数，加速编译）
rollout_seq_ctrl_single_jit  = jax.jit(rollout_seq_ctrl_single,  static_argnums=(3,))
rollout_seq_ctrl_batched_jit = jax.jit(rollout_seq_ctrl_batched, static_argnums=(3,))


def run_parallel(p: CoreParams,
                 num_envs: int = 1000,
                 horizon: int = 256,
                 micro_batch: int = 250,
                 seed: int = 0,
                 store_traj: bool = False):
    """
    显存友好的批量并行模拟 + 性能统计。
    打印: 每批耗时、平均步耗时、Physics Steps/s、推理效率。
    """
    import math
    from statistics import mean, stdev

    n_chunks = math.ceil(num_envs / micro_batch)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_chunks)

    # warmup (编译预热)
    bs0 = min(micro_batch, num_envs)
    sb_warm = core_reset_batch(p, batch_size=bs0, key=keys[0], qpos_noise_std=0.0)
    ctrl_b0 = jnp.zeros((bs0, p.mjx_model.nu), dtype=jnp.float32)
    _s_end_warm, _ = jax.jit(rollout_fixed_ctrl, static_argnums=(3,))(p, sb_warm, ctrl_b0, 2)
    _s_end_warm.data.qpos.block_until_ready()

    print(f"\n[Parallel Run] envs={num_envs} (micro={micro_batch}), horizon={horizon}, substeps={p.substeps}")
    print("Compiling complete. Starting timed simulation...\n")

    total_phys_steps = 0
    chunk_times = []

    _roll = jax.jit(rollout_fixed_ctrl, static_argnums=(3,))

    t0_global = time.perf_counter()

    for i in range(n_chunks):
        bs = micro_batch if (i < n_chunks - 1 or num_envs % micro_batch == 0) else (num_envs % micro_batch)
        sb = core_reset_batch(p, batch_size=bs, key=keys[i], qpos_noise_std=0.0)
        ctrl_b = jnp.zeros((bs, p.mjx_model.nu), dtype=jnp.float32)

        t0 = time.perf_counter()
        s_end, traj = _roll(p, sb, ctrl_b, horizon)
        s_end.data.qpos.block_until_ready()
        t1 = time.perf_counter()

        dur = t1 - t0
        chunk_times.append(dur)
        total_phys_steps += bs * horizon * p.substeps
        print(f"  Chunk {i+1}/{n_chunks}: B={bs:<4d}, time={dur:6.3f}s "
              f"({bs * horizon * p.substeps / dur:8.1f} steps/s)")

    t1_global = time.perf_counter()
    total_time = t1_global - t0_global
    total_steps = total_phys_steps
    steps_per_sec = total_steps / total_time
    mean_chunk = mean(chunk_times)
    std_chunk = stdev(chunk_times) if len(chunk_times) > 1 else 0.0
    per_env_step = (total_time / (num_envs * horizon)) * 1000  # ms/env-step

    print("\n=== Parallel Performance Report ===")
    print(f" Total Envs:        {num_envs}")
    print(f" Micro Batch Size:  {micro_batch}")
    print(f" Horizon (per env): {horizon}")
    print(f" Substeps:          {p.substeps}")
    print(f" Total Physics Steps: {total_steps:,}")
    print(f" Wall Time:         {total_time:8.3f} s")
    print(f" Physics Steps/s:   {steps_per_sec:,.1f}")
    print(f" Avg Chunk Time:    {mean_chunk:6.3f} s   ± {std_chunk:5.3f}")
    print(f" Avg Env-Step Time: {per_env_step:6.4f} ms  (wall time / env / horizon)\n")

    return {
        "num_envs": num_envs,
        "micro_batch": micro_batch,
        "horizon": horizon,
        "substeps": p.substeps,
        "total_time": total_time,
        "steps_per_sec": steps_per_sec,
        "avg_chunk_time": mean_chunk,
        "std_chunk_time": std_chunk,
        "ms_per_env_step": per_env_step,
    }


# =========================
# Main: quick tests
# =========================
if __name__ == '__main__':
    # （可选）在 shell 里先设置这些更稳：
    # export XLA_PYTHON_CLIENT_PREALLOCATE=false
    # export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
    # export XLA_PYTHON_CLIENT_ALLOCATOR=platform

    # 1) 加载模型
    xml_path = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml"  # 改成你的
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    # （可选）数值参数更稳一些
    mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    mj_model.opt.iterations = 30
    mj_model.opt.ls_iterations = 10

    # 2) 构建共享 MJX 参数（控制频率按需改）
    p = core_build_params(mj_model, control_freq=20, sites=("LLLend_effector",))
    print(f"Model loaded: nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}, dt={p.model_dt:.4f}, substeps={p.substeps}")

    # 3) 单环境 warmup
    s = core_reset(p)
    ctrl = jnp.zeros((mj_model.nu,), dtype=jnp.float32)
    s_warm = core_step_jit(p, s, ctrl)
    s_warm.data.qpos.block_until_ready()

    T = 300
    # s_warm -> s_b0（batch=1）
    s_b0 = core_reset_batch(p, batch_size=1, key=jax.random.PRNGKey(0), qpos_noise_std=0.0)
    # 用 s_warm 的 data 覆盖一下（可选）
    s_b0 = CoreState(
        data=jax.tree.map(lambda a, b: a.at[0].set(b), s_b0.data, s_warm.data),
        t=s_b0.t
    )
    # ctrl_tb: [T, 1, nu]
    ctrl_tb = jnp.zeros((T, 1, mj_model.nu), dtype=jnp.float32)

    s_end_b, _ = rollout_seq_ctrl_batched_jit(p, s_b0, ctrl_tb, T)
    # 取回单环境结果（batch 0）
    s_end = CoreState(
        data=jax.tree.map(lambda x: x[0], s_end_b.data),
        t=s_end_b.t[0]
    )
    s_end.data.qpos.block_until_ready()
    print("[Single via batch=1] Finished T=300.")


    # 5) 批量：先试小批，再跑显存友好的微批 1000 环境
    # 5.1 小批 64 环境 + 固定 ctrl
    B_small = 1
    s_b = core_reset_batch(p, batch_size=B_small, key=jax.random.PRNGKey(0), qpos_noise_std=0.0)
    ctrl_b = jnp.zeros((B_small, mj_model.nu), dtype=jnp.float32)
    roll_fixed = jax.jit(rollout_fixed_ctrl, static_argnums=(3,))
    s_end_b, _ = roll_fixed(p, s_b, ctrl_b, T)
    s_end_b.data.qpos.block_until_ready()
    print(f"[Batched] Finished B={B_small}, T={T}.")

    # 5.2 微批并行：例如 1000 环境、每批 250、horizon=256
    run_parallel(p, num_envs=2000, horizon=300, micro_batch=1, seed=42, store_traj=False)
