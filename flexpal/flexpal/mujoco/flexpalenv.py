# env_flexpal.py
from typing import Any, Dict
from flax import struct
import jax, jax.numpy as jnp
from brax import base
from brax.envs.env import PipelineEnv, State as BraxState
from core import CoreParams, CoreState, build_core_params, core_reset, core_step, read_site_pos, read_actuator_ctrl
from control import (PIDParams, PIDState, SensorOuterParams, InnerActParams,
                     ControlParams, ControlState, control_pipeline_sensor_outer)

@struct.dataclass
class FlexParams:
    core: CoreParams
    ctrl: ControlParams

@struct.dataclass
class FlexState:
    pipeline_state: base.State   # 直接复用 brax pipeline 的 state
    core: CoreState
    ctrl: ControlState
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)

class FlexPALEnv(PipelineEnv):
    """
    物理后端仍由 PipelineEnv 负责（backend='mjx'），
    我们在 step() 内调用“传感空间外环 + 执行器内环”的控制管线，得到 ctrl_full，再推进物理。
    """
    def __init__(self, sys: base.System,
                 mj_model,                         # 需要原始 mujoco.MjModel 做 MJX core
                 actuator_names, site_names,
                 outer_pid=PIDParams(2.0, 0.0, 0.0),
                 inner_pid=PIDParams(8.0, 0.5, 0.1),
                 control_freq=25,
                 n_frames=1, backend='mjx', debug=False):
        super().__init__(sys=sys, backend=backend, n_frames=n_frames, debug=debug)
        core_p = build_core_params(mj_model, control_freq, actuator_names, site_names)
        ctrl_p = ControlParams(
            outer=SensorOuterParams(pid=outer_pid, tol=1e-3),
            inner=InnerActParams(pid=inner_pid)
        )
        self._p = FlexParams(core=core_p, ctrl=ctrl_p)

    @property
    def params(self) -> FlexParams:
        return self._p

    # ========== Reset ==========
    def reset(self, rng: jax.Array) -> BraxState:
        # Brax pipeline state（用于渲染/通用 API）
        q = jnp.zeros((self.sys.q_size(),))
        qd = jnp.zeros((self.sys.qd_size(),))
        ps = self.pipeline_init(q, qd)

        # 我们自己的 MJX core state
        core_s = core_reset(self._p.core)

        # 控制状态初始化
        N = self._p.core.actuator_index.shape[0]
        ctrl_s = ControlState(
            pid_outer=PIDState(jnp.zeros((N,)), jnp.zeros((N,))),
            pid_inner=PIDState(jnp.zeros((N,)), jnp.zeros((N,))),
        )

        obs = self._compute_obs(core_s)
        return BraxState(
            pipeline_state=ps,
            obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(0),
            metrics={},
            info=dict(core=core_s, ctrl=ctrl_s)  # 把自定义状态存进 info，保持 Brax State 兼容
        )

    # ========== Step ==========
    def step(self, state: BraxState, high_level_target: jax.Array) -> BraxState:
        core_s: CoreState = state.info['core']
        ctrl_s: ControlState = state.info['ctrl']

        # 读取传感空间当前值（比如 site 位置）
        sensor_cur = read_site_pos(core_s, self._p.core)           # [K,3]
        sensor_tgt = high_level_target.reshape(sensor_cur.shape)   # enforce shape

        # 执行器当前
        ctrl_full = read_actuator_ctrl(core_s)                     # [A]

        # 控制管线（外环→内环）
        ctrl_full_new, ctrl_s_new, reached, dbg = control_pipeline_sensor_outer(
            self._p.ctrl, ctrl_s, sensor_tgt, sensor_cur,
            ctrl_full, self._p.core.actuator_index, self.dt
        )

        # 推进 MJX core（我们自己的）
        core_s_new = core_step(self._p.core, core_s, ctrl_full_new)

        # 同时推进 Brax 的 pipeline（用于通用渲染/接口；传同样 ctrl）
        ps_new = self.pipeline_step(state.pipeline_state, ctrl_full_new)

        # 观测/奖励/终止（示例：到位给正奖）
        obs = self._compute_obs(core_s_new)
        reward = jnp.where(reached > 0, 1.0, -jnp.minimum(1.0, dbg['outer_err']))
        done = jnp.array(0)

        info = dict(state.info)
        info.update(core=core_s_new, ctrl=ctrl_s_new, reached=reached, **dbg)

        return BraxState(
            pipeline_state=ps_new,
            obs=obs,
            reward=reward,
            done=done,
            metrics={"outer_err": dbg['outer_err']},
            info=info
        )

    # ========== 观测构造 ==========
    def _compute_obs(self, core_s: CoreState) -> jax.Array:
        # 例：拼接 {site位置, qpos 的一部分, ctrl 的一部分}
        site = read_site_pos(core_s, self._p.core).reshape(-1)     # [K*3]
        # 也可以加上 qpos/qvel：core_s.data.qpos / qvel
        return site

    @property
    def observation_size(self):
        # 用 reset 推断（Brax 的惯例）
        obs = self.unwrapped.reset(jax.random.PRNGKey(0)).obs
        return obs.shape[-1]
