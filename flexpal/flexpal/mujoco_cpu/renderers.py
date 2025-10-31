import time
import sys
from collections import deque
from typing import Optional
import numpy as np
import mujoco

try:
    import cv2
except Exception:
    cv2 = None


class MjRenderer:
    """
    轻量、可复用的 MuJoCo 渲染器：
    - human: 使用 mujoco.viewer (launch_passive)
    - rgb_array: 使用 mujoco.Renderer 离屏渲染，返回 np.uint8[H,W,3]
    - 支持轨迹绘制（轻量 deque）
    - 不主动 sys.exit；资源在 close() 里释放
    """
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        mode: Optional[str] = "human",       # "human" | "rgb_array" | None
        traj_maxlen: int = 10_000,
        width: int = 960,
        height: int = 720,
        camera: Optional[str] = None,        # 命名相机
    ):
        self.model = mj_model
        self.data = mj_data
        self.mode = mode
        self.width = int(width)
        self.height = int(height)
        self.camera = camera

        self.viewer = None                   # human
        self.offscreen = None                # rgb_array
        self.last_rgb = None

        self.traj = deque(maxlen=traj_maxlen)
        self.render_paused = False
        self.exit_flag = False

        if self.mode == "human":
            self._init_viewer()
        elif self.mode == "rgb_array":
            self._init_offscreen()

    # ---------- init ----------
    def _init_viewer(self):
        from mujoco import viewer

        def key_callback(keycode):
            # Space: pause / ESC: exit flag
            if keycode == 32:
                self.render_paused = not self.render_paused
            elif keycode == 256:
                self.exit_flag = True

        # 非阻塞窗口
        self.viewer = viewer.launch_passive(self.model, self.data, key_callback=key_callback)

    def _init_offscreen(self):
        # 离屏渲染器
        self.offscreen = mujoco.Renderer(self.model, self.width, self.height)

    # ---------- lifecycle ----------
    def update(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        """外部在 step 后调用一次，更新指针（避免频繁重建对象）"""
        self.model = mj_model
        self.data = mj_data

    def render_human(self):
        if self.viewer is None:
            self._init_viewer()
        # 同步窗口（注意：不要频繁重建 viewer）
        if self.viewer.is_running() and not self.exit_flag:
            # 需要改 opt 等用 with self.viewer.lock():
            self.viewer.sync()
        else:
            # 仅关闭 viewer，不退出进程
            self.close()

    def render_rgb_array(self):
        if self.offscreen is None:
            self._init_offscreen()
        self.offscreen.update_scene(self.data, camera=self.camera)
        self.last_rgb = self.offscreen.render()
        return self.last_rgb

    def render(self):
        if self.mode == "human":
            # paused 时也让窗口刷新（只是你可选择不推进物理）
            self.render_human()
            return None
        elif self.mode == "rgb_array":
            return self.render_rgb_array()
        return None

    # ---------- trajectory ----------
    def add_traj_point(self, pos: np.ndarray, every_n: int = 10):
        """可选：按一定频率添加轨迹点（pos: (3,) world 坐标）"""
        if len(self.traj) == 0 or (self.data.time % every_n) < 1e-6:
            self.traj.append(np.asarray(pos, dtype=float).copy())

    def draw_traj(self):
        """仅在 human 模式下绘制一些球点到 user_scn"""
        if self.mode != "human" or self.viewer is None:
            return
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = 0
            for i, point in enumerate(self.traj):
                if i >= len(self.viewer.user_scn.geoms):
                    break
                mujoco.mjv_initGeom(
                    self.viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.003, 0.003, 0.003]),
                    pos=point,
                    mat=np.eye(3).ravel(),
                    rgba=np.array([1.0, 0.0, 0.0, 1.0]),
                )
            self.viewer.user_scn.ngeom = min(len(self.traj), len(self.viewer.user_scn.geoms))

    # ---------- close ----------
    def close(self):
        try:
            if self.offscreen is not None:
                self.offscreen.close()
        except Exception:
            pass
        self.offscreen = None

        try:
            if self.viewer is not None:
                # 避免在关闭窗口后继续 sync
                self.exit_flag = True
                self.viewer.close()
        except Exception:
            pass
        self.viewer = None
