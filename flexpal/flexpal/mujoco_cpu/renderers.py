import time

import mujoco
from collections import deque
import sys
import numpy as np
from queue import Queue
# TODO: separate a script
try:
    import cv2
except:
    print('Could not import cv2, please install it to enable camera viewer.')


class MjRenderer:
    def __init__(self, mj_model, mj_data, is_render, renderer, enable_camera_viewer=False, cam_mode='rgb'):
        self.mj_model = mj_model
        self.mj_data = mj_data

        self.renderer = renderer
        self.enable_camera_viewer = enable_camera_viewer
        self.cam_mode = cam_mode

        # Set up mujoco viewer
        self.viewer = None
        if is_render:
            self._init_renderer()

        self.traj = deque(maxlen=20000000)

        # keyboard flag
        self.render_paused = True
        self.exit_flag = False

        self._image = None
        self.image_queue = Queue()

        self.image_renderer = mujoco.Renderer(self.mj_model)

    def _init_renderer(self):
        """ Initialize renderer, choose official renderer with "viewer"(joined from version 2.3.3),
            another renderer with "mujoco_viewer"
        """

        def key_callback(keycode):
            if keycode == 32:
                self.render_paused = not self.render_paused
            elif keycode == 256:
                self.exit_flag = not self.exit_flag

        if self.renderer == "viewer":
            from mujoco import viewer
            # This function does not block, allowing user code to continue execution.
            self.viewer = viewer.launch_passive(
                self.mj_model, self.mj_data, key_callback=key_callback)

        else:
            raise ValueError('Invalid renderer name.')

    def render(self):
        """ render mujoco """
        if self.viewer is not None and self.render_paused is True and self.renderer == "viewer":
            if self.viewer.is_running() and self.exit_flag is False:
                self.viewer.sync()
            else:
                self.close()

    def close(self):
        """ close the environment. """
        if self.enable_camera_viewer:
            cv2.destroyAllWindows()
        self.viewer.close()
        sys.exit(0)

    def set_renderer_config(self):
        """ Setup mujoco global config while using viewer as renderer.
            It should be noted that the render thread need locked.
        """
        with self.viewer.lock():
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                self.mj_data.time % 2)

    def render_traj(self, pos, cur_time):
        """ Render the trajectory from deque above,
            you can push the cartesian position into this deque.

        :param pos: One of the cartesian position of the trajectory to render.
        """
        if self.renderer == "viewer":
            if cur_time % 10 == 0:
                self.traj.append(pos.copy())
            self.viewer.user_scn.ngeom = 0
            i = 0
            for point in self.traj:
                mujoco.mjv_initGeom(self.viewer.user_scn.geoms[i], type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array(
                    [0.003, 0.003, 0.003]), pos=point, mat=np.eye(3).flatten(), rgba=np.array([1, 0, 0, 1]))
                i += 1
            self.viewer.user_scn.ngeom = i
            self.viewer.sync()
