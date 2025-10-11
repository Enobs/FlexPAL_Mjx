import abc
import csv
import os
import pandas as pd

import mujoco
import numpy as np



class MujocoEnv:
    """ This environment is the base class.

    :param xml_path(str): Load xml file from xml_path to build the mujoco model.
    :param is_render(bool): Choose if use the renderer to render the scene or not.
    :param renderer(str): choose official renderer with "viewer",
    another renderer with "mujoco_viewer"
    :param control_freq(int): Upper-layer control frequency.
    Note that high frequency will cause high time-lag.
    :param enable_camera_viewer(bool): Use camera or not.
    :param cam_mode(str): Camera mode, "rgb" or "depth".
    """

    def __init__(self,
                 robot=None,
                 is_render=False,
                 renderer="viewer",
                 control_freq=10,
                 enable_camera_viewer=False,
                 cam_mode='rgb',
                 write_tendon=True,
                 tendon_num=9):
        self.write_tendon = write_tendon
        self.robot = robot
        self.is_render = is_render
        self.control_freq = control_freq
        self.tendon_num = tendon_num
        self.mj_model: mujoco.MjModel = self.robot.robot_model
        self.mj_data: mujoco.MjData = self.robot.robot_data
        self.cur_time = 0
        self.timestep = 0
        self.model_timestep = 0
        self.control_timestep = 0
        self.pose_reach = 0
        self.traj_state_array = []
        self.traj_state_array_tendon = []
        self.traj_state_array_bellow = []
        self.pid_last = 0
        self.file_name_traj = "logging_state"
        self.file_name_goal = "final_state"
        self.file_name_reset_data = "reset_data"
        self.file_name_tendon = ["Layer1_tendon","Layer2_tendon","Layer3_tendon"]
        self.file_name_bellow = ["Layer1_bellow","Layer2_bellow","Layer3_bellow"]
        self.file_name_tendon_traj = ["Layer1_tendon_traj","Layer2_tendon_traj","Layer3_tendon_traj"]
        self.file_name_bellow_traj = ["Layer1_bellow_traj","Layer2_bellow_traj","Layer3_bellow_traj"]
        self.last_point = np.array([0, 0, 0])
        self.tendon_index = [
            'layer1_left', 'layer1_middle', 'layer1_right', 'layer2_left', 'layer2_middle', 'layer2_right', 'layer3_left', 'layer3_middle', 'layer3_right']
        self.rota_const = np.array([[1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                                    [0.00000000e+00,  0.00000000e+00, 1.00000000e+00],
                                    [0.00000000e+00,  -1.00000000e+00,  0.00000000e+00]])
        if is_render:
            self.renderer = MjRenderer(self.mj_model, self.mj_data, self.is_render, renderer, enable_camera_viewer,
                                       cam_mode)
        else:
            self.renderer = None
        self.init_file_finder = 0
        self._initialize_time()
        file_path = "/home/yinan/Documents/SoftJointROS/reset_data.csv"
        if os.path.exists(file_path):
            self.df_reset = pd.read_csv(file_path)
            print("File content:\n", self.df_reset)
            self.init_file_finder = 1
        else:
            print(f"{file_path} did not exist and has been created as an empty file.")
        # self._set_init_pose()
        self.init_written()
        # if self.write_tendon == True:
        #     self._initialize_writing(self.file_name)

    def init_written(self):
        layers = []
        coordinate_labels = ["x", "y", "z"]  # Labels for coordinates
        
        # Iterate through the different positions
        positions = ["left", "middle", "right"]
        for j in range(len(self.file_name_tendon)):
            for position in positions:
                # For each position, generate strings with indices from 0 to 6
                for i in range(7):
                    for label in coordinate_labels:
                        layer_string = f"layer{j+1}_{i}_{position}_{label}"
                        layers.append(layer_string)
            self.csv_handle = open(self.file_name_tendon[j], 'a+', newline='')
            self.writer = csv.writer(self.csv_handle)
            self.writer.writerow(layers)
            self.csv_handle.close()
            self.csv_handle = open(self.file_name_tendon_traj[j], 'a+', newline='')
            self.writer = csv.writer(self.csv_handle)
            self.writer.writerow(layers)
            self.csv_handle.close()
            self.csv_handle = open(self.file_name_bellow_traj[j], 'a+', newline='')
            self.writer = csv.writer(self.csv_handle)
            self.writer.writerow(layers)
            self.csv_handle.close()
            self.csv_handle = open(self.file_name_bellow[j], 'a+', newline='')
            self.writer = csv.writer(self.csv_handle)
            self.writer.writerow(layers)
            self.csv_handle.close()
            layers = []
            items = [
            "tendon_1_left",
            "tendon_1_middle",
            "tendon_1_right",
            "tendon_2_left",
            "tendon_2_middle",
            "tendon_2_right",
            "tendon_3_left",
            "tendon_3_middle",
            "tendon_3_right",
            "actuation_1_left",
            "actuation_1_middle",
            "actuation_1_right",
            "actuation_2_left",
            "actuation_2_middle",
            "actuation_2_right",
            "actuation_3_left",
            "actuation_3_middle",
            "actuation_3_right",
            "R_0_1",
            "R_0_2",
            "R_0_3",
            "R_1_1",
            "R_1_2",
            "R_1_3",
            "R_2_1",
            "R_2_2",
            "R_2_3",
            "T1",
            "T2",
            "T3"
            ]
        self.csv_handle = open(self.file_name_goal, 'a+', newline='')
        self.writer = csv.writer(self.csv_handle)
        self.writer.writerow(items)
        self.csv_handle.close()
        name_joint= [
            "layer1_1_joint_x",
            "layer1_1_joint_y",
            "layer1_1_joint_z",
            "layer1_2_joint_x",
            "layer1_2_joint_y",
            "layer1_2_joint_z",
            "layer1_3_joint_x",
            "layer1_3_joint_y",
            "layer1_3_joint_z",
            "layer1_4_joint_x",
            "layer1_4_joint_y",
            "layer1_4_joint_z",
            "layer1_5_joint_x",
            "layer1_5_joint_y",
            "layer1_5_joint_z",
            "layer1_6_joint_x",
            "layer1_6_joint_y",
            "layer1_6_joint_z",
            "layer2_1_joint_x",
            "layer2_1_joint_y",
            "layer2_1_joint_z",
            "layer2_2_joint_x",
            "layer2_2_joint_y",
            "layer2_2_joint_z",
            "layer2_3_joint_x",
            "layer2_3_joint_y",
            "layer2_3_joint_z",
            "layer2_4_joint_x",
            "layer2_4_joint_y",
            "layer2_4_joint_z",
            "layer2_5_joint_x",
            "layer2_5_joint_y",
            "layer2_5_joint_z",
            "layer2_6_joint_x",
            "layer2_6_joint_y",
            "layer2_6_joint_z",
            "layer3_1_joint_x",
            "layer3_1_joint_y",
            "layer3_1_joint_z",
            "layer3_2_joint_x",
            "layer3_2_joint_y",
            "layer3_2_joint_z",
            "layer3_3_joint_x",
            "layer3_3_joint_y",
            "layer3_3_joint_z",
            "layer3_4_joint_x",
            "layer3_4_joint_y",
            "layer3_4_joint_z",
            "layer3_5_joint_x",
            "layer3_5_joint_y",
            "layer3_5_joint_z",
            "layer3_6_joint_x",
            "layer3_6_joint_y",
            "layer3_6_joint_z",
            'muscle_layer1_left',
            'muscle_layer1_middle',
            'muscle_layer1_right',
            'muscle_layer2_left',
            'muscle_layer2_middle',
            'muscle_layer2_right',
            'muscle_layer3_left',
            'muscle_layer3_middle',
            'muscle_layer3_right']
        self.csv_handle = open(self.file_name_reset_data, 'a+', newline='')
        self.writer = csv.writer(self.csv_handle)
        self.writer.writerow(name_joint)
        self.csv_handle.close()
        

        

    def step(self, action):
        """ 
        This method will be called with one-step in mujoco
        :param action: Input action
        :return: None
        """
        if self.renderer is not None and self.renderer.render_paused:
            self.cur_time += 1
            self.inner_step(action)

            mujoco.mj_forward(self.mj_model, self.mj_data)
            mujoco.mj_step(self.mj_model, self.mj_data)
            # point = self.get_site_pos('trajectory_point')

            # self.renderer.render_traj(point, self.cur_time)
            # self.last_point = point
        elif self.renderer is None:
            self.cur_time += 1
            self.inner_step(action)
            mujoco.mj_forward(self.mj_model, self.mj_data)
            mujoco.mj_step(self.mj_model, self.mj_data)

    @abc.abstractmethod
    def inner_step(self, action):
        """  This method will be called with one-step in mujoco, before mujoco step.
        For example, you can use this method to update the robot's joint position.

        :param action: input actions
        :return: None
        """
        raise NotImplementedError

    def reset(self,init_state):
        """ Reset the simulate environment, in order to execute next episode. """
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.reset_object()
        self._set_init_pose(init_state)
        mujoco.mj_forward(self.mj_model, self.mj_data)
    
    @abc.abstractmethod
    def get_current_state(self):
        raise NotImplementedError

    def PID(self, object, current):
        error_last = self.pid_last
        self.pid_last = object-current
        # This is for learning
        if np.abs(object-current) > 100:
            return 0.7*(object-current)+current
        elif np.abs(object-current) < 100 and np.abs(object-current) > 10:
            return 0.3*(object-current)+current
        elif np.abs(object-current) < 10 and np.abs(object-current) > 5:
            return 0.1*(object-current)+current
        else:
            return 0.05*(object-current)+current
        
    def reset_object(self):
        """ Set pose of the object. """
        pass

    def render(self, mode="human"):
        """ render mujoco """
        if self.is_render is True:
            self.renderer.render()

    def close(self):
        """ close the environment. """
        self.renderer.close()

    def _initialize_writing(self, file_name):
        pass

    def _initialize_time(self):
        """ Initializes the time constants used for simulation.
        :param control_freq (float): Hz rate to run control loop at within the simulation
        """
        self.cur_time = 0
        self.timestep = 0
        self.model_timestep = self.mj_model.opt.timestep
        if self.model_timestep <= 0:
            raise ValueError("Invalid simulation timestep defined!")
        if self.control_freq <= 0:
            raise ValueError(
                "Control frequency {} is invalid".format(self.control_freq))
        self.control_timestep = 1.0 / self.control_freq

    def _set_init_pose(self,init_state):
        """ Set or reset init joint position when called env reset func. """
        line_num = init_state
        if self.init_file_finder == 1:
            joint_pos = self.df_reset.iloc[line_num,:-9]
            actuator_pos = self.df_reset.iloc[line_num,54:]
            for column_name, value in joint_pos.items():
                self.set_object_pose(column_name, value)
            for column_name, value in actuator_pos.items():
                self.set_actuator_value(column_name,value)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def set_object_pose(self, obj_joint_name: str = None, obj_pose: np.ndarray = None):
        """ Set pose of the object. """
        if isinstance(obj_joint_name, str):
            # assert obj_pose.shape[0] == 7
            self.mj_data.joint(obj_joint_name).qpos = obj_pose
    
    def set_actuator_value(self, name, value: int=None):
        self.mj_data.actuator(name).ctrl = value
    
    def get_object_pose(self, obj_joint_name: str = None):
        return self.mj_data.joint(obj_joint_name).qpos

    def get_body_id(self, name: str):
        """ Get body id from body name.
        :param name: body name
        :return: body id
        """
        return mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)

    def get_actuator_value(self, name):
        return self.mj_data.actuator(name)

    def get_body_jacp(self, name):
        """ Query the position jacobian of a mujoco body using a name string.

        :param name: The name of a mujoco body
        :return: The jacp value of the mujoco body
        """
        bid = self.get_body_id(name)
        jacp = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacBody(self.mj_model, self.mj_data, jacp, None, bid)
        return jacp

    def get_body_jacr(self, name):
        """ Query the rotation jacobian of a mujoco body using a name string.

        :param name: The name of a mujoco body
        :return: The jacr value of the mujoco body
        """
        bid = self.get_body_id(name)
        jacr = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacBody(self.mj_model, self.mj_data, None, jacr, bid)
        return jacr

    def get_body_pos(self, name: str):
        """ Get body position from body name.

        :param name: body name
        :return: body position
        """
        return self.mj_data.body(name).xpos.copy()

    def get_body_quat(self, name: str):
        """ Get body quaternion from body name.

        :param name: body name
        :return: body quaternion
        """
        return self.mj_data.body(name).xquat.copy()

    def get_body_rotm(self, name: str):
        """ Get body rotation matrix from body name.

        :param name: body name
        :return: body rotation matrix
        """
        return self.mj_data.body(name).xmat.copy().reshape(3, 3)

    def get_body_xvelp(self, name: str) -> np.ndarray:
        """ Get body velocity from body name.

        :param name: body name
        :return: translational velocity of the body
        """
        jacp = self.get_body_jacp(name)
        xvelp = np.dot(jacp, self.mj_data.qvel)
        return xvelp.copy()

    def get_body_xvelr(self, name: str) -> np.ndarray:
        """ Get body rotational velocity from body name.

        :param name: body name
        :return: rotational velocity of the body
        """
        jacr = self.get_body_jacr(name)
        xvelr = np.dot(jacr, self.mj_data.qvel)
        return xvelr.copy()

    def get_site_id(self, name: str):
        """ Get site id from site name.

        :param name: site name
        :return: site id
        """
        return mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, name)

    def get_site_jacp(self, name):
        """ Query the position jacobian of a mujoco site using a name string.

        :param name: The name of a mujoco site
        :return: The jacp value of the mujoco site
        """
        sid = self.get_site_id(name)
        jacp = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacSite(self.mj_model, self.mj_data, jacp, None, sid)
        return jacp

    def get_site_pos(self, name: str):
        """ Get body position from site name.

        :param name: site name
        :return: site position
        """
        return self.mj_data.site(name).xpos.copy()

    def get_site_length(self, start_point: str, end_point: str):
        """ Get body length from site name.

        :param name: site name
        :return: site position
        """
        start_point_array = self.mj_data.site(start_point).xpos.copy()
        end_point_array = self.mj_data.site(end_point).xpos.copy()
        length = np.sqrt(np.power(start_point_array[0]-end_point_array[0], 2)+np.power(
            start_point_array[1]-end_point_array[1], 2)+np.power(start_point_array[2]-end_point_array[2], 2))
        return length

    def get_length(self, start_point_array, end_point_array):
        return np.sqrt(np.power(start_point_array[0]-end_point_array[0], 2)+np.power(
            start_point_array[1]-end_point_array[1], 2)+np.power(start_point_array[2]-end_point_array[2], 2))

    def get_tendon_length(self, name: str):
        """ Get body position from site name.

        :param name: site name
        :return: site position
        """
        return self.mj_data.tendon(name).length.copy()

    def get_site_xvelp(self, name: str) -> np.ndarray:
        """ Get site velocity from site name.

        :param name: site name
        :return: translational velocity of the site
        """
        jacp = self.get_site_jacp(name)
        xvelp = np.dot(jacp, self.mj_data.qvel)
        return xvelp.copy()