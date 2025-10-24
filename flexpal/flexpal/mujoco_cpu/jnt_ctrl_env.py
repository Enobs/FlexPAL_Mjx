import logging

import numpy as np
from base import MujocoEnv


class JntCtrlEnv(MujocoEnv):
    """ Single arm environment.

    :param robot(str): Robot configuration.
    :param is_render: Choose if use the renderer to render the scene or not.
    :param renderer: Choose official renderer with "viewer",
            another renderer with "mujoco_viewer"
    :param jnt_controller: Choose the joint controller.
    :param control_freq: Upper-layer control frequency. i.g. frame per second-fps
            Note that high frequency will cause high time-lag.
    :param is_interpolate: Use interpolator while stepping.
    :param enable_camera_viewer: Use camera or not.
    :param cam_mode: Camera mode, "rgb" or "depth".
    """

    def __init__(self,
                 robot=None,
                 is_render=False,
                 renderer="viewer",
                 control_freq=20,
                 write_tendon=True
                 ):

        super().__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq,
            write_tendon=write_tendon
        )

        self.nsubsteps = int(self.control_timestep / self.model_timestep)
        if self.nsubsteps == 0:
            raise ValueError("Control frequency is too low. Checkout you are not in renderer mode."
                             "Current Model-Timestep:{}".format(self.model_timestep))
            
    def get_bellow_state(self):
        cur_tendon_layer = []
        for j in range(3):
            for i in range(7):
                cur_tendon_layer.extend(self.get_site_pos("layer"+str(j+1)+"_"+str(i)+"_left").tolist())
            for i in range(7):
                cur_tendon_layer.extend(self.get_site_pos("layer"+str(j+1)+"_"+str(i)+"_middle").tolist())
            for i in range(7):
                cur_tendon_layer.extend(self.get_site_pos("layer"+str(j+1)+"_"+str(i)+"_right").tolist())
        return cur_tendon_layer
    
    def get_tendon_state(self):
        cur_tendon_layer = []
        for j in range(3):
            for i in range(7):
                cur_tendon_layer.extend(self.get_site_pos("spring_layer"+str(j+1)+"_"+str(i)+"_left").tolist())
            for i in range(7):
                cur_tendon_layer.extend(self.get_site_pos("spring_layer"+str(j+1)+"_"+str(i)+"_middle").tolist())
            for i in range(7):
                cur_tendon_layer.extend(self.get_site_pos("spring_layer"+str(j+1)+"_"+str(i)+"_right").tolist())
        return cur_tendon_layer
    
    def get_current_joint_act(self):
        cur_joint_act = []
        for j in range(3):
            for i in range(6):
                cur_joint_act.extend(self.get_object_pose("layer"+str(j+1)+"_"+str(i+1)+"_joint_x").tolist())
                cur_joint_act.extend(self.get_object_pose("layer"+str(j+1)+"_"+str(i+1)+"_joint_y").tolist())
                cur_joint_act.extend(self.get_object_pose("layer"+str(j+1)+"_"+str(i+1)+"_joint_z").tolist())
        for i in range(0, self.robot.jnt_num):
            cur_joint_act.append(self.get_actuator_value(
                self.robot.actuator_index[i]).ctrl.item())
        return cur_joint_act
    
    def get_current_state(self):
        cur_tendon = []
        for i in range(0, self.robot.jnt_num):
            cur_tendon.append(self.get_tendon_length(
                self.tendon_index[i]).item())
        for i in range(0, self.robot.jnt_num):
            cur_tendon.append(self.get_actuator_value(
                self.robot.actuator_index[i]).ctrl.item())
        body_pos = self.get_body_pos(
            'LLLmount')
        body_rota = self.get_body_rotm('LLLmount')
        # body_rota = np.dot(self.rota_const, body_rota)
        for i in range(3):
            for j in range(3):
                cur_tendon.append(body_rota[i][j])
        for i in range(3):
            cur_tendon.append(body_pos[i])
        return cur_tendon

    def inner_step(self, action: np.ndarray):
        # Send torque to simulation
        pose_reach = 0
        for i in range(self.robot.jnt_num):
            if np.abs(np.sum(action[i]-self.mj_data.actuator(
                    self.robot.actuator_index[i]).ctrl)) < 0.01:
                pose_reach += 1
            else:
                set_value = self.PID(action[i], self.mj_data.actuator(
                    self.robot.actuator_index[i]).ctrl)
                self.mj_data.actuator(
                    self.robot.actuator_index[i]).ctrl = min(250, max(-350, set_value))
        if pose_reach == self.robot.jnt_num:
            self.pose_reach = 1

    def gripper_ctrl(self, actuator_name: str = None, gripper_action: int = 1):
        """ Gripper control.

        :param actuator_name: Gripper actuator name.
        :param gripper_action: Gripper action, 0 for close, 1 for open.
        """
        self.mj_data.actuator(
            actuator_name).ctrl = 10 if gripper_action == 0 else 36

    def step(self, action: np.ndarray):
        # step into inner loop
        for i in range(self.nsubsteps):
            super().step(action)

    def reset(self,init_state):
        super().reset(init_state)
        
        
    def exp_to_sim(self,exp_value):
        exp_max = [4380,7956,6017,4020,4321,7136,10228,2743,3319]
        exp_min = [2543,5921,3666,2124,1625,4719,7754,  -34, 915]
        
        sim_min = [0.129689,0.129707,0.129637,0.100779,0.100849,0.100863,0.102487,0.102270,0.102375]
        sim_max = [0.198789,0.199084,0.198784,0.203662,0.203581,0.204481,0.199845,0.200172,0.200965]
        sim_values = []
        for i, n in enumerate(exp_value):
            if n < exp_min[i]:
                sim_values.append(sim_min[i])
            elif n > exp_max[i]:
                sim_values.append(sim_max[i])
            else:
                v = sim_min[i] + ((sim_max[i] - sim_min[i]) * (n - exp_min[i])) / (exp_max[i] - exp_min[i])
                sim_values.append(v)
        return sim_values

    def sim_to_exp(self,sim_value):
        exp_max = [4380,7956,6017,4020,4321,7136,10228,2992,3319]
        exp_min = [2543,5921,3666,2124,1625,4719,7754,  78, 915]
        
        sim_min = [0.129689,0.129707,0.129637,0.100779,0.100849,0.100863,0.102487,0.102270,0.102375]
        sim_max = [0.198789,0.199084,0.198784,0.203662,0.203581,0.204481,0.199845,0.200172,0.200965]
        
        exp_values = []
        for i, n in enumerate(sim_value):
            if n < sim_min[i]:
                exp_values.append(exp_min[i])
            elif n > sim_max[i]:
                exp_values.append(exp_max[i])
            else:
                v = exp_min[i] + ((exp_max[i] - exp_min[i]) * (n - sim_min[i])) / (sim_max[i] - sim_min[i])
                exp_values.append(v)
        return exp_values

if __name__ == "__main__":
    # scipy.stats.qmc.Sobol
    # 2^9 + 1500 sobol
    
    from soft_med import SoftMed

    env = JntCtrlEnv(
        robot=SoftMed(),
        renderer='viewer',
        is_render=True,
        control_freq=20,
    )
    env.reset(1)
    if env.is_render:
        env.render()
    action = np.array([0.2,0.2,-1,-1,-0.2,-1,-1,-0.2,0.2])
    stop_flag = True
    # env.gripper_ctrl("gripper_movement", 1)
    # while stop_flag:
    for i in range(400):
        check = env.get_current_state()
        if (env.pose_reach == 1):
            env.pose_reach = 0
        env.step(action)

        if env.is_render:
            env.render()
    print(env.get_site_pos("LLLend_effector"))