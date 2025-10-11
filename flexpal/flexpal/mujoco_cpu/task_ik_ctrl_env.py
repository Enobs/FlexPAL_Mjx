import numpy as np
from jnt_ctrl_env import JntCtrlEnv
import os
import configparser
import logging
import MatrixCov as MatrixCov
import time


class SpringCtrlEnv(JntCtrlEnv):
    def __init__(self,
                 robot=None,
                 is_render=True,
                 renderer="viewer",
                 control_freq=20,
                 write_tendon=True,
                 random_explor=False
                 ):
        super().__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq,
            write_tendon=write_tendon,
        )
        self.last_state = []
        self.sensor_reached = False
        self.p_cart = 1
        self.d_cart = 0.01
        self.p_quat = 0.2
        self.d_quat = 0.01
        self.vel_des = np.zeros(3)
        self.random_explor = random_explor
        self.random_range = range(-70, 20)
        if self.random_explor == True:
            self.action_explor = np.random.choice(self.random_range, size=9)

    def random_exploration(self):
        if (self.pose_reach == 1):
            self.pose_reach = 0
            cur_state = self.get_current_state()
            self.write_row(cur_state)
            self.action_explor = np.random.choice(self.random_range, size=9)
        self.step(self.action_explor)
        cur_state = self.get_current_state()
        if (len(self.traj_state_array) > 30):
            self.write_rows(self.traj_state_array)
            self.traj_state_array = []
        else:
            self.traj_state_array.append(cur_state)

        if self.is_render:
            self.render()

    def compute_pd_increment(self, p_goal: np.ndarray,
                             p_cur: np.ndarray,
                             pd_goal: np.ndarray = np.zeros(1),
                             pd_cur: np.ndarray = np.zeros(1)):
        pos_incre = self.p_cart * (p_goal - p_cur)
        return pos_incre

    def step_controller(self, action_sensor):
        action_pump = np.zeros(action_sensor.size)
        for i in range(0, self.robot.jnt_num):
            pos_incre = self.compute_pd_increment(
                action_sensor[i], self.get_tendon_length(self.tendon_index[i]).item())
            action_pump[i] = self.get_actuator_value(
                self.robot.actuator_index[i]).ctrl.item() + pos_incre
        return action_pump

    def sensor_based_control(self, action):
        self.step(action)
        cur_state = self.get_current_state()
        result = 0
        if self.last_state != []:
            for j in range(9):
                result += abs(self.last_state[j] - cur_state[j])
        # if result < 5e-7 and result != 0:
        if result < 1e-5 and result != 0:
            self.sensor_reached = True
        else:
            self.last_state = cur_state
        if (len(self.traj_state_array) > 30):
            self.traj_state_array = []
        else:
            self.traj_state_array.append(cur_state)
        if self.is_render:
            self.render()

    def step(self, action):
        if self.random_explor == False:
            action_pump = self.step_controller(action)
        else:
            action_pump = action
        super().step(action_pump)


if __name__ == "__main__":
    from soft_med import SoftMed

    config = configparser.ConfigParser()
    config.read("Config.ini", encoding="utf-8")

    env = SpringCtrlEnv(
        robot=SoftMed(),
        renderer='viewer',
        is_render=True,
        control_freq=20,
        write_tendon=True,
        random_explor=False
    )
    env.reset(0)
    last_state = []

    action = np.array([0.32173350183307176, 0.3061071000479501, 0.29778047217958026, 0.18338709963682318, 0.24410781817312252, 0.20115296141843292, 0.19013270451052566, 0.2324103725289386, 0.3231094563371839])
    start_time = time.time()
    out_time = 0
    while 1:
        out_time += 1
        env.step(action)
        cur_state = env.get_current_state()
        result = 0
        if last_state != []:
            for j in range(9):
                result += abs(action[j] - cur_state[j])
        if (result < 0.0005 and result != 0) or (out_time>300):
            print(out_time)
            out_time = 0
            time_after_first_code = time.time()
            duration_first_code = time_after_first_code - start_time
            print(
                f"Time taken by the first code block: {duration_first_code} seconds")
        if env.is_render:
            env.render()
