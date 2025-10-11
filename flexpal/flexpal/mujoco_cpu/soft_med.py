import os
from abc import ABC
import numpy as np
import mujoco

class SoftMed(ABC):
    """ Soft robot base class. """

    def __init__(self):
        xml_path = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml"
        self.robot_model = mujoco.MjModel.from_xml_path(
            filename=xml_path, assets=None)
        self.robot_data = mujoco.MjData(self.robot_model)
        self.actuator_index = [
                "L_axial_0" ,
                "L_axial_1" ,
                "L_axial_2" ,
                "LL_axial_0",
                "LL_axial_1",
                "LL_axial_2",
                "LLL_axial_0",
                "LLL_axial_1",
                "LLL_axial_2"]
        self.jnt_num = 9


