import os
import pandas as pd
import numpy as np
from torch import nn
import torch
import math
import os
import itertools
import sobol_seq
import numpy as np

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.degrees(np.array([x, y, z]))


def GetRotaTrans(torch_tensor):
    rotation_matrix = np.zeros((3, 3))
    translation_matirx = np.zeros((3, 1))
    if type(torch_tensor) == torch.Tensor:
        torch_tensor = torch_tensor[0].detach().numpy()
    rotation_matrix[0, 0] = torch_tensor[0]
    rotation_matrix[0, 1] = torch_tensor[1]
    rotation_matrix[0, 2] = torch_tensor[2]
    rotation_matrix[1, 0] = torch_tensor[3]
    rotation_matrix[1, 1] = torch_tensor[4]
    rotation_matrix[1, 2] = torch_tensor[5]
    rotation_matrix[2, 0] = torch_tensor[6]
    rotation_matrix[2, 1] = torch_tensor[7]
    rotation_matrix[2, 2] = torch_tensor[8]
    translation_matirx[0, 0] = torch_tensor[9]
    translation_matirx[1, 0] = torch_tensor[10]
    translation_matirx[2, 0] = torch_tensor[11]
    return translation_matirx, rotation_matrix


def ReadCSV(name, skip_rows):
    if skip_rows:
        job_list = pd.read_csv(os.getcwd() + "/" + name + ".csv",
                               on_bad_lines='error', encoding="gbk", dtype=float, skiprows=lambda x: x > 0 and x % 10 != 0)
    else:
        job_list = pd.read_csv(os.getcwd() + "/" + name + ".csv",header=0,
                               on_bad_lines='error', encoding="gbk", dtype=float)

    print(os.getcwd() + "/" + name + ".csv")
    # .iloc[:, 2:]
    return job_list


def IterProduct(repeat, total_chamber):
    permutations = list(itertools.product([-280,0, 200], repeat=repeat))
    res = [list(i) for i in permutations]
    array_product = np.array(res)
    append_array = np.zeros((array_product.shape[0], total_chamber-repeat))
    action_min = np.full((1, total_chamber), -280)
    action_max = np.full((1, total_chamber), 200)
    action_sum = np.hstack((append_array,array_product))
    action_sum = np.vstack((action_sum,action_max,action_min))
    return action_sum

def SobolSeq(dimensions,samples):
    lower_bound = -280
    upper_bound = 200
    scale = upper_bound - lower_bound

    # Generate Sobol sequence points
    sobol_points = sobol_seq.i4_sobol_generate(dimensions, samples)

    # Scale the Sobol sequence to the range [-280, 200]
    scaled_sobol_points = sobol_points * scale + lower_bound

    # Round the scaled Sobol points to the nearest integer and convert to int type
    integer_sobol_points = np.rint(scaled_sobol_points).astype(int)
    return integer_sobol_points
    # Print the integer Sobol points
    print("Integer Sobol Points:")
    print(integer_sobol_points)