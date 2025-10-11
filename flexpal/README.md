# 3D Robot Environment in Gym

This repository contains a custom Gym environment, `ThreeDimRobotEnv`, for a serial robot with 3 joints, each moving in a 3D space. The robot is controlled by altering the joint angles. 

This environment uses a Neural Network to perform the forward kinematics computation. The neural network and its optimizer's parameters are loaded from a previously saved state.

## Features

1. Custom Gym environment for a 3-joint robot in 3D.
2. Forward kinematics computation using a pre-trained Neural Network.
3. Rendering of the robot's state.

## Dependencies

- Gym
- Numpy
- Scipy
- Matplotlib
- Torch
- Sklearn
- Pandas

## How to Use

1. Initialize the environment with `env = ThreeDimRobotEnv()`.
2. You can interact with the environment using standard Gym functions:
   - `env.reset()`: Resets the environment and returns the initial observation/state.
   - `env.step(action)`: Takes an action as input, performs the forward kinematics to compute the new state of the robot, and returns the new observation/state, reward, done (whether the episode is finished), and info (additional information as a dictionary).
   - `env.render()`: Visualizes the current state of the robot in 3D.

For example, to make a step in the environment and then render the result:

```python
check = ThreeDimRobotEnv()
check_data = np.array([np.array([0.8838375, 0.3976425, 0.390211667,
                       0.3976425, 0.390211667, 0.8838375, 0.8838375, 0.3976425, 0.390211667])])
check.step(check_data)
check.render()
```

## Configuration

Configuration options for the `DataNet` neural network (used for the forward kinematics) are specified in the `Config.ini` file. These include:

- `save_path`: Path where the network's state has been saved.
- `lr`: Learning rate for the Adam optimizer.
- `beta1` and `beta2`: Parameters for the Adam optimizer.

## Note

The code assumes that the state of the `DataNet` neural network (i.e., model parameters and optimizer state) is saved at the path specified in the configuration file. Ensure this file is available before running the code.