import gymnasium as gym

gym.register(
    id="Gym_softmujoco-v0",
    entry_point="flexpal.mujoco_cpu.flexpal_env_cpu:FlexPALSB3Env",
    kwargs={
        "xml_path": "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/model/pickandplace.xml",
        "goal_library_path": "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/lpos_samples_abs_actions_mp_60.npz",
        "goal_voxel":0.002, 
        "action_mode": "absolute",
        "render_mode": None,
    },
)