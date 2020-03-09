from gym.envs.registration import register

register(
    id='RobotArm2D-v0',
    entry_point='robot_arm2.envs:RobotEnv',
)