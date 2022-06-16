from .carla_env import CarlaEnv
from gym.envs.registration import register

register(
    id='carla-v0',
    entry_point='gym_carla:CarlaEnv',
)

__all__ = ["CarlaEnv"]
