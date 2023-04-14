import gymnasium as gym

from gymnasium_pomdps.envs.pomdp import POMDP


class MDP(gym.Wrapper):
    """Exposes the underlying MDP of a POMDP"""

    def __init__(self, env):
        if not isinstance(env.unwrapped, POMDP):
            raise TypeError(f"Env is not a POMDP (got {type(env)}).")

        super().__init__(env)
        self.observation_space = env.state_space

    def reset(self, seed=None, options=None):  # pylint: disable=arguments-differ
        observation, info = self.env.reset(seed=seed, options=options)
        info.update({"observation": observation})

        return self.state, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info.update({"observation": observation})

        return self.state, reward, terminated, truncated, info
