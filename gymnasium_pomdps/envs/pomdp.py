from typing import Optional, Tuple, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from rl_parsers.pomdp import parse

from gymnasium_pomdps.rendering.renderer import Renderer
from gymnasium_pomdps.types import (
    Action,
    NoAction,
    NoObservation,
    NoState,
    Observation,
    State,
)

__all__ = ["POMDP"]


class POMDP(gym.Env):  # pylint: disable=abstract-method
    """Environment specified by POMDP file."""
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        text,
        *,
        episodic: bool,
        renderer: Optional[Renderer] = None,
        render_mode: Optional[str] = None,
    ):
        model = parse(text)
        self.episodic = episodic
        self.renderer = renderer

        if model.values == "cost":
            raise ValueError("Unsupported `cost` values.")

        self.model = model
        self.discount = model.discount
        self.state_space = spaces.Discrete(len(model.states))
        self.action_space = spaces.Discrete(len(model.actions))
        # plus unique new initial observation
        self.observation_space = spaces.Discrete(len(model.observations) + 1)
        self.reward_range = model.R.min(), model.R.max()

        self.rewards_dict = {r: i for i, r in enumerate(np.unique(model.R))}

        if model.start is None:
            self.start = np.full(self.state_space.n, 1 / self.state_space.n)
        else:
            self.start = model.start
        self.T = model.T.transpose(1, 0, 2).copy()
        if model.flags["O_includes_state"]:
            self.O = model.O.transpose(1, 0, 2, 3).copy()
        else:
            self.O = np.expand_dims(model.O, axis=0).repeat(self.state_space.n, axis=0)
        self.R = model.R.transpose(1, 0, 2, 3).copy()

        if episodic:
            self.D = model.reset.T.copy()  # only if episodic

        # NOTE currently elsewhere
        # self.TO = np.expand_dims(self.T, axis=-1) * self.O
        # self.EO = self.TO.sum(-2)
        # self.ER = (self.TO * self.R).sum((-2, -1))

        self.state: State = NoState
        self.observation: Observation = NoObservation

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    @property
    def _observation0(self) -> Observation:
        return self.observation_space.n - 1

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        self.state, self.observation, info = self.reset_functional()

        if self.render_mode == "human":
            self.render()

        return self.observation, info

    def step(
        self, action: Action
    ) -> Tuple[Observation, float, bool, bool, dict[str, Any]]:
        ret = self.step_functional(self.state, action)
        self.state, self.observation, reward, terminated, truncated, info = ret

        if self.render_mode == "human":
            self.render()

        return self.observation, reward, terminated, truncated, info

    def reset_functional(self) -> Tuple[State, Observation, dict[str, Any]]:
        state = self.np_random.multinomial(1, self.start).argmax().item()
        observation = self._observation0
        return state, observation, {}

    def step_functional(
        self, state: State, action: Action
    ) -> Tuple[State, Observation, float, bool, bool, dict[str, Any]]:
        if (state == NoState) != (action == NoAction):
            raise ValueError(f"Invalid state-action pair ({state}, {action}).")

        if state == NoState and action == NoAction:
            return NoState, NoObservation, 0.0, True, False, {}

        assert 0 <= state < self.state_space.n
        assert 0 <= action < self.action_space.n

        next_state = (
            self.np_random.multinomial(1, self.T[state, action]).argmax().item()
        )
        observation = (
            self.np_random.multinomial(1, self.O[state, action, next_state])
            .argmax()
            .item()
        )
        # NOTE below is the same but unified in single sampling op; requires TO
        # next_state, observation = divmod(
        #     self.np_random.multinomial(
        #         1, self.TO[state, action].ravel()).argmax().item(),
        #     self.observation.n,
        # )

        reward = self.R[state, action, next_state, observation].item()

        terminated = self.D[state, action].item() if self.episodic else False
        if terminated:
            next_state = NoState

        reward_cat = self.rewards_dict[reward]
        info = dict(reward_cat=reward_cat)

        return next_state, observation, reward, terminated, False, info

    def render(self):
        mode = self.render_mode

        if self.renderer is None:
            return super().render(mode)

        if mode == "human":
            return self.renderer.show(self.observation)
        elif mode == "rgb_array":
            return self.renderer.render(self.observation)
