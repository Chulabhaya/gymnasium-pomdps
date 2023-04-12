import itertools as itt

import gymnasium as gym
import pytest

import gymnasium_pomdps as gym_pomdps


def test_pomdps_list_exists():
    pomdps = gym_pomdps.list_pomdps()
    assert len(pomdps) > 0


def test_pomdps_list_startswith():
    for pomdp in gym_pomdps.list_pomdps():
        assert pomdp.startswith('POMDP')


@pytest.mark.parametrize(
    'env_id',
    [
        'POMDP-shopping_2-continuing-v0',
        'POMDP-shopping_2-episodic-v0',
        'POMDP-arrowtrail-continuing-v0',
    ],
)
def test_functional(env_id: str):
    env = gym.make(env_id)

    for _ in range(20):
        s, o, _ = env.reset_functional()
        assert isinstance(s, int)
        assert isinstance(o, int)
        assert 0 <= s < env.state_space.n
        assert o == env.observation_space.n - 1

        for s, a in itt.product(range(env.state_space.n), range(env.action_space.n)):
            s1, o, r, terminated, truncated, info = env.step_functional(s, a)
            done = terminated or truncated
            if done:
                assert isinstance(s1, int)
                assert isinstance(o, int)
                assert isinstance(r, float)
                assert isinstance(done, bool)
                assert (s1, done) == (-1, True)
                assert 0 <= o < env.observation_space.n
                assert r in env.rewards_dict.keys()
                assert info is None or isinstance(info, dict)
            else:
                assert isinstance(s1, int)
                assert isinstance(o, int)
                assert isinstance(r, float)
                assert isinstance(done, bool)
                assert 0 <= s1 < env.state_space.n
                assert 0 <= o < env.observation_space.n
                assert r in env.rewards_dict.keys()
                assert info is None or isinstance(info, dict)

        for a in range(env.action_space.n):
            with pytest.raises(ValueError):
                env.step_functional(-1, a)

        s1, o, r, terminated, truncated, info = env.step_functional(-1, -1)
        done = terminated or truncated
        assert isinstance(s1, int)
        assert isinstance(o, int)
        assert isinstance(r, float)
        assert isinstance(done, bool)
        assert (s1, o, r, done) == (-1, -1, 0.0, True)
        assert info is None or isinstance(info, dict)


@pytest.mark.parametrize(
    'env_id',
    [
        'POMDP-shopping_2-continuing-v0',
        'POMDP-shopping_2-episodic-v0',
        'POMDP-arrowtrail-continuing-v0',
    ],
)
def test_run(env_id: str):
    env = gym.make(env_id)
    for _ in range(10):
        done = False
        env.reset()
        for _ in range(100):
            if done:
                a = -1
                o, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                assert isinstance(o, int)
                assert isinstance(r, float)
                assert isinstance(done, bool)
                assert (o, r, done) == (-1, 0.0, True)
                assert info is None or isinstance(info, dict)
            else:
                a = env.action_space.sample()
                o, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                assert isinstance(o, int)
                assert isinstance(r, float)
                assert isinstance(done, bool)
                assert 0 <= o < env.observation_space.n
                assert r in env.rewards_dict.keys()
                assert info is None or isinstance(info, dict)


@pytest.mark.parametrize('env_id', ['POMDP-tiger-continuing-v0'])
@pytest.mark.parametrize(
    'seed1,seed2',
    [
        (17, 17),
        (17, 18),
        (18, 18),
    ],
)
def test_reproducibility(env_id: str, seed1: int, seed2: int):
    env = gym.make(env_id)
    actions = list(range(env.action_space.n)) * 20

    outputs1 = [env.reset(seed=seed1)] + list(map(env.step, actions))

    outputs2 = [env.reset(seed=seed2)] + list(map(env.step, actions))

    result: bool = outputs1 == outputs2
    expected: bool = seed1 == seed2
    assert result == expected