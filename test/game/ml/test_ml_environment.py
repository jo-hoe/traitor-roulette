
from src.game.ml.ml_environment import TraitorRouletteEnv, reward_to_bankroll


def test_reward():
    initial_bankroll = 68000
    env = TraitorRouletteEnv(initial_bankroll)

    # test positive values
    env.game._bankroll = initial_bankroll * 3
    assert env._get_reward() == 1, "reward should be 1"
    env.game._bankroll = env.game.initial_bankroll * 1.5
    assert env._get_reward() == .25, "reward should be .25"

    # test 0
    env.game._bankroll = initial_bankroll
    assert env._get_reward() == 0, "reward should be 0"

    # test negative values
    env.game._bankroll = initial_bankroll / 2
    assert env._get_reward() == -.5, "reward should be -.5"
    env.game._bankroll = 0
    assert env._get_reward() == -1, "reward should be -1"


def test_get_bankroll():
    initial_bankroll = 68000
    max_bankroll = initial_bankroll * 3

    # test positive values
    assert reward_to_bankroll(
        1, initial_bankroll, max_bankroll) == max_bankroll, "bankroll should be max_bankroll"

    # test == 0
    assert reward_to_bankroll(
        0, initial_bankroll, max_bankroll) == initial_bankroll, "bankroll should be initial_bankroll"

    # test negative values
    assert reward_to_bankroll(-.5, initial_bankroll, max_bankroll) == initial_bankroll / \
        2, "bankroll should be initial_bankroll / 2"
    assert reward_to_bankroll(-1, initial_bankroll,
                              max_bankroll) == 0, "bankroll should be 0"
