
from src.game.ml.ml_environment import TraitorRouletteEnv


def test_reward():
    initial_bankroll = 68000
    env = TraitorRouletteEnv(initial_bankroll)

    # test positive values
    env.game._bankroll = initial_bankroll * 3
    assert env._get_reward() == 1, "reward should be 1"
    env.game._bankroll = env.game.initial_bankroll * 1.5
    assert env._get_reward() == .5, "reward should be .5"

    # test 0
    env.game._bankroll = initial_bankroll
    assert env._get_reward() == 0, "reward should be 0"

    # test negative values
    env.game._bankroll = initial_bankroll / 2
    assert env._get_reward() == -.5, "reward should be -.5"
    env.game._bankroll = 0
    assert env._get_reward() == -1, "reward should be -1"