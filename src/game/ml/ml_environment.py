import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random

from src.game.pocket import PocketType
from src.game.traitor_roulette_game import TraitorRouletteGame
from stable_baselines3.common.env_util import make_vec_env


def bankroll_to_reward(bankroll: int, initial_bankroll: int, max_bankroll: int) -> float:
    reward = 0
    # reward winnings
    if bankroll > initial_bankroll:
        reward = bankroll / max_bankroll
    # punish losses
    elif bankroll < initial_bankroll:
        reward = -1 + (bankroll / initial_bankroll)

    return reward


def reward_to_bankroll(reward: float, initial_bankroll: int, max_bankroll: int) -> int:
    if reward == 0:
        bankroll = initial_bankroll
    # reward is positive (or zero), meaning bankroll >= initial_bankroll
    elif reward > 0:
        bankroll = reward * max_bankroll
    # reward is negative, meaning bankroll < initial_bankroll
    else:
        bankroll = (reward + 1) * initial_bankroll

    return round(bankroll)


class TraitorRouletteEnv(gym.Env):

    def __init__(self, initial_bankroll: int):
        super().__init__()
        self.game = TraitorRouletteGame(initial_bankroll)
        self.initial_bankroll = initial_bankroll
        self.max_bet_percentage = 0

        # Action space: bet_percentage (.01-1)
        self.action_space = spaces.Box(low=0.01, high=1.0, shape=())

        # Observation space: [current_round, bankroll]
        self.observation_space = spaces.Box(
            # Minimum values for [current_round, bankroll]
            low=np.array([1, 0]),
            # Maximum values for [current_round, bankroll]
            high=np.array([4, 1]),
            dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_obs(), {}

    def step(self, action):
        bet_percentage = action

        bet_size = self.game.get_valid_bet_size(bet_percentage * 100)
        color = random.choice([PocketType.RED, PocketType.BLACK])

        self.game.play(bet_size, color)

        reward = self._get_reward()
        done = self.game.has_game_ended()

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # Observation space: [current_round, bankroll]
        return np.array([
            self.game.current_round / 4,
            self.game.bankroll / self.game.max_value
        ], dtype=np.float32)

    def _get_reward(self) -> float:
        return bankroll_to_reward(self.game.bankroll, self.initial_bankroll,
                                  self.game.max_value)


def create_environment(initial_bankroll: int) -> TraitorRouletteEnv:
    return make_vec_env(lambda: TraitorRouletteEnv(initial_bankroll), n_envs=1)
