import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from src.game.pocket import PocketType
from src.game.traitor_roulette_game import TraitorRouletteGame
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

def create_environment(initial_bankroll : int) -> VecEnv:
    # Create and wrap the environment
    env = TraitorRouletteEnv(initial_bankroll)
    return DummyVecEnv([lambda: env])

class TraitorRouletteEnv(gym.Env):

    def __init__(self, initial_bankroll : int):
        super().__init__()
        self.game = TraitorRouletteGame(initial_bankroll)
        self.initial_bankroll = initial_bankroll
        self.max_bet_percentage = 0
        
        # Action space: bet_percentage (1-100)
        self.action_space = spaces.Box(low=1.0, high=100.0, shape=())
        
        # Observation space: [current_bankroll, current_round]
        self.observation_space = spaces.Box(
            low=np.array([0, 1]),  # Minimum values for bankroll and round
            high=np.array([3*initial_bankroll, 3]),  # Maximum values
            dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_obs(), {}

    def step(self, action):
        bet_percentage = action
        print(bet_percentage)
        
        bet_size = TraitorRouletteGame.get_valid_bet_size(bet_percentage, self.game.initial_bankroll, self.game.bankroll)
        
        color = random.choice([PocketType.RED, PocketType.BLACK])
        
        self.game.play(bet_size, color)
        
        reward = self._get_reward()
        done = self.game.has_game_ended()
        
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return np.array([
            float(self.game.bankroll),
            float(self.game.current_round)
        ], dtype=np.float32)
    
    def _get_reward(self) -> float:
        reward = 0
        # reward winnings
        if self.game.bankroll > self.game.initial_bankroll:
            reward = self.game.bankroll / self.game.max_value
        # punish losses
        elif self.game.bankroll < self.game.initial_bankroll:
            reward = -1 + (self.game.bankroll / self.game.initial_bankroll)
        
        return reward