import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from src.game.ml.ml_environment import create_environment

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.losses = []
        self.std_devs = []
        self.final_bankrolls = []

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:  # Collect data every 1000 steps
            loss = self.model.logger.name_to_value['train/loss']
            self.losses.append(loss)
            
            # Evaluate std dev of rewards
            rewards = []
            for _ in range(100):  # Run 100 episodes to calculate std dev
                obs = self.training_env.reset()
                if isinstance(obs, tuple):  # New Gymnasium API
                    obs = obs[0]
                done = False
                episode_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    step_result = self.training_env.step(action)
                    if len(step_result) == 5:  # New Gymnasium API
                        obs, reward, terminated, truncated, _ = step_result
                        done = terminated or truncated
                    else:  # Old Gym API
                        obs, reward, done, _ = step_result
                    episode_reward += reward
                rewards.append(episode_reward)
            self.std_devs.append(np.std(rewards))
            self.final_bankrolls.append(obs[0])
        return True
    
class BetPercentageCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(BetPercentageCallback, self).__init__(verbose)
        self.bet_percentages = []

    def _on_step(self) -> bool:
        bet_percentage = self.training_env.actions[len(self.training_env.actions) - 1]
        #print(bet_percentage)
        self.bet_percentages.append(bet_percentage)
        if len(self.bet_percentages) % 1000 == 0:
            mean_bet = sum(self.bet_percentages[-1000:]) / 1000
            print(f"Average bet percentage over last 1000 steps: {mean_bet:.2f}")
        return True

def train(model_file_path: str, initial_bankroll: int, total_timesteps: int):
    env = create_environment(initial_bankroll)
    model = SAC(
        "MlpPolicy",
        env
    )

    model.learn(total_timesteps=500000, log_interval=1000, callback=BetPercentageCallback())
    model.save(model_file_path)# Evaluate the agent
    mean_bet, std_bet = evaluate_policy(model, model.get_env(), n_eval_episodes=100, return_episode_rewards=False)
    print(f"Mean bet percentage: {mean_bet:.2f} +/- {std_bet:.2f}")