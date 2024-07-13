import statistics
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from src.game.ml.ml_environment import create_environment
    
class BetPercentageCallback(BaseCallback):
    def __init__(self, output_interval:int, verbose=0):
        super(BetPercentageCallback, self).__init__(verbose)
        self.bet_percentages = []
        self.rewards = []
        self.final_bankrolls_rounds = [] # bankrolls, total number of rounds
        self.output_interval = output_interval

    def _on_step(self) -> bool:        
        game = self.training_env.envs[0].unwrapped.game

        round = game.current_round
        percentage = self.training_env.actions[len(self.training_env.actions) - 1]
        reward = self.training_env.buf_rews[0]

        # add an array for the next round
        # if the round is 2 this indicates that the first round is over
        if round == 2:
            self.bet_percentages.append([])
            self.rewards.append([])
        self.bet_percentages[len(self.bet_percentages)-1].append(percentage)
        self.rewards[len(self.rewards)-1].append(reward)

        # if game was reset in previous step, the final observations are kept in the buffer
        final_observations = next(self._get_final_observations_before_reset())
        if type(final_observations) is np.ndarray:
            self.final_bankrolls_rounds.append(
                # adding [total number of rounds, final bankroll]
                [final_observations[0] * 4 -1, final_observations[1] * game.max_value]
            )

        return super()._on_step()
    
    def _get_final_observations_before_reset(self):
        for env_idx in range(self.model.n_envs):
            # Get the terminal observation of the just finished trajectory
            yield self.locals["infos"][env_idx].get("terminal_observation")
    
def _flatten(matrix):
    return [x for xs in matrix for x in xs]
    
def _get_nth_elements(input:List[List[float]], n : int) -> List[float]:
    return [sublist[n] for sublist in input if n < len(sublist)]

def _get_moving_average(input:List[float], window_size: int) -> List[float]:
    return [statistics.mean(input[i:i+window_size]) for i in range(len(input)-window_size+1)]

def train(model_file_path: str, initial_bankroll: int, total_timesteps: int):
    env = create_environment(initial_bankroll)
    model = SAC(
        "MlpPolicy",
        env
    )

    callback = BetPercentageCallback(int(total_timesteps/100))
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    rewards = _flatten(callback.rewards)
    moving_avg_rewards = _get_moving_average(rewards, int(total_timesteps/100))
    game_number = [i for i in range(0, len(moving_avg_rewards))]

    model.save(model_file_path)
    mean_bet, std_bet = evaluate_policy(model, model.get_env(), n_eval_episodes=100, return_episode_rewards=False)
    print(f"Mean bet percentage: {mean_bet:.2f} +/- {std_bet:.2f}")

    plt.figure(figsize=(12, 10))
    plt.plot(game_number, moving_avg_rewards, 'b-', label='Moving average over rewards [-1, 1]')
    plt.xlabel('Steps')
    plt.ylabel('Reward [-1, 1]')
    plt.title('Reward over time steps')
    plt.show()