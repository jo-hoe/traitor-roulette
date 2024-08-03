import argparse
import os
import statistics
from typing import List
from stable_baselines3 import SAC

from src.game.ml.ml_environment import create_environment
from stable_baselines3.common.evaluation import evaluate_policy


def _flatten(matrix):
    return [x for xs in matrix for x in xs]


def _get_nth_elements(input: List[List[float]], n: int) -> List[float]:
    return [sublist[n] for sublist in input if n < len(sublist)]


def _get_moving_average(input: List[float], window_size: int) -> List[float]:
    return [statistics.mean(input[i:i+window_size]) for i in range(len(input)-window_size+1)]


def evaluate(model_path: str, initial_bankroll: int, num_episodes: int = 1000):
    model = SAC.load(model_path)

    total_rewards = []
    bet_sizes = [[] for _ in range(3)]  # List for each round
    final_bankrolls = []

    env = create_environment(initial_bankroll)

    for _ in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):  # New Gymnasium API
            obs = obs[0]
        done = False
        episode_reward = 0
        round_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)

            obs, reward, done, info = step_result

            episode_reward += reward

            if round_count < 3:
                # Assuming action[0] is the bet percentage
                bet_sizes[round_count].append(action[0])
            round_count += 1

        total_rewards.append(episode_reward)
        final_bankrolls.append(obs[0])  # Assuming obs[0] is the bankroll


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate a machine learning model of Traitor Roulette.')
    parser.add_argument('--model-name', dest='model_name',
                        type=str,
                        help='name of the model in output folder')
    parser.add_argument('--bankroll', dest='bankroll',
                        default=68000, type=int,
                        help='set your initial bankroll should be a multiple of 2000, default is 68000')
    parser.add_argument('--num-episodes', dest='num_episodes',
                        default=1<<10, type=int,
                        help='set the number of simulated episodes, default is 1024')
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(
        dir_path, "output", args.model_name)

    evaluate(model_path, args.bankroll, args.num_episodes)
