import argparse
import os
import statistics
from typing import List
import pandas as pd
from stable_baselines3 import SAC
import matplotlib.pyplot as plt

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

    games = []
    env = create_environment(initial_bankroll)

    for _ in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):  # New Gymnasium API
            obs = obs[0]
        done = False
        game = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            inner_game_state = env.unwrapped.envs[0].unwrapped.game

            if len(game) == 0:
                has_won = inner_game_state.initial_bankroll < inner_game_state.bankroll
            else:
                has_won = game[-1]["bankroll"] < inner_game_state.bankroll

            game.append({
                    "bankroll": inner_game_state.bankroll,
                    "has_won": has_won,
                    "bet_size": action[0]
            })

        games.append(game)

    print(len(games))
    # Extract data for plotting
    rounds = []
    bankrolls = []
    win_status = []
    bet_sizes = []

    for game in games:
        for round_num, entry in enumerate(game):
            rounds.append(round_num)
            bankrolls.append(entry["bankroll"])
            win_status.append(entry["has_won"])
            bet_sizes.append(entry["bet_size"])

    # Convert data to DataFrame for easier plotting with Pandas
    data = pd.DataFrame({
        'round': rounds,
        'bankroll': bankrolls,
        'has_won': win_status,
        'bet_size': bet_sizes
    })

    # Plotting
    plt.figure(figsize=(14, 6))

    # Plot bankroll over rounds
    plt.subplot(1, 2, 1)
    plt.plot(data['round'], data['bankroll'], label='Bankroll', alpha=0.7)
    plt.title('Bankroll Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Bankroll')
    plt.grid()
    plt.legend()

    # Plot winning status
    plt.subplot(1, 2, 2)
    plt.plot(data['round'], data['has_won'], label='Win Status', alpha=0.7, marker='o', linestyle='None')
    plt.title('Winning Status Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Win Status (1=Won, 0=Lost)')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()



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
                        default=1 << 10, type=int,
                        help='set the number of simulated episodes, default is 1024')
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(
        dir_path, "output", args.model_name)

    evaluate(model_path, args.bankroll, args.num_episodes)
