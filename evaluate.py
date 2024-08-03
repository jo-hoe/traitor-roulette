import argparse
import os
from typing import Dict, List
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from tqdm import tqdm

from src.common import generate_filepath
from src.game.ml.ml_environment import create_environment, reward_to_bankroll


def play(model_path: str, initial_bankroll: int, num_games: int) -> List[Dict]:
    model = SAC.load(model_path)

    games = []
    env = create_environment(initial_bankroll)
    for _ in tqdm(range(num_games)):
        max_bankroll = env.unwrapped.envs[0].unwrapped.game.max_value
        initial_bankroll = env.unwrapped.envs[0].unwrapped.game.initial_bankroll
        observation = env.reset()
        done = False
        game = []

        round = 0
        game.append({
            "round": round,
            "bankroll": initial_bankroll,
            "bet_size": None
        })

        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)

            # we need to calc the bankroll from the reward as data from
            # inner game state as such:
            # env.unwrapped.envs[0].unwrapped.game.<property name>
            # nor from the observations can be used
            # as they auto reset on once done becomes true, therefore not allowing
            # to pull the last state of the environment before done
            current_bankroll = reward_to_bankroll(
                reward[0], initial_bankroll, max_bankroll)

            round += 1
            game.append({
                "round": round,
                "bankroll": current_bankroll,
                "bet_size": action[0]
            })

        games.append(game)

    return games


def plot_all_games(games: list):
    # Visualization
    # Plotting the bankroll over rounds
    for game in games:
        rounds = [entry["round"] for entry in game]
        bankrolls = [entry["bankroll"] for entry in game]
        plt.plot(rounds, bankrolls)

    plt.title('Bankroll Over Rounds for Multiple Episodes')
    plt.xlabel('Round')
    plt.ylabel('Bankroll')
    plt.ylim(bottom=0)  # Assuming bankroll can't be negative
    plt.xticks(range(0, 4))  # Assuming max 3 rounds
    plt.xlim(0, 3)
    plt.grid()

    plt.savefig(generate_filepath("ai-behavior.png"), dpi=600)


def print_results(games: list, num_games: int):
    bust_counter = 0
    max_counter = 0
    final_bankrolls = [game[-1]["bankroll"] for game in games]

    bust_counter = sum(
        1 for final_bankroll in final_bankrolls if final_bankroll == 0)
    max_counter = sum(
        1 for final_bankroll in final_bankrolls if final_bankroll == default_bankroll * 3)
    average_final_bankroll = np.mean(final_bankrolls)

    bust_percentage = (bust_counter * 100) / num_games
    max_percentage = (max_counter * 100) / num_games
    file_path = generate_filepath("ai-behavior.txt")
    # Writing results to a file
    with open(file_path, "w") as f:
        f.write(f"{bust_percentage}% of games where going bust\n")
        f.write(f"{max_percentage}% of games achieved the maximum bankroll\n")
        f.write(f"Average final bankroll: {average_final_bankroll}\n")


if __name__ == "__main__":
    default_bankroll = 68000
    default_num_games = 1 << 16

    parser = argparse.ArgumentParser(
        description='Evaluate a machine learning model of Traitor Roulette.')
    parser.add_argument('--model-name', dest='model_name',
                        type=str,
                        help='name of the model in output folder')
    parser.add_argument('--bankroll', dest='bankroll',
                        default=default_bankroll, type=int,
                        help=f'set your initial bankroll should be a multiple of 2000, default is {default_bankroll}')
    parser.add_argument('--num-games', dest='num_games',
                        default=default_num_games, type=int,
                        help=f'set the number of simulated games, default is {default_num_games}')
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(
        dir_path, "output", args.model_name)

    games = play(model_path, args.bankroll, args.num_games)

    print_results(games, args.num_games)
    plot_all_games(games)
