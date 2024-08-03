import argparse
from collections import defaultdict
import os
import random
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from src.game.ml.ml_environment import create_environment, reward_to_bankroll


def evaluate(model_path: str, initial_bankroll: int, num_episodes: int = 1 << 13):
    model = SAC.load(model_path)

    games = []
    env = create_environment(initial_bankroll)
    for _ in range(num_episodes):
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

    print(len(games))


def plot_all_games(games: list):
    all_rounds = np.array([])
    all_bankrolls = np.array([])

    for game in games:
        rounds = np.array([x["round"] for x in game])
        bankrolls = np.array([x["bankroll"] for x in game])

        all_rounds = np.concatenate((all_rounds, rounds))
        all_bankrolls = np.concatenate((all_bankrolls, bankrolls))
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(all_rounds, all_bankrolls, marker='o', linestyle='-', color='b')
    plt.title("Bankroll Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Bankroll")
    plt.grid()
    plt.xticks(range(0, 4))  # Assuming max 3 rounds
    plt.xlim(0, 3)
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
