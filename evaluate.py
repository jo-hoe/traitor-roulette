import argparse
import os
from typing import Dict, List
from stable_baselines3 import SAC
from tqdm import tqdm

from output_game_results import analyze_result_betsize, plot_all_games, print_results
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
    
    analyze_result_betsize(games)

    print_results(games, args.num_games, default_bankroll)
    plot_all_games(games)
