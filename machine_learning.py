import argparse
import datetime
import os

from src.game.ml.train import train
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a machine learning model on Traitor Roulette.')
    parser.add_argument('--total_timesteps', dest='total_timesteps',
                        default=131072, type=int,
                        help='set the number of trained time steps, default is 131072')
    parser.add_argument('--bankroll', dest='bankroll',
                        default=68000, type=int,
                        help='set your initial bankroll should be a multiple of 2000, default is 68000')
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    date = datetime.now().strftime("%Y%m%d_%H%M")

    output_path = os.path.join(
        dir_path, "output", f"ppo_trained_model_{date}.zip")

    train(output_path, initial_bankroll=args.bankroll,
          total_timesteps=args.total_timesteps)
