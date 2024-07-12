import argparse
import os

from src.game.ml.train import train
from src.game.ml.evaluate import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train and evaluate a machine learning model on Traitor Roulette.')
    parser.add_argument('--total_timesteps', dest='total_timesteps',
                        default=100000, type=int,
                        help='set the number of trained time steps, default is 100000')
    parser.add_argument('--bankroll', dest='bankroll',
                        default=68000, type=int,
                        help='set your initial bankroll should be a multiple of 2000, default is 68000')
    args = parser.parse_args()

    model_file_path = "ppo_trained_model.zip"
    if not os.path.isfile(model_file_path):
        train(model_file_path, initial_bankroll=args.bankroll, total_timesteps=args.total_timesteps)
    
    # evaluate(model_file_path, initial_bankroll=args.bankroll)