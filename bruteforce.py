
import argparse
import csv
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.game.pocket import PocketType
from src.game.roulette_wheel import RouletteWheel
from src.game.traitor_roulette_game import TraitorRouletteGame

class Run():

    def __init__(self, percentage: float) -> None:
        self.bet_percentage = percentage
        self.final_bankrolls = []

    def avg(self):
        return sum(self.final_bankrolls) / len(self.final_bankrolls)

    def min(self):
        return min(self.final_bankrolls)

    def max(self):
        return max(self.final_bankrolls)

    def add_final_bankroll(self, bankroll: int):
        self.final_bankrolls.append(bankroll)


if __name__ == "__main__":
    wheel = RouletteWheel()

    parser = argparse.ArgumentParser(
        description='Lets bruteforce Traitor Roulette.')
    parser.add_argument('--iterations_count', dest='iterations_count',
                        default=10000, type=int,
                        help='set the number of iterations per valid betting size, default is 10000')
    parser.add_argument('--step_size', dest='step_size',
                        default=.01, type=float,
                        help='step size for the betting percentage, default is .01')
    parser.add_argument('--bankroll', dest='bankroll',
                        default=68000, type=int,
                        help='set your initial bankroll should be a multiple of 2000, default is 68000')
    args = parser.parse_args()

    if args.bankroll % 2000 != 0:
        raise ValueError("Bankroll should be a multiple of 2000")
    if args.step_size < .000001:
        raise ValueError("Cannot run with step size less than .000001")


    results = []

    game = TraitorRouletteGame(args.bankroll)
    run_count = 100/args.step_size
    for i in tqdm(range(1, int(run_count) + 1)):
        run = Run(round(i/run_count * 100, 6))

        for j in range(args.iterations_count):
            if game.has_game_ended():
                run.add_final_bankroll(game.bankroll)
                game.reset()
                continue

            bet_size = game.bankroll * (run.bet_percentage / 100)
            bet_size = round(bet_size / 2000) * 2000

            # cannot bet 0
            if bet_size == 0:
                bet_size = 2000
            # cannot bet more than bankroll
            if bet_size > game.bankroll:
                bet_size = game.bankroll
            # cannot bet more than initial bankroll
            if bet_size > args.bankroll:
                bet_size = args.bankroll

            game.play(bet_size, random.choice(
                [PocketType.RED, PocketType.BLACK]))

        results.append([run.bet_percentage, run.avg()])


    # Convert results to numpy array
    results = np.array(results)

    # Plotting
    plt.figure(figsize=(10, 6))
    if results.ndim == 2 and results.shape[1] == 2:
        plt.plot(results[:, 0], results[:, 1], 'b-')
    elif results.ndim == 1 and results.size % 2 == 0:
        x = results[0::2]  # Even indices for bet percentages
        y = results[1::2]  # Odd indices for average bankrolls
        plt.plot(x, y, 'b-')
    else:
        print("Unexpected results structure. Please check the data.")
        
    plt.xlabel('Bet Percentage')
    plt.ylabel('Average Final Bankroll')
    plt.title('Average Final Bankroll vs Bet Percentage')
    plt.grid(True)
    plt.show()

