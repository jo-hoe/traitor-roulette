
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.common import generate_filepath
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


def play(bankroll: int, iterations_count: int, step_size: float):
    results = []

    game = TraitorRouletteGame(bankroll)
    run_count = 100/step_size
    for i in tqdm(range(1, int(run_count) + 1)):
        run = Run(round(i/run_count * 100, 6))

        for _ in range(iterations_count):
            if game.has_game_ended():
                run.add_final_bankroll(game.bankroll)
                game.reset()
                continue

            bet_size = game.get_valid_bet_size(run.bet_percentage)

            game.play(bet_size, random.choice(
                [PocketType.RED, PocketType.BLACK]))

        results.append([run.bet_percentage, run.avg(), run.min(), run.max()])

    # Convert results to numpy array
    return np.array(results)


def print_results(results: np.ndarray, default_bankroll):
    run_with_best_avg = results[np.argmax(results[:, 1])]

    file_path = generate_filepath("bruteforce.txt")
    # Writing results to a file
    with open(file_path, "w") as f:
        f.write(f"Bet percentage with best average: \
                {run_with_best_avg[0]}\n")
        f.write(f"Average final bankroll: \
                {run_with_best_avg[1]}\n")


def plot_results(results: np.ndarray):
    plt.figure(figsize=(12, 10))

    if results.ndim == 2 and results.shape[1] == 4:
        bet_percentages = results[:, 0]
        avg_bankrolls = results[:, 1]
        min_bankrolls = results[:, 2]
        max_bankrolls = results[:, 3]

        plt.plot(bet_percentages, avg_bankrolls, 'b-', label='Average (AU$)')
        plt.plot(bet_percentages, min_bankrolls, 'r-', label='Minimum (AU$)')
        plt.plot(bet_percentages, max_bankrolls, 'g-', label='Maximum (AU$)')
    else:
        print("Unexpected results structure. Please check the data.")

    plt.xlabel('Bet Percentage (%)')
    plt.ylabel('Australian Dollars (AU$) at the end of the game')
    plt.title('Bankroll vs Bet Percentage (%)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    x_ticks = np.arange(0, 101, 5)
    plt.xticks(x_ticks)

    plt.savefig(generate_filepath("bruteforce.png"), dpi=600)


if __name__ == "__main__":
    default_bankroll = 68000
    default_iterations_count = 1 << 17
    default_step_size = .01

    wheel = RouletteWheel()

    parser = argparse.ArgumentParser(
        description='Lets bruteforce Traitor Roulette.')
    parser.add_argument('--iterations_count', dest='iterations_count',
                        default=default_iterations_count, type=int,
                        help=f'set the number of iterations per valid betting size, default is {default_iterations_count}')
    parser.add_argument('--step_size', dest='step_size',
                        default=default_step_size, type=float,
                        help=f'step size for the betting percentage, default is {default_step_size}')
    parser.add_argument('--bankroll', dest='bankroll',
                        default=default_bankroll, type=int,
                        help=f'set your initial bankroll should be a multiple of 2000, default is {default_bankroll}')
    args = parser.parse_args()

    if args.bankroll % 2000 != 0:
        raise ValueError("Bankroll should be a multiple of 2000")
    if args.step_size < .000001:
        raise ValueError("Cannot run with step size less than .000001")

    results = play(args.bankroll, args.iterations_count, args.step_size)

    print_results(results, args.bankroll)
    plot_results(results)
