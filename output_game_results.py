from typing import Dict, List
from matplotlib import pyplot as plt
import numpy as np

from src.common import generate_filepath


def analyze_result_betsize(games: List[Dict]) -> None:
    bet_sizes = {1: [], 2: [], 3: []}

    # Iterate through all games and collect bet sizes per round
    for game in games:
        for round_data in game:
            round_num = round_data["round"]
            bet_size = round_data["bet_size"]
            if bet_size is not None:  # Make sure bet size is valid
                bet_sizes[round_num].append(bet_size)

    # Calculate average and standard deviation for each round
    averages = []
    std_devs = []
    for round_num in range(1, 4):
        if bet_sizes[round_num]:  # Check if there are any bets for this round
            averages.append(np.mean(bet_sizes[round_num]))
            std_devs.append(np.std(bet_sizes[round_num]))
        else:
            averages.append(0)  # No bets to average
            std_devs.append(0)  # No bets to calculate std deviation

    np_averages = np.array(averages)
    np_std_devs =np.array(std_devs)

    rounds = [1, 2, 3]
    x = np.arange(len(rounds))

    plt.bar(x, np_averages, yerr=np_std_devs, capsize=5, color='lightblue', edgecolor='black')
    plt.xticks(x, [f'Round {i}' for i in rounds])
    plt.xlabel('Game Rounds')
    plt.ylabel('Average Bet Size')
    plt.title('Average Bet Size per Round with Standard Deviation')
    plt.grid(axis='y')
    plt.savefig(generate_filepath("ai-betsize.png"), dpi=600)

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


def print_results(games: list, num_games: int, default_bankroll: int):
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