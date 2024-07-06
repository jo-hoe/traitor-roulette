
import argparse
from src.game.pocket import PocketType
from src.game.traitor_roulette_game import TraitorRouletteGame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play a game of Traitor Roulette.')
    parser.add_argument('--bankroll', dest='bankroll',
                        default=68000, type=int,
                        help='set your initial bankroll, default is 68000')
    args = parser.parse_args()

    game = TraitorRouletteGame(args.bankroll)

    while game.has_game_ended() == False:
        input_color = input('Choose an a color [r: red, b: black]: ')
        input_color = input_color.lower()
        if input_color != 'r' and input_color != 'b':
            print('Invalid color')
            continue
        
        input_bet = input(f'Choose your bet size [2000$ increments] your bankroll is {str(game.bankroll)}: ')
        if input_bet.isdigit() == False or int(input_bet) % 2000 != 0:
            print('Invalid bet')
            continue

        pocket, winnings = game.play(int(input_bet), PocketType.RED if input_color == 'r' else PocketType.BLACK)
        print(f"Ball landed in {str(pocket)} you get {winnings}$ back. Your new bankroll is {str(game.bankroll)}")
        

    print(f"The game has ended. You your bankroll is {game.bankroll}$")