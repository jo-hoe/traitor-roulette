

import random
from src.game.pocket import PocketType
from src.game.traitor_roulette_game import MAX_ROUNDS, TraitorRouletteGame


def test_has_game_ended():
    initial_bankroll = 68000
    game = TraitorRouletteGame(initial_bankroll)
    assert game.has_game_ended() == False, "game should not have ended"

    game._round = 3
    assert game.has_game_ended() == False, "game should not have ended as round 3 is not over"

    game.reset()
    game._round = 4
    assert game.has_game_ended() == True, "game should not have ended as round >3"
    
    game.reset()
    game._bankroll = initial_bankroll * 3 + 1
    assert game.has_game_ended() == True, "game should have ended as bankroll is > initial_bankroll x 3"

    game.reset()
    game._bankroll = 0
    assert game.has_game_ended() == True, "game should have ended as bankroll is 0"

def test_play():
    initial_bankroll = 68000
    bet_size = 2000
    max_win_lose = bet_size * 2
    rounds = MAX_ROUNDS
    min_final_bankroll = initial_bankroll - (bet_size * rounds)
    # you can only win twice the amount on a traitor tile
    # as you your initial bet gets deducted from your bankroll
    # and is then added 3x
    max_final_bankroll = initial_bankroll + ((bet_size * 2) * rounds)
    game = TraitorRouletteGame(initial_bankroll)

    # play games
    for i in range(0, 124):
        # play three rounds
        for j in range(0, 3):
            winnings = game.play(bet_size, random.choice([PocketType.RED, PocketType.BLACK]))

            if winnings != 0 and (winnings % (bet_size * 2) != 0) and (winnings % (bet_size * 3) != 0):
                assert False, f"winnings was {winnings}, it should be between 0 or {bet_size * 2} or {bet_size * 3}"

            if j == 2:
                assert game.bankroll >= min_final_bankroll and game.bankroll <= max_final_bankroll, f"bankroll was {game.bankroll}, it should be between {min_final_bankroll} and {max_final_bankroll}"
                assert game.has_game_ended() == True, "game should have ended"
                game.reset()




