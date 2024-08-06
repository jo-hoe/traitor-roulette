

from typing import Tuple
from src.game.pocket import PocketType, Pocket
from src.game.roulette_wheel import RouletteWheel, TraitorRouletteWheel

MAX_ROUNDS = 3
MAX_MULTIPLIER = 3
BET_SIZE_INCREMENTS = 2000


class TraitorRouletteGame():

    def __init__(self, initial_bankroll: int, wheel: RouletteWheel = TraitorRouletteWheel()):
        self._initial_bankroll = initial_bankroll
        self._bankroll = initial_bankroll
        self._max_bankroll = initial_bankroll * MAX_MULTIPLIER
        self._round = 1
        self._wheel = wheel

    @property
    def bankroll(self):
        return self._bankroll

    @property
    def initial_bankroll(self):
        return self._initial_bankroll

    @property
    def current_round(self):
        return self._round

    @property
    def max_value(self):
        return self._max_bankroll

    def has_game_ended(self) -> bool:
        return self._bankroll == 0 or \
            self._bankroll >= self._max_bankroll or \
            self._round > MAX_ROUNDS

    def reset(self):
        self._bankroll = self._initial_bankroll
        self._round = 1

    def play(self, bet: int, prediction: PocketType) -> Tuple[Pocket, int]:
        '''
        Returns the winnings
        '''
        if bet > self._initial_bankroll:
            raise ValueError("Bet must be less than initial bankroll")
        if bet % 2000 != 0 and self.bankroll >= 2000:
            raise ValueError("Bet must be a multiple of 2000")
        if bet > self._bankroll:
            raise ValueError("Bet must be less than bankroll")
        if prediction not in [PocketType.RED, PocketType.BLACK]:
            raise ValueError("Prediction must be either black or red")
        if self.has_game_ended():
            raise ValueError("Game has ended")

        self._bankroll -= bet

        pocket = self._wheel.spin()

        winnings = 0
        if pocket.type == PocketType.TRAITOR:
            winnings = bet * 3
        elif pocket.type == prediction:
            winnings = bet * 2

        # cannot win more that 3x the initial bankroll
        if winnings + self.bankroll > self._max_bankroll:
            winnings = self._max_bankroll - self.bankroll

        self._bankroll += winnings
        self._round += 1

        return pocket, winnings

    def get_valid_bet_size(self, bet_percentage: float) -> int:
        """
        Implements constraints on betting size.
        Return a valid bet size based on the percentage of the bankroll to bet.
        """
        bet_size = self._bankroll * (bet_percentage / 100)
        bet_size = round(bet_size / 2000) * 2000

        # cannot bet 0
        if bet_size == 0:
            bet_size = 2000
        # cannot bet more than current bankroll
        if bet_size > self._bankroll:
            bet_size = self._bankroll
        # cannot bet more than initial bankroll
        if bet_size > self._initial_bankroll:
            bet_size = self._initial_bankroll

        return bet_size
