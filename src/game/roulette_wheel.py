import copy
import random
from typing import List
from src.game.pocket import Pocket, PocketType

class RouletteWheel():

    _wheel = []

    def __init__(self) -> None:
        self._wheel = self._generate_wheel()

    def spin(self) -> Pocket:
        choses_pocket = random.choice(self._wheel)
        # returning copy to avoid messing up the wheel by accident 
        return copy.deepcopy(choses_pocket)


    def _generate_wheel(self) -> List[Pocket]:
        result = []

        result.append(Pocket(0, PocketType.GREEN))

        for i in range(1, 37):
            if i % 3 == 0:
                result.append(Pocket(i, PocketType.TRAITOR))
            elif i % 2 == 0:
                result.append(Pocket(i, PocketType.BLACK))
            else:
                result.append(Pocket(i, PocketType.RED))

        return result
