
from enum import Enum

class PocketType(Enum):
    GREEN = 0
    RED = 1
    BLACK = 2
    TRAITOR = 3

class Pocket():

    def __init__(self, number: int, type: PocketType) -> None:
        self._number = number
        self._type = type

    @property
    def number(self) -> int:
        return self._number

    @property
    def type(self) -> PocketType:
        return self._type
    
    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self._type == other.type and self.number == other.number
        return False
    
    def __hash__(self):
        return hash((self.number, self.type))
    
    def __repr__(self) -> str: 
        return f"Pocket({self.number}, {self.type.name})"


        