
from src.game.ml.train import _get_nth_elements


def test_get_nth_elements():
    test = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8],
        [7, 8]
    ]

    assert _get_nth_elements(test, 0) == [1, 4, 7, 7]
    assert _get_nth_elements(test, 1) == [2, 5, 8, 8]
    assert _get_nth_elements(test, 2) == [3, 6]