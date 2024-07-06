from src.game.pocket import Pocket, PocketType
from src.game.roulette_wheel import RouletteWheel

def test_generate_wheel():
    wheel = RouletteWheel()._wheel

    assert len(wheel) == 37, "Wheel should have 37 pockets" 
    
    assert wheel.count(Pocket(0, PocketType.GREEN)) == 1, "Wheel should have 1 green pocket"

    assert count_pocket_types(wheel, PocketType.GREEN) == 1, "Wheel should have 1 green pocket"
    assert count_pocket_types(wheel, PocketType.BLACK) == 12, "Wheel should have 12 black pockets"
    assert count_pocket_types(wheel, PocketType.RED) == 12, "Wheel should have 12 red pockets"
    assert count_pocket_types(wheel, PocketType.TRAITOR) == 12, "Wheel should have 12 green pockets"

    assert wheel[0] == Pocket(0, PocketType.GREEN), "Pocket 0 should be green"
    assert wheel[34] == Pocket(34, PocketType.BLACK), "Pocket 35 should be black"
    assert wheel[35] == Pocket(35, PocketType.RED), "Pocket 36 should be red"
    assert wheel[36] == Pocket(36, PocketType.TRAITOR), "Pocket 37 should be a traitor pocket"

def count_pocket_types(wheel : RouletteWheel, pocket_type : PocketType):
    result = 0
    for pocket in wheel:
        if pocket.type == pocket_type:
            result += 1
    return result


