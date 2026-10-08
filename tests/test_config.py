from terrain.config import WORLD_MAX_Y, WORLD_MIN_Y


def test_world_limits_are_section_aligned():
    # Minecraft requires build limits on 16-block chunk-section boundaries.
    assert WORLD_MIN_Y < WORLD_MAX_Y
    assert WORLD_MIN_Y % 16 == 0
    assert WORLD_MAX_Y % 16 == 0
