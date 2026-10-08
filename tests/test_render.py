import io

import numpy as np
import pytest

from terrain.compiler import compile_state
from terrain.materials import SURFACE_BLOCKS
from terrain.ops import apply_op
from terrain.state import WorldState


def test_colour_table_covers_every_palette_block():
    from terrain.render.l2 import block_colors

    table = block_colors()
    missing = (set(SURFACE_BLOCKS) | {"water", "stone", "dirt"}) - set(table)
    assert not missing
    assert table["grass_block"]["top"][1] > table["grass_block"]["top"][0]  # tinted green, not grey


@pytest.fixture
def small_volume():
    s = WorldState.flat(32, 32, base_y=64)
    s = apply_op(s, "raise_peak", {"center": [0.5, 0.5], "radius": 0.4, "height": 20})
    s.height[2:8, 2:8] = 60
    s.water[2:8, 2:8] = 63
    return s, compile_state(s)


def _image(png: bytes):
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def test_l2_renders_a_non_blank_image(small_volume):
    from terrain.render.l2 import render_volume

    _, vol = small_volume
    img = _image(render_volume(vol, "overview_ne", size_px=256))
    assert img.shape == (256, 256, 3)
    assert img.reshape(-1, 3).std(axis=0).max() > 20  # not a flat background
    green = (img[..., 1] > img[..., 0] + 15) & (img[..., 1] > img[..., 2])
    assert green.mean() > 0.02  # grass is visible


def test_l2_close_up_and_bad_view(small_volume):
    from terrain.opstack import BaseSpec, OpStack
    from terrain.render.l2 import render_volume

    stack = OpStack(BaseSpec(32, 32, 64))
    stack.add("raise_peak", {"center": [0.5, 0.5], "radius": 0.4, "height": 20}, id="hill")
    vol = compile_state(stack.state)
    assert len(render_volume(vol, feature=stack.state.features["hill"], size_px=128)) > 1000
    with pytest.raises(ValueError, match="unknown view"):
        render_volume(vol, "top_down")


def test_l1_relief_renders(small_volume):
    from terrain.render.l1 import render_l1

    s, _ = small_volume
    img = _image(render_l1(s, size_px=512))
    assert img.shape[0] == 512 and img.shape[1] == 512
