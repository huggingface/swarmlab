"""Dependency-free PNG rendering of colour grids (docs/INTERFACE-M5.md §2)."""
import json
import re
import struct
import zlib

import pytest

from swarmlab.colors import PALETTE_HEX, PALETTE_RGB, hex_to_rgb
from swarmlab.world.flaggame import COLOURS
from swarmlab.world.render import PALETTE, decode_png_rgb, grid_to_png, png_size

GRID = ["rrgg", "bbyy", "kwop"]


def test_png_structure_and_size():
    data = grid_to_png(GRID, PALETTE, 5)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert png_size(data) == (20, 15)
    # IHDR: 8-bit RGB, no interlace
    assert data[12:16] == b"IHDR"
    assert struct.unpack(">IIBBBBB", data[16:29]) == (20, 15, 8, 2, 0, 0, 0)
    # every chunk's CRC is right
    pos = 8
    kinds = []
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        (crc,) = struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])
        assert crc == zlib.crc32(kind + body) & 0xFFFFFFFF
        kinds.append(kind)
        pos += 12 + n
    assert kinds == [b"IHDR", b"IDAT", b"IEND"]


@pytest.mark.parametrize("cell_px", [1, 3, 12])
def test_decode_is_exact(cell_px):
    px = decode_png_rgb(grid_to_png(GRID, PALETTE, cell_px))
    assert len(px) == len(GRID) * cell_px and len(px[0]) == len(GRID[0]) * cell_px
    for y, row in enumerate(px):
        for x, rgb in enumerate(row):
            assert rgb == PALETTE[GRID[y // cell_px][x // cell_px]]


def test_deterministic_and_errors():
    assert grid_to_png(GRID, PALETTE, 4) == grid_to_png(list(GRID), dict(PALETTE), 4)
    assert grid_to_png(GRID, PALETTE, 4) != grid_to_png(GRID, PALETTE, 5)
    with pytest.raises(ValueError, match="palette"):
        grid_to_png(["rz"], PALETTE, 2)
    with pytest.raises(ValueError):
        grid_to_png([], PALETTE, 2)
    with pytest.raises(ValueError):
        grid_to_png(["rr", "r"], PALETTE, 2)
    with pytest.raises(ValueError):
        grid_to_png(["rr"], PALETTE, 0)


def test_decoder_handles_other_filters():
    # a 2x1 image written with filter 1 (sub) and filter 2 (up), as other encoders do
    rows = [b"\x01" + bytes([10, 20, 30, 5, 5, 5]), b"\x02" + bytes([1, 1, 1, 1, 1, 1])]
    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)

    def chunk(k, d):
        return struct.pack(">I", len(d)) + k + d + struct.pack(">I", zlib.crc32(k + d) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"".join(rows)))
           + chunk(b"IEND", b""))
    assert decode_png_rgb(png) == [[(10, 20, 30), (15, 25, 35)], [(11, 21, 31), (16, 26, 36)]]


def test_palette_is_one_source_shared_with_the_viewer():
    assert PALETTE is PALETTE_RGB
    assert set(PALETTE_HEX) == set(COLOURS) == set("rgbykwopcmnt")
    assert all(PALETTE_RGB[k] == hex_to_rgb(v) for k, v in PALETTE_HEX.items())
    assert len(set(PALETTE_RGB.values())) == len(PALETTE_RGB)  # colours are distinguishable
    from swarmlab.viewer.build import TEMPLATE
    assert "__COLOURS__" in TEMPLATE.read_text() and "#d62728" not in TEMPLATE.read_text()


def test_viewer_page_carries_the_palette(tmp_path):
    from swarmlab import Board, Experiment
    from swarmlab.participants import Silent
    from swarmlab.viewer.build import build
    from swarmlab.worlds import FlagGame

    run = Experiment(name="v", world=FlagGame(), participants=[Silent()] * 2,
                     medium=Board()).run(seed=1, max_rounds=1, out=tmp_path)
    html = build(run.dir).read_text()
    m = re.search(r"var COLOURS = (\{.*?\});", html)
    assert m and json.loads(m.group(1)) == PALETTE_HEX
