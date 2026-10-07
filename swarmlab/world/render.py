"""Dependency-free PNG rendering of colour-letter grids (docs/INTERFACE-M5.md §2).

`grid_to_png(rows, palette, cell_px)` draws every letter as a solid `cell_px x cell_px` square in
`palette[letter]` and encodes the picture as an 8-bit RGB, non-interlaced PNG with zlib + struct
only. `png_size(data) -> (w, h)` reads the IHDR; `image_label(png_b64)` is the text stand-in
`[image: PNG w×h]` that exports and prompt listings use for an image part; `decode_png_rgb(data)` returns the pixel rows as
lists of `(r, g, b)` tuples (used by tests; it accepts any 8-bit RGB non-interlaced PNG, all five
scanline filters).

Decisions where the contract is silent:

- Every scanline uses filter type 0 (none) and the stream is `zlib.compress(raw, 9)`, so the bytes
  are a pure function of the grid, the palette and `cell_px` (deterministic across runs and
  platforms using the same zlib).
- No ancillary chunks (no gamma, no text, no timestamp): IHDR, one IDAT, IEND.
- A letter missing from the palette raises `ValueError`; an empty grid or ragged rows too.
- `PALETTE` is `swarmlab.colors.PALETTE_RGB`, the palette the viewer uses.
"""
from __future__ import annotations

import struct
import zlib

from ..colors import PALETTE_RGB

PALETTE: dict[str, tuple[int, int, int]] = PALETTE_RGB
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

RGB = tuple[int, int, int]


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def grid_to_png(rows: list[str], palette: dict[str, RGB] | None = None, cell_px: int = 12) -> bytes:
    """PNG bytes of `rows` (strings of colour letters), each cell a `cell_px` square."""
    palette = PALETTE if palette is None else palette
    if not rows or not rows[0]:
        raise ValueError("cannot render an empty grid")
    if len({len(r) for r in rows}) != 1:
        raise ValueError("grid rows must all have the same length")
    if not isinstance(cell_px, int) or cell_px < 1:
        raise ValueError("cell_px must be an integer >= 1")
    w, h = len(rows[0]) * cell_px, len(rows) * cell_px
    raw = bytearray()
    for row in rows:
        try:
            line = b"".join(bytes(palette[c]) * cell_px for c in row)
        except KeyError as e:
            raise ValueError(f"colour letter {e.args[0]!r} is not in the palette") from None
        scan = b"\x00" + line
        raw += scan * cell_px
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + _chunk(b"IEND", b""))


def _chunks(data: bytes):
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG")
    pos = len(PNG_SIGNATURE)
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        yield kind, data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IEND":
            return


def png_size(data: bytes) -> tuple[int, int]:
    """(width, height) from the IHDR chunk."""
    for kind, body in _chunks(data):
        if kind == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
            return w, h
    raise ValueError("PNG has no IHDR chunk")


def image_label(png_b64: str | None) -> str:
    """Text stand-in for an image part: `[image: PNG <w>×<h>]` (exports, prompt listings)."""
    import base64
    import binascii

    try:
        w, h = png_size(base64.b64decode(png_b64 or "", validate=True))
    except (ValueError, struct.error, binascii.Error):
        return "[image: not a readable PNG]"
    return f"[image: PNG {w}×{h}]"


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def decode_png_rgb(data: bytes) -> list[list[RGB]]:
    """Pixel rows of an 8-bit RGB non-interlaced PNG."""
    ihdr = None
    idat = bytearray()
    for kind, body in _chunks(data):
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
    if ihdr is None:
        raise ValueError("PNG has no IHDR chunk")
    w, h, depth, ctype, _, _, interlace = ihdr
    if (depth, ctype, interlace) != (8, 2, 0):
        raise ValueError("only 8-bit RGB non-interlaced PNGs are supported")
    raw = zlib.decompress(bytes(idat))
    stride = 3 * w
    out: list[list[RGB]] = []
    prev = bytearray(stride)
    for y in range(h):
        ftype = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            a = line[i - 3] if i >= 3 else 0
            b = prev[i]
            c = prev[i - 3] if i >= 3 else 0
            if ftype == 1:
                line[i] = (line[i] + a) & 0xFF
            elif ftype == 2:
                line[i] = (line[i] + b) & 0xFF
            elif ftype == 3:
                line[i] = (line[i] + (a + b) // 2) & 0xFF
            elif ftype == 4:
                line[i] = (line[i] + _paeth(a, b, c)) & 0xFF
            elif ftype != 0:
                raise ValueError(f"unknown PNG filter type {ftype}")
        out.append([(line[i], line[i + 1], line[i + 2]) for i in range(0, stride, 3)])
        prev = line
    return out
