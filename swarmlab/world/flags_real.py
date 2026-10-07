"""The 28 real stripe-and-triangle national flags of the Flag Game paper (docs/INTERFACE-M6.md §1).

The paper (Pavlova & Tanaka, arXiv 2609.19124) draws the hidden flag from "the country pool of 28
stripe and triangle flags" rendered on a 24x16 canvas, but does not list the 28. This is our list:
well-known flags whose design is horizontal or vertical stripes, optionally with a triangle at the
hoist, with no canton, cross or emblem (or with the emblem dropped, see below), chosen so that no
two flags share a colour-letter layout and every flag renders distinctly on the 24x16 and the
default 12x8 canvas (tested in tests/test_flags_real.py).

Layout spec
-----------
Each `RealFlag` has `name` (the country name agents must answer with), `direction` (`"h"`
horizontal stripes top to bottom, `"v"` vertical stripes hoist to fly), `bands` (`(letter,
weight)` pairs; band edges are `floor(size * cumulative_weight / total + 0.5)`), an optional
`triangle` `(letter, apex)` (an isosceles triangle with its base on the whole hoist edge and its
apex at `apex * width` on the horizontal centre line, rasterised by cell centre, see
`render.triangle_cells`) and `colours` (letter -> official RGB approximation).

Letters are semantic colour names shared with the synthetic game where they exist (`r` red, `w`
white, `k` black, `g` green, `y` yellow/gold, `b` blue, `o` orange, `c` light blue/aquamarine), so
a text-mode grid is readable; the PNG uses the flag's own RGB (`palette`), so two flags may use the
same letter for slightly different shades (e.g. the blues of France and Russia).

Every flag is drawn on the canvas whatever its real aspect ratio (as in the paper: one 24x16
canvas for all), so proportions are stretched; band weights are the official ones.

Simplifications
---------------
- Peru: the civil flag (no coat of arms), which is the plain red-white-red vertical triband.
- Cuba: the white star in the red triangle is dropped.
- Philippines: the sun and the three stars in the white triangle are dropped.
- Bahamas, Czechia, Palestine and Sudan are drawn exactly (pure triangle-and-stripe designs).
- Triangle apexes: Czechia 1/2 of the length; Sudan and Palestine 1/3; Cuba, Philippines
  (equilateral on a 1:2 flag) 0.433; Bahamas 0.4 (approximation).
- Left out on purpose: Monaco (same layout as Indonesia), Chad (as Romania), Luxembourg (as the
  Netherlands up to a blue shade), Jordan (as Palestine once its star is dropped), and flags whose
  triangles or emblems do not survive a 24x16 grid (Guyana, Eritrea, East Timor, Vanuatu).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .render import fill_triangle

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class RealFlag:
    name: str
    direction: str  # "h" or "v"
    bands: tuple[tuple[str, int], ...]
    colours: dict[str, RGB] = field(hash=False, compare=False)
    triangle: tuple[str, float] | None = None

    @property
    def layout_id(self) -> str:
        """E.g. `h3` (three horizontal bands), `v3`, `h5+tri`."""
        return f"{self.direction}{len(self.bands)}" + ("+tri" if self.triangle else "")

    def render(self, width: int, height: int) -> list[str]:
        """The flag as `height` rows of `width` colour letters."""
        return render_flag(self, width, height)


W: RGB = (255, 255, 255)
K: RGB = (0, 0, 0)


def _f(name: str, direction: str, bands: str, colours: dict[str, RGB],
       weights: tuple[int, ...] | None = None, triangle: tuple[str, float] | None = None) -> RealFlag:
    ws = weights or (1,) * len(bands)
    return RealFlag(name, direction, tuple(zip(bands, ws, strict=True)), colours, triangle)


REAL_FLAGS: tuple[RealFlag, ...] = (
    # horizontal tribands and bicolours
    _f("Germany", "h", "kry", {"k": K, "r": (221, 0, 0), "y": (255, 206, 0)}),
    _f("Austria", "h", "rwr", {"r": (200, 16, 46), "w": W}),
    _f("Netherlands", "h", "rwb", {"r": (174, 28, 40), "w": W, "b": (33, 70, 139)}),
    _f("Russia", "h", "wbr", {"w": W, "b": (0, 57, 166), "r": (213, 43, 30)}),
    _f("Hungary", "h", "rwg", {"r": (205, 42, 62), "w": W, "g": (67, 111, 77)}),
    _f("Bulgaria", "h", "wgr", {"w": W, "g": (0, 150, 110), "r": (214, 38, 18)}),
    _f("Estonia", "h", "bkw", {"b": (0, 114, 206), "k": K, "w": W}),
    _f("Lithuania", "h", "ygr", {"y": (253, 185, 19), "g": (0, 106, 68), "r": (193, 39, 45)}),
    _f("Colombia", "h", "ybr", {"y": (252, 209, 22), "b": (0, 56, 147), "r": (206, 17, 38)},
       weights=(2, 1, 1)),
    _f("Yemen", "h", "rwk", {"r": (206, 17, 38), "w": W, "k": K}),
    _f("Ukraine", "h", "by", {"b": (0, 87, 183), "y": (255, 215, 0)}),
    _f("Poland", "h", "wr", {"w": W, "r": (220, 20, 60)}),
    _f("Indonesia", "h", "rw", {"r": (237, 28, 36), "w": W}),
    # vertical tribands
    _f("Italy", "v", "gwr", {"g": (0, 146, 70), "w": (241, 242, 241), "r": (206, 43, 55)}),
    _f("France", "v", "bwr", {"b": (0, 35, 149), "w": W, "r": (237, 41, 57)}),
    _f("Belgium", "v", "kyr", {"k": K, "y": (253, 218, 36), "r": (239, 51, 64)}),
    _f("Ireland", "v", "gwo", {"g": (22, 155, 98), "w": W, "o": (255, 136, 62)}),
    _f("Nigeria", "v", "gwg", {"g": (0, 135, 81), "w": W}),
    _f("Romania", "v", "byr", {"b": (0, 43, 127), "y": (252, 209, 22), "r": (206, 17, 38)}),
    _f("Mali", "v", "gyr", {"g": (20, 181, 58), "y": (252, 209, 22), "r": (206, 17, 38)}),
    _f("Ivory Coast", "v", "owg", {"o": (247, 127, 0), "w": W, "g": (0, 158, 96)}),
    _f("Peru", "v", "rwr", {"r": (217, 16, 35), "w": W}),
    # stripes plus a hoist triangle
    _f("Czechia", "h", "wr", {"w": W, "r": (215, 20, 26), "b": (17, 69, 126)}, triangle=("b", 0.5)),
    _f("Sudan", "h", "rwk", {"r": (210, 16, 52), "w": W, "k": K, "g": (0, 114, 41)},
       triangle=("g", 1 / 3)),
    _f("Palestine", "h", "kwg", {"k": K, "w": W, "g": (0, 150, 57), "r": (206, 17, 38)},
       triangle=("r", 1 / 3)),
    _f("Bahamas", "h", "cyc", {"c": (0, 119, 139), "y": (255, 199, 44), "k": K},
       triangle=("k", 0.4)),
    _f("Cuba", "h", "bwbwb", {"b": (0, 42, 143), "w": W, "r": (203, 21, 21)},
       triangle=("r", 0.433)),
    _f("Philippines", "h", "br", {"b": (0, 56, 168), "r": (206, 17, 38), "w": W},
       triangle=("w", 0.433)),
)

COUNTRY_NAMES: tuple[str, ...] = tuple(f.name for f in REAL_FLAGS)
BY_NAME: dict[str, RealFlag] = {f.name: f for f in REAL_FLAGS}


def _edges(size: int, weights: list[int]) -> list[int]:
    """Band start indices plus `size`: edge i = floor(size * cum_i / total + 0.5)."""
    total, cum, out = sum(weights), 0, [0]
    for w in weights:
        cum += w
        out.append(int(size * cum / total + 0.5))
    return out


def render_flag(flag: RealFlag, width: int, height: int) -> list[str]:
    """The flag on a `width x height` grid of colour letters (bands, then the triangle)."""
    letters = [b for b, _ in flag.bands]
    size = height if flag.direction == "h" else width
    edges = _edges(size, [w for _, w in flag.bands])
    band_of = [next(i for i in range(len(letters)) if edges[i] <= j < edges[i + 1])
               for j in range(size)]
    if flag.direction == "h":
        rows = [letters[band_of[y]] * width for y in range(height)]
    else:
        rows = ["".join(letters[band_of[x]] for x in range(width))] * height
    if flag.triangle is not None:
        letter, apex = flag.triangle
        rows = fill_triangle(rows, ((0.0, 0.0), (0.0, float(height)),
                                    (apex * width, height / 2)), letter)
    return rows


def flag_palette(flag: RealFlag) -> dict[str, RGB]:
    """Letter -> RGB for rendering this flag's grid (and crops of it) as a PNG."""
    return dict(flag.colours)


def normalise_name(name: str) -> str:
    """Case-insensitive, whitespace-collapsed form used to match an answer to a country."""
    return " ".join(str(name).split()).casefold()


def match_country(name: object, names: tuple[str, ...] | list[str] = COUNTRY_NAMES) -> str | None:
    """The allowed name `name` denotes (exact match after `normalise_name`), else None."""
    if not isinstance(name, str):
        return None
    key = normalise_name(name)
    return next((n for n in names if normalise_name(n) == key), None)


ALLOWED_PREFIX = "Allowed countries: "


def allowed_names(text: str) -> list[str]:
    """The names of an `Allowed countries: <JSON list>` line in `text` (first such line), else []."""
    import json

    for line in (text or "").split("\n"):
        line = line.strip()
        if line.startswith(ALLOWED_PREFIX):
            try:
                value = json.loads(line[len(ALLOWED_PREFIX):].rstrip("."))
            except ValueError:
                continue
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                return list(value)
    return []
