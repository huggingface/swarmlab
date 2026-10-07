"""M6 §1: the 28 real flags, triangle rasterisation, name-only candidates (docs/INTERFACE-M6.md)."""
import base64
import json

import pytest

from swarmlab.probes import BeliefProbe
from swarmlab.providers.base import ChatMessage
from swarmlab.rng import derive
from swarmlab.world.flaggame import (
    ALLOWED_PREFIX,
    BLIND_NOTE,
    CROP_HEADER,
    FlagGame,
    candidate_names,
    contains,
)
from swarmlab.world.flags_real import (
    BY_NAME,
    COUNTRY_NAMES,
    REAL_FLAGS,
    allowed_names,
    match_country,
    render_flag,
)
from swarmlab.world.render import decode_png_rgb, png_size, triangle_cells

AGENTS = ["a000", "a001", "a002", "a003"]


def real_world(**kw):
    params = {"flags": "real", "candidates": "names", "canvas": [24, 16], "crop": [6, 4], **kw}
    w = FlagGame(**params)
    w.reset(derive(3, "world"), list(AGENTS))
    return w


# ---- render.py ---------------------------------------------------------------------------------
def test_triangle_cells_by_centre():
    # right triangle with legs on the top and left edges of a 4x4 grid
    cells = triangle_cells(4, 4, ((0.0, 0.0), (4.0, 0.0), (0.0, 4.0)))
    # a centre (x+.5, y+.5) is inside iff x + y + 1 <= 4 (on the hypotenuse counts)
    assert cells == {(y, x) for y in range(4) for x in range(4) if x + y + 1 <= 4}
    # winding order does not matter
    assert cells == triangle_cells(4, 4, ((0.0, 4.0), (4.0, 0.0), (0.0, 0.0)))


# ---- the 28 flags ------------------------------------------------------------------------------
def test_twenty_eight_unique_countries():
    assert len(REAL_FLAGS) == 28 == len(set(COUNTRY_NAMES))
    for f in REAL_FLAGS:
        letters = {b for b, _ in f.bands} | ({f.triangle[0]} if f.triangle else set())
        assert letters <= set(f.colours), f.name  # every letter has an RGB
        assert all(len(c) == 3 and all(0 <= v <= 255 for v in c) for c in f.colours.values())


@pytest.mark.parametrize("size", [(24, 16), (12, 8)])
def test_all_flags_render_distinctly(size):
    w, h = size
    letter_grids, rgb_grids = set(), set()
    for f in REAL_FLAGS:
        g = render_flag(f, w, h)
        assert len(g) == h and all(len(r) == w for r in g)
        letter_grids.add(tuple(g))
        rgb_grids.add(tuple(tuple(f.colours[c] for c in row) for row in g))
    assert len(letter_grids) == 28 and len(rgb_grids) == 28


def test_layouts_and_proportions():
    assert BY_NAME["Germany"].layout_id == "h3"
    assert BY_NAME["France"].layout_id == "v3"
    assert BY_NAME["Cuba"].layout_id == "h5+tri"
    col = render_flag(BY_NAME["Colombia"], 24, 16)
    assert [r[0] for r in col] == list("y" * 8 + "b" * 4 + "r" * 4)  # 2:1:1
    cz = render_flag(BY_NAME["Czechia"], 24, 16)
    assert cz[7][10] == "b" and cz[7][11] == "w" and cz[0][0] == "b" and cz[0][1] == "w"
    assert cz[15][23] == "r" and cz[0][23] == "w"


def test_every_crop_position_is_contained_in_its_flag():
    w = real_world()
    for name in COUNTRY_NAMES:
        w.truth = name
        grid = w.candidates[name]
        for y in range(16 - 4 + 1):
            for x in range(24 - 6 + 1):
                w.crops["a000"] = (y, x)
                crop = w.crop_rows("a000")
                assert len(crop) == 4 and all(len(r) == 6 for r in crop)
                assert contains(grid, crop)
                assert crop == [r[x:x + 6] for r in grid[y:y + 4]]


def test_names_mode_observation_has_no_grids_and_lists_28_names():
    w = real_world(modality="image")
    obs = w.observe("a001")
    texts = [p.text for p in obs.parts if p.type == "text"]
    images = [p for p in obs.parts if p.type == "image"]
    assert len(images) == 1  # the crop only
    assert png_size(base64.b64decode(images[0].image_png_b64)) == (150, 100)  # cell_px 25
    assert texts[0].startswith(ALLOWED_PREFIX)
    names = json.loads(texts[0][len(ALLOWED_PREFIX):])
    assert names == sorted(COUNTRY_NAMES) and len(names) == 28
    assert texts[1] == CROP_HEADER
    assert not any(t.count("\n") > 1 for t in texts)  # no grid rows anywhere
    assert candidate_names("\n".join(texts)) == names
    assert set(obs.private) == {"crop_y", "crop_x"}


def test_crop_png_uses_the_flags_official_colours():
    w = real_world(modality="image")
    pix = decode_png_rgb(base64.b64decode(w.observe("a000").parts[2].image_png_b64))
    crop = w.crop_rows("a000")
    pal = BY_NAME[w.truth].colours
    for cy in range(4):
        for cx in range(6):
            assert pix[cy * 25 + 12][cx * 25 + 12] == pal[crop[cy][cx]]


def test_names_mode_text_modality_and_blind():
    w = real_world(blind_agents=1)
    text = w.observe("a001").parts[0].text
    lines = text.split("\n")
    assert lines[0].startswith(ALLOWED_PREFIX) and lines[2] == CROP_HEADER
    assert lines[3:] == w.crop_rows("a001")
    blind = w.observe("a000")
    assert blind.parts[0].text.endswith(BLIND_NOTE) and blind.private == {}
    assert "no crop" in w.description_for("a000")


def test_task_description_is_the_papers_framing():
    w = real_world()
    d = w.description()
    assert d.startswith("You are one player in a flag identification game. All players are "
                        "identifying the same underlying flag. You see a crop of it. Allowed countries: [")
    assert "cooperat" not in d.lower() and "truth" not in d.lower()
    assert allowed_names(ALLOWED_PREFIX + json.dumps(list(COUNTRY_NAMES))) == list(COUNTRY_NAMES)


def test_guess_matches_country_names_case_insensitively():
    w = real_world()
    assert w.guess("a001", "  ivory   COAST ").accepted
    assert w.guesses["a001"] == "Ivory Coast"
    assert not w.guess("a001", "Atlantis").accepted
    assert not w.guess("a001", "A").accepted
    assert match_country("czechia") == "Czechia" and match_country(3) is None


def test_verify_reports_country_and_layout():
    v = real_world().verify()
    assert v["country"] == v["truth"] in COUNTRY_NAMES
    assert v["layout"] == BY_NAME[v["truth"]].layout_id
    assert v["rival"] is None


def test_truth_is_a_uniform_draw_and_deterministic():
    truths = set()
    for seed in range(200):
        w = FlagGame(flags="real")
        w.reset(derive(seed, "world"), list(AGENTS))
        truths.add(w.truth)
    assert len(truths) > 20
    a, b = real_world(), real_world()
    assert (a.truth, a.crops) == (b.truth, b.crops)


def test_grids_mode_with_real_flags():
    w = FlagGame(flags="real", modality="image")
    w.reset(derive(1, "world"), list(AGENTS))
    obs = w.observe("a000")
    assert sum(p.type == "image" for p in obs.parts) == 29  # 28 candidates + the crop
    assert "28 candidate flags" in w.description()


def test_spec_unchanged_at_defaults_and_validation():
    plain = FlagGame().spec()
    assert FlagGame(flags="synthetic", candidates="grids", canvas=None, crop=None).spec() == plain
    assert FlagGame(cell_px=12).spec() == plain
    spec = FlagGame(flags="real", candidates="names", canvas=(24, 16), crop=(6, 4)).spec()["params"]
    assert spec["canvas"] == [24, 16] and spec["crop"] == [6, 4] and "cell_px" not in spec
    with pytest.raises(ValueError, match="names"):
        FlagGame(candidates="names")
    with pytest.raises(ValueError, match="canvas"):
        FlagGame(canvas=[24])
    with pytest.raises(ValueError, match="flags"):
        FlagGame(flags="fictional")


def test_belief_probe_reads_allowed_countries_and_country_key():
    w = real_world()
    ctx = [ChatMessage(role="user", content=w.observe("a001").parts)]
    probe = BeliefProbe()
    names = probe.candidates_from_context(ctx)
    assert names == sorted(COUNTRY_NAMES)
    ok, parsed = probe.parse('{"country": "germany", "reason": "black band"}', names)
    assert ok and parsed["candidate"] == "Germany"
