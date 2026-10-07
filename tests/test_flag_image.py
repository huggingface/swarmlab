"""FlagGame image modality (docs/INTERFACE-M5.md §1, §2, §5 items 1-2)."""
import base64
import hashlib

import pytest

from swarmlab import Board, Experiment
from swarmlab.participants import EvidenceAggregator, LLMAgent
from swarmlab.rng import derive
from swarmlab.world.flaggame import (
    CROP_CHANGED,
    CROP_HEADER,
    PREAMBLE,
    candidate_names,
    parse_observation,
)
from swarmlab.world.render import PALETTE, decode_png_rgb, png_size
from swarmlab.worlds import FlagGame

AGENTS = ["a000", "a001", "a002"]

# Computed on main (2c52456) before the image modality landed: text mode must not move.
MAIN_HASH_LLM = "1f8034f10b265f86e0c5b1f33389804524ebebcd18f2031f9953c3fa47ea21a8"
MAIN_HASH_EV = "a890a638b4899878144c347732b16b17c35f14f10d3a537cab0bdd3e2ea3944d"
MAIN_OBS_SHA = "52ca8ce0c89e0035bd0cb1b75ced232291bf2a60f88d628a341405d612e10811"
MAIN_DESC_SHA = "19d0c4d9ac0687c88b5d0b45a1c61983454c0bb61db6d5c10ec3176aae59b6ae"


def world(**kw):
    w = FlagGame(**kw)
    w.reset(derive(11, "world"), AGENTS)
    return w


def png(part):
    assert part.type == "image" and part.text is None
    return base64.b64decode(part.image_png_b64)


def grid_pixels(rows, cell):
    return [[PALETTE[rows[y // cell][x // cell]] for x in range(len(rows[0]) * cell)]
            for y in range(len(rows) * cell)]


# ---- acceptance 1: text mode unchanged ---------------------------------------------------------
def test_text_mode_spec_hash_and_observation_match_main():
    exp = Experiment(name="g", world=FlagGame(), participants=[LLMAgent(model="fake:reader")] * 4,
                     medium=Board(topology="broadcast"))
    assert exp.spec_hash(seed=3, max_rounds=4) == MAIN_HASH_LLM
    exp2 = Experiment(name="g", world=FlagGame(height=6, width=9, palette=8, guess_limit=3,
                                               crop_overrides={"a000": [0, 0]}),
                      participants=[EvidenceAggregator()] * 3, medium=Board())
    assert exp2.spec_hash(seed=7, max_rounds=2) == MAIN_HASH_EV
    w = world()
    txt = "".join(w.observe(a).parts[0].text for a in w.agents)
    assert all(len(w.observe(a).parts) == 1 for a in w.agents)
    assert hashlib.sha256(txt.encode()).hexdigest() == MAIN_OBS_SHA
    assert hashlib.sha256(w.description().encode()).hexdigest() == MAIN_DESC_SHA
    # explicit defaults are dropped from the spec as well
    assert "modality" not in FlagGame(modality="text", cell_px=12, image_text_hint=False).spec()["params"]


def test_image_params_enter_the_spec_only_when_set():
    assert FlagGame(modality="image").spec()["params"]["modality"] == "image"
    p = FlagGame(modality="image", cell_px=8, image_text_hint=True).spec()["params"]
    assert (p["cell_px"], p["image_text_hint"]) == (8, True)
    assert FlagGame(cell_px=8).spec()["params"]["cell_px"] == 8
    with pytest.raises(ValueError, match="modality"):
        FlagGame(modality="audio")
    with pytest.raises(ValueError, match="cell_px"):
        FlagGame(modality="image", cell_px=0)


# ---- acceptance 2: image observation -----------------------------------------------------------
@pytest.mark.parametrize("cell", [12, 5])
def test_image_observation_order_and_exact_pixels(cell):
    w = world(modality="image", cell_px=cell)
    obs = w.observe("a001")
    parts = obs.parts
    n = len(w.candidates)
    assert [p.type for p in parts] == ["text"] + ["image"] * n + ["text", "image"]
    intro = parts[0].text
    assert intro.startswith(PREAMBLE + "\n")
    assert "Candidates A, B, C, D, E, F, G, H are shown as images in that order" in intro
    assert "your crop follows" in intro and "no label" in intro
    assert parts[n + 1].text == CROP_HEADER
    for part, (name, grid) in zip(parts[1:n + 1], w.candidates.items(), strict=True):
        data = png(part)
        assert png_size(data) == (w.width * cell, w.height * cell)
        assert decode_png_rgb(data) == grid_pixels(grid, cell), name
    crop = png(parts[-1])
    assert png_size(crop) == (w.crop_w * cell, w.crop_h * cell)
    assert decode_png_rgb(crop) == grid_pixels(w.crop_rows("a001"), cell)
    # the crop image is the truth's image at the crop position
    y, x = w.crops["a001"]
    truth = decode_png_rgb(png(parts[1 + list(w.candidates).index(w.truth)]))
    sub = [row[x * cell:(x + w.crop_w) * cell] for row in truth[y * cell:(y + w.crop_h) * cell]]
    assert decode_png_rgb(crop) == sub
    assert obs.private == {"crop_y": y, "crop_x": x}
    # no grid text leaks without the hint
    texts = "\n".join(p.text for p in parts if p.type == "text")
    assert not any(row in texts for g in w.candidates.values() for row in g)
    assert candidate_names(texts) == list(w.candidates)
    assert "image" in w.description() and "grid of colour letters" not in w.description()


def test_image_rendering_is_deterministic_and_matches_text_mode_world():
    a, b = world(modality="image"), world(modality="image")
    t = world()
    assert a.observe("a000") == b.observe("a000")
    assert a.candidates == t.candidates and a.crops == t.crops and a.truth == t.truth


def test_text_hint_adds_grids_after_each_image():
    w = world(modality="image", image_text_hint=True)
    parts = w.observe("a002").parts
    n = len(w.candidates)
    assert [p.type for p in parts] == ["text"] + ["image", "text"] * n + ["text", "image", "text"]
    for i, (name, grid) in enumerate(w.candidates.items()):
        assert parts[2 + 2 * i].text == "\n".join([f"{name}:", *grid])
    assert parts[-2].type == "image" and parts[-1].text == "\n".join(w.crop_rows("a002"))
    text = "\n".join(p.text for p in parts if p.type == "text")
    assert parse_observation(text) == (w.candidates, w.crop_rows("a002"))
    assert "also written out as a grid" in w.description()


def test_crop_changed_note_in_image_mode():
    w = world(modality="image")
    w.patch_private("a000", {"crop": [0, 0]})
    parts = w.observe("a000").parts
    assert parts[-1].text == CROP_CHANGED and parts[-2].type == "image"
    assert decode_png_rgb(png(parts[-2])) == grid_pixels(w.crop_rows("a000"), 12)
    assert w.observe("a000").parts[-1].type == "image"  # said once


def test_snapshot_skips_png_cache():
    w = world(modality="image")
    w.observe("a000")
    assert w._png_cache
    w2 = FlagGame(modality="image")
    w2.restore(w.snapshot())
    assert not getattr(w2, "_png_cache", None)
    assert w2.observe("a000") == w.observe("a000")
