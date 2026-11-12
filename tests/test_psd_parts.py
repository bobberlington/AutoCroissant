"""
The parts editor's data (docs/Finished/session-2026-10-26/PARTS_EDITOR.md §11, Track A's 1-3): the variant tag rule, psd_parts.card_parts on
the four fixture PSDs (tests/fixtures/parts/, each read at its own blob) with the parser's
fields their joins, and a tagged A Snack parsed and drawn per picture.
"""
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from psd_tools import PSDImage

from commands import psd_parts
from commands.psd_parser import layer_tag, shown_in, title_text, variant_view, variants_of
from conftest import card_blob
from golden_cards import psd_parser  # noqa: F401 (fixture)

FIXTURES = Path(__file__).parent / "fixtures" / "parts"
sys.path.insert(0, str(FIXTURES))
from make_fixtures import CARDS  # noqa: E402 (fixture: (path, card name, {lyid: name} tags applied in memory))


def _fixture_psd(clone, fixture: str):
    """(fixture JSON, its PSD at the fixture's blob with the tags applied)."""
    want = json.loads((FIXTURES / f"{fixture}.json").read_text(encoding="utf-8"))
    psd = PSDImage.open(BytesIO(card_blob(clone, want["blob"])))
    for layer in psd.descendants():
        if layer.layer_id in CARDS[fixture][2]:
            layer.name = CARDS[fixture][2][layer.layer_id]
    return want, psd


def test_layer_tag():
    assert layer_tag("Name #2") == ("Name", frozenset({2}))
    assert layer_tag("Layer 6 #2,3") == ("Layer 6", frozenset({2, 3}))
    assert layer_tag("Hornet fight - atlas0 #10607 (1)") == ("Hornet fight - atlas0 #10607 (1)", None)
    assert layer_tag("Ability FX: Frozen In Time #1") == ("Ability FX: Frozen In Time #1", None)
    assert layer_tag("Name#2") == ("Name#2", None) and layer_tag("Name #2, 3") == ("Name #2, 3", None)


@pytest.mark.parametrize("fixture", list(CARDS))
def test_card_parts_equal_the_fixtures(clone, psd_parser, fixture):
    """Order, padding stripped, copies folded; the title parts join to the printed title."""
    want, psd = _fixture_psd(clone, fixture)
    rel, name, tags = CARDS[fixture]
    card = psd_parser.parse(psd, rel, variant=1 if tags else None)
    got = psd_parts.card_parts(psd, rel, name, psd_parser, card=card, templates=psd_parts.Templates.from_dir(str(clone)))
    assert got == want["parts"]
    shown = [p for p in got["title"] if p["variants"] is None or 1 in p["variants"]]
    if shown:
        assert " ".join(title_text(p["text"]) for p in shown) == (card.display_name or card.name)


def test_an_edited_split_title_keeps_its_parts(clone, psd_parser):
    """Warcraftian Druid's first box edited ('Warcraftiaa'): 'Druid ' still reads as a part, the title is the whole."""
    _, psd = _fixture_psd(clone, "warcraftian_druid")
    rel, name, _ = CARDS["warcraftian_druid"]
    next(layer for layer in psd.descendants() if layer.name == "Name").text = "Warcraftiaa \x03\r"
    got = psd_parts.card_parts(psd, rel, name, psd_parser)
    assert [title_text(p["text"]) for p in got["title"]] == ["Warcraftiaa", "Druid"]
    assert psd_parser.parse(psd, rel).display_name == "Warcraftiaa Druid"
    rel = "Creatures/Other/2 Stars/Auto_Croissant.psd"      # cut to 'Croissant': the epithet stays out
    psd = PSDImage.open(clone / rel)
    next(layer for layer in psd.descendants() if layer.name == "Name").text = "Croissant"
    assert [title_text(p["text"]) for p in psd_parts.card_parts(psd, rel, "Auto Croissant", psd_parser)["title"]] == ["Croissant"]
    assert psd_parser.parse(psd, rel).display_name == "Croissant"


def test_a_snack_tagged_per_picture(clone, psd_parser):
    """Three pictures, each its own parse (its title, the shared ability) and render, within 0.1 of its PNG."""
    _, psd = _fixture_psd(clone, "a_snack_tagged")
    rel = CARDS["a_snack_tagged"][0]
    assert variants_of(psd) == [1, 2, 3]
    titles = {}
    for n in (1, 2, 3):
        png = clone / rel.replace(".psd", f"{'' if n == 1 else n}.png")
        card = psd_parser.parse(psd, rel, variant=n)
        titles[n] = card.display_name
        assert card.ability == "Heal 3 HP.\nInflict 2T2 Regen and 2T1 Stuffed." and card.types == ["food"] and card.stars == 2
        with variant_view(psd, n):
            assert all(shown_in(layer, n) == layer.is_visible() for layer in psd.descendants())
            render = psd.composite(ignore_preview=True, apply_icc=True).convert("RGB")
        small = lambda image: np.asarray(image.resize((300, 450), Image.LANCZOS), dtype=np.float32)
        assert np.abs(small(render) - small(Image.open(png).convert("RGB"))).mean() < 0.1, n     # 0-255, at 300 px
    assert titles == {1: "A snack", 2: "Breakfast", 3: "Supper"}
    assert [layer.visible for layer in psd.descendants() if layer.layer_id in (1015, 1022, 1024)] == [True, False, False]


def test_card_parts_once_per_frozen_psd(monkeypatch):
    """A frozen PSD (the bake's read-only opens) runs psd_parts.card_parts once per (rel, name, parser), each call its
    own copy; any other PSD every time."""
    from commands import psd_edit as pe
    runs = []
    monkeypatch.setattr(psd_parts, "card_parts", lambda psd, rel, name, p: runs.append(rel) or {"title": [{"text": "A"}]})

    class Card:
        pass
    kept, plain, p = pe.frozen(Card()), Card(), object()
    pe.card_parts(kept, "a.psd", "A", p)["title"][0]["text"] = "B"
    pe.card_parts(kept, "a.psd", "A", p)["title"].clear()
    assert pe.card_parts(kept, "a.psd", "A", p) == {"title": [{"text": "A"}]}
    pe.card_parts(kept, "b.psd", "A", p)
    pe.card_parts(plain, "a.psd", "A", p)
    pe.card_parts(plain, "a.psd", "A", p)
    assert runs == ["a.psd", "b.psd", "a.psd", "a.psd"]
