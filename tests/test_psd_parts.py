"""
The parts editor's data (docs/Future/PARTS_EDITOR.md §11, Track A's 1-3): the variant tag rule, psd_parts.card_parts on
the five fixture PSDs (tests/fixtures/parts/, each read at its own blob: skipped when the clone lacks it) with the parser's
fields their joins, and a tagged A Snack parsed and drawn per picture.
"""
import json
import subprocess
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from psd_tools import PSDImage

from commands import psd_parts
from commands.psd_parser import layer_tag, shown_in, title_text, variant_view, variants_of
from conftest import no_lazy
from golden_cards import parser

FIXTURES = Path(__file__).parent / "fixtures" / "parts"
CARDS = {      # fixture: (path, card name, {lyid: name} tags applied in memory: §9's for A Snack)
    "mini_doomer": ("Auxiliary/Minions/Mini_Doomer.psd", "Mini Doomer", {}),
    "miracle_matter": ("Creatures/Kirby/3 Stars/Miracle_Matter.psd", "Miracle Matter", {}),
    "warcraftian_druid": ("Creatures/World Of Warcraft/4 Stars/Warcraftian_Druid.psd", "Warcraftian Druid", {}),
    "a_snack_tagged": ("Items/Heal/2 Stars/A_Snack.psd", "A Snack",
                       {1015: "A Snack #1", 327: "Name #1", 1022: "Breakfast #2", 1023: "Name #2", 1024: "Supper #3", 1025: "Name #3"}),
    "getting_started": ("Auxiliary/Rulebook/01_Getting_Started.psd", "01 Getting Started", {}),
}


@pytest.fixture(scope="module")
def psd_parser(clone):
    return parser()


def _fixture_psd(clone, fixture: str):
    """(fixture JSON, its PSD at the fixture's blob with the tags applied); skips when the clone lacks the blob."""
    want = json.loads((FIXTURES / f"{fixture}.json").read_text(encoding="utf-8"))
    done = subprocess.run(["git", "-C", str(clone), "cat-file", "blob", want["blob"]], capture_output=True, env=no_lazy())
    if done.returncode:
        pytest.skip(f"blob {want['blob'][:8]} not in the clone")
    psd = PSDImage.open(BytesIO(done.stdout))
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
    """Order, padding stripped, copies folded; the title parts join to the printed title, the ability parts (a rule
    page's blocks) to the parser's ability."""
    want, psd = _fixture_psd(clone, fixture)
    rel, name, tags = CARDS[fixture]
    card = psd_parser.parse(psd, rel, variant=1 if tags else None)
    got = psd_parts.card_parts(psd, rel, name, psd_parser, card=card, templates=psd_parts.Templates.from_dir(str(clone)))
    assert got == want["parts"]
    shown = [p for p in got["title"] if p["variants"] is None or 1 in p["variants"]]
    if shown:
        assert " ".join(title_text(p["text"]) for p in shown) == (card.display_name or card.name)
    ids = [p["id"] for p in got["blocks" if "Rulebook" in rel else "ability"] if p["variants"] is None or 1 in p["variants"]]
    layers = {layer.layer_id: layer for layer in psd.descendants()}
    with variant_view(psd, 1 if tags else None):
        assert psd_parser.join_ability(psd, [layers[i] for i in ids], rulepage="Rulebook" in rel) == card.ability


def test_a_snack_tagged_per_picture(clone, psd_parser):
    """Three pictures, each its own parse (its title, the shared ability) and render, within 0.1 of its PNG."""
    _, psd = _fixture_psd(clone, "a_snack_tagged")
    rel = CARDS["a_snack_tagged"][0]
    assert variants_of(psd) == [1, 2, 3]
    titles = {}
    for n in (1, 2, 3):
        png = clone / rel.replace(".psd", f"{'' if n == 1 else n}.png")
        if not png.exists():
            pytest.skip(f"{png.name} not in the clone")
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
