"""Parser regression gate: fresh parses of golden PSDs must equal tests/goldens/ exactly (ability text byte for byte)."""
import json

import pytest

from conftest import blob_sha
from golden_cards import CARDS, CERTIFIED, golden_file, parse, parser


@pytest.fixture(scope="module")
def psd_parser(clone):
    return parser()


@pytest.mark.parametrize("rel", CARDS)
def test_golden(psd_parser, clone, rel):
    expected = json.loads(golden_file(rel).read_text(encoding="utf-8"))
    if not (clone / rel).exists() or blob_sha(clone / rel) != expected["blob"]:
        pytest.skip(f"{rel} changed upstream since the golden was made (re-pin: tests/make_goldens.py)")
    got = parse(psd_parser, clone, rel)
    assert got["card"] == expected["card"]
    assert got["alt_names"] == expected["alt_names"]
    assert got["validator"] == expected["validator"]


def test_certified_values():
    """The certified goldens (validation-and-qa section 3) as recorded there."""
    doomer, freezer, lich = (json.loads(golden_file(rel).read_text(encoding="utf-8")) for rel in CERTIFIED)
    d = doomer["card"]
    assert (d["type"], d["stars"], d["types"]) == ("minion", 2, ["wind", "defensive_minion", "light"])
    assert (d["hp"], d["def"], d["atk"], d["spd"]) == (5, 2, 5, 5)
    assert d["ability"].endswith("(BOT) Die.") and doomer["validator"] == []
    f = freezer["card"]
    assert (f["type"], f["stars"], f["types"]) == ("field", 1, ["field"])
    assert f["ability"] == ("For friendly creatures:\n[water] Turns into [ice]. [ice]: +1 DEF\n"
                            "(1PGT) Choose a friendly creature to turn into a [ice] and lower its speed by 1.")
    assert lich["card"]["types"] == ["ice", "undead"]
    assert lich["card"]["ability"].splitlines()[-1] == "Minions inherit [undead]."


def test_goldens_cover_every_folder():
    tops = {rel.split("/")[0] for rel in CARDS} | {"/".join(rel.split("/")[:2]) for rel in CARDS if rel.startswith("Auxiliary")}
    assert {"Creatures", "Items", "Field", "N.M.E", "Auxiliary/Minions", "Auxiliary/Items", "Auxiliary/Creatures",
            "Auxiliary/Status Effects", "Auxiliary/Rulebook"} <= tops
