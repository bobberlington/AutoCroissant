"""discord_psds.fields_of and header_types.ncc / score on made-up reads and bands."""
import numpy as np

import commands.discord_psds as dp
import commands.header_types as ht


def test_a_posted_psd_s_fields():
    """Name, ability, types as the PSD has them; stars never; an item's subtypes never; an unread ability layer
    to review; stats only when every printed row is sure; an art PSD has none."""
    bars = {"stats": {k: {"v": 3, "sure": True, "row": "label"} for k in ("hp", "def", "atk", "spd")}}
    row = {"read": {"name": "Bolt", "kind": "creature", "types": ["fire"], "ability": "Zap it", "stars_drawn": 5},
           "bars": bars}
    f = dp.fields_of(row)
    assert set(f) == {"name", "ability", "types", "stats"} and f["stats"]["value"]["hp"] == 3
    item = dp.fields_of({**row, "read": {**row["read"], "kind": "item"}, "ability_from": "text layers",
                         "ability_recall": 0.5})
    assert "types" not in item and item["ability"]["confidence"] == "review"
    assert dp.fields_of({"read": {"name": "Frog", "kind": "creature", "types": [], "ability": ""}}) == {}


def test_a_missing_icon_lowers_the_header_match():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 255, (52, 60, 3)).astype(np.float32)
    assert ht.ncc(a, a) == 1.0
    b = a.copy()
    b[:26, :30] = 20.0                                   # one icon cell blank
    assert ht.ncc(a, b) < 0.5 < ht._ncc(a, b)


def test_a_text_layer_ability_must_print_in_order():
    """Sure only when the composite prints its words in that order: labels moved to the end (Twin Emperors) or an
    overflowed last word (Old Chica) go to review; OCR run-togethers ('caatxopt') are its words."""
    lines = ["TWIN", "Health", "Vek'nilash: can only be hit by", "spells. Vek'lor: can only be hit by", "attacks.",
             "Royalty"]
    moved = dp.printed_in_order("can only be hit by spells.\ncan only be hit by attacks.\nVek'nilash:\nVek'lor:", lines)
    assert moved["unprinted"] and not dp.ability_sure({"ability_from": "t", "ability_check": moved})
    kept = dp.printed_in_order("Vek'nilash: can only be hit by spells. Vek'lor: can only be hit by attacks.", lines)
    assert kept == {"order": 1.0, "unprinted": [], "inner": []}
    over = dp.printed_in_order("Gain 2 ATK for that attack.\nattack", ["CHICA", "Gain 2 ATK for that attack.", "Dancer"])
    assert over["unprinted"] == ["attack"]
    assert dp.printed_in_order("(AAT)(OPT) Negate it.", ["CAATXOPT) Negate it."])["unprinted"] == []
    row = {"read": {"name": "Twin", "kind": "creature", "types": [], "ability": "x"}, "ability_from": "t",
           "ability_check": moved}
    assert dp.fields_of(row)["ability"]["confidence"] == "review"


def test_icons_outside_the_band_are_compared_in_their_boxes():
    """A version's icon below the band (a two-line title) must be on the picture too; a version with types and no
    known icon boxes is never compared (None); one with no icons is compared where the card's others have them."""
    from PIL import Image
    rng = np.random.default_rng(2)
    art = rng.uniform(0, 255, (200, 100, 3)).astype(np.uint8)
    icon = art.copy()
    icon[30:44, 5:20] = 255                              # an icon at y 0.15-0.22, x 0.05-0.2
    head = rng.uniform(0, 255, (52, 60, 3)).astype(np.float32)
    look = lambda a: (head, Image.fromarray(a))
    box = [0.05, 0.15, 0.2, 0.22]
    assert ht.outside(box) and not ht.outside([0.02, 0.02, 0.1, 0.1])
    assert ht.score(look(icon), look(icon), [box]) == 1.0
    assert ht.score(look(art), look(icon), [box]) < 0.5
    boxes, types = {"k1": [box], "k2": []}, {"k1": ["light"], "k2": ["fire"], "k3": []}
    slots = ht.card_slots([{"key": "k1", "card": "c"}, {"key": "k3", "card": "c"}], boxes)
    assert ht.compared("k1", "c", types, boxes, slots) == [box] and ht.compared("k2", "c", types, boxes, slots) is None
    assert ht.compared("k3", "c", types, boxes, slots) == [box]


def test_eye_references_carry_types_over_the_bar(tmp_path):
    """Eras A-C: a reference typed by eye gives its types to the card's same-era picture whose header matches over
    the bar (the highest other-types pair + MARGIN: here a check with one more icon); a header with another icon
    stays unsure."""
    import json
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(3)
    base = rng.uniform(0, 255, (800, 400, 3)).astype(np.uint8)

    def save(name, icon=None, seed=None, box=(5, 5, 50, 45)):
        a = base.copy() if seed is None else np.random.default_rng(seed).uniform(0, 255, (800, 400, 3)).astype(np.uint8)
        img = Image.fromarray(a)
        if icon:
            ImageDraw.Draw(img).ellipse(box, fill=icon)
        (tmp_path / "attachments").mkdir(exist_ok=True)
        img.save(tmp_path / "attachments" / f"{name}.png")
        return f"attachments/{name}.png"
    files = {"P-ref": save("ref"), "P-same": save("same"), "P-icon": save("icon", (255, 0, 0)),
             "P-other": save("other", seed=9), "P-chk": save("chk", (0, 0, 255), box=(20, 20, 36, 36))}
    d = tmp_path / "derived"
    for sub in ("match", "readers/text", "readers/bars", "readers/types"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    rows = [{"id": k, "verdict": "draft", "card": "c" if k != "P-other" else "o", "file": f} for k, f in files.items()]
    (d / "match/verdicts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "readers/text/reads.jsonl").write_text("".join(json.dumps({"src": "discord", "file": f, "era": "B"}) + "\n"
                                                        for f in files.values()))
    (d / "readers/bars/reads.jsonl").write_text("")
    eye = {"pictures": {"P-ref": {"card": "c", "era": "B", "types": ["fire"], "file": files["P-ref"]},
                        "P-other": {"card": "o", "era": "B", "types": ["rock"], "file": files["P-other"]}},
           "checks": {"P-chk": {"card": "c", "era": "B", "types": ["fire", "ice"], "file": files["P-chk"]}}}
    (d / "readers/types/eye.json").write_text(json.dumps(eye))
    got = {r["id"]: r for r in ht.eye_read(str(tmp_path))}
    assert got["P-ref"]["sure"] and got["P-same"]["sure"] and got["P-same"]["types"] == ["fire"]
    assert not got["P-icon"]["sure"] and not got["P-chk"]["sure"]
