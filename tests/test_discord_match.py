"""discord_match on synthetic cards and a fake archive: hashes, names, rulings, verdicts, score + build end to end."""
import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from PIL import Image, ImageDraw

import commands.discord_match as dm
from conftest import REPO

G = "100000000000000001"                            # a made-up guild id


def art(seed: int, w: int = 600, h: int = 900) -> Image.Image:
    """A fake card: coarse random art, a title band and a text box."""
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (h // 50, w // 50, 3), dtype=np.uint8)).resize((w, h), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, w, round(0.15 * w)), fill=(200, 196, 190))
    draw.text((w // 3, round(0.05 * w)), f"CARD {seed}", fill=(20, 20, 20))
    draw.rectangle((0, round(1.0 * w), w, h), fill=(240, 236, 228))
    return img


def jpeg(img: Image.Image, width: int = 400, quality: int = 70) -> Image.Image:
    return dm.degrade(img.convert("RGB"), width, quality)


def snowflake(seconds: float) -> str:
    return str((int(seconds * 1000) - 1420070400000) << 22)


# ========================
# Pieces
# ========================
def test_hashes_survive_resizing_but_tell_cards_apart():
    a, b = art(1), art(2)
    same = dm.hash_distance(dm.hashes(a), dm.hashes(jpeg(a)))
    other = dm.hash_distance(dm.hashes(a), dm.hashes(b))
    assert same <= dm.HASH_FULL and other >= dm.HASH_ZERO
    assert dm.hash_score(same) == 1.0 and dm.hash_score(other) == 0.0


def test_trim_and_shapes():
    card = art(3)
    padded = Image.new("RGB", (card.width + 80, card.height + 80), dm.DISCORD_GRAY)
    padded.paste(card, (40, 40))
    trimmed = dm.trim(padded)
    assert abs(trimmed.width - card.width) <= 2 and abs(trimmed.height - card.height) <= 2
    assert [dm.shape_of(*s) for s in ((1200, 1800), (1200, 2400), (900, 900), (1920, 1080))] == ["2:3", "1:2",
                                                                                                  "1:1", None]
    assert dm.shape_of(1920, 1080, force=True) == "1:1"


CARDS = {
    "apex-predator": {"id": "apex-predator", "name": "A P E X P R E D A T O R", "key": "Apex Predator",
                      "png": "Creatures/R.E.P.O/2 Stars/Apex_Predator.png"},
    "stone-golem": {"id": "stone-golem", "name": "Stone Golem", "key": "Stone Golem",
                    "png": "Creatures/Crypt/2 Stars/Stone_Golem.png", "ability": "This creature takes piercing "
                    "shockwave damage each time it attacks, then reaches half HP and attacks twice"},
    "indoctrination": {"id": "indoctrination", "name": "Indoctrinati0n", "key": "Indoctrination",
                       "aliases": ["zealotry"], "png": "Items/Indoctrination.png"},
    "bob": {"id": "bob", "name": "Bob", "key": "Bob", "png": "Creatures/Bob.png"},
    "salt-pepper": {"id": "salt-pepper", "name": "Salt & Pepper", "key": "Salt & Pepper", "png": "Items/SP.png"},
}


def test_names_fuzzy_exact_and_mentions():
    names = dm.Names(CARDS)
    assert list(names.fuzzy("A P E X  P R E D A T O R", 0.6)) == ["apex-predator"]
    assert names.fuzzy("INDOCTRINATION", 0.6)["indoctrination"][0] == 1.0       # 0 read as o
    assert next(iter(names.fuzzy("Stone G0lem", 0.6))) == "stone-golem"
    assert names.exact("zealotry") == {"indoctrination": "alias"}
    assert names.mentions("honestly stone golem is op")["stone-golem"] == "stone golem"
    assert names.mentions("bobcat, bob and zealotry") == {"indoctrination": "zealotry"}   # 3-letter words skipped


def test_file_names():
    names = dm.Names(CARDS)
    assert dm.file_scores(names, "Stone_Golem.png") == {"stone-golem": (1.0, "Stone_Golem")}
    assert set(dm.file_scores(names, "Stone_Golem_v2_final.png")) == {"stone-golem"}
    for generic in ("image.png", "unknown.png", "IMG_2041.jpg", "Screenshot_2021-01-01_123456.png", "image0.png",
                    "8f14e45fceea167a5a36dedd4bea2543.webp"):
        assert dm.file_scores(names, generic) == {}, generic


def test_file_name_ids_spoilers_and_numbers():
    names = dm.Names({**CARDS, "kit": {"id": "kit", "name": "Kit", "key": "Kit"},
                      "kit-2": {"id": "kit-2", "name": "Kit 2", "key": "Kit 2", "png": "Items/Kit2.png"}})
    assert set(dm.file_scores(names, "SPOILER_SPOILER_Stone_Golem.png")) == {"stone-golem"}
    assert set(dm.file_scores(names, "Bob2.png")) == {"bob"}                       # a version number
    assert set(dm.file_scores(names, "Kit2.png")) == {"kit-2"}                     # a name's own number
    for upload in ("q7Rx2Tb.png", "hvkdz41.jpg", "aKwPzQe.png"):
        assert dm.name_tokens(upload) == [], upload
    assert dm.name_tokens("PoD11ZOMG.png") == ["pod11zomg"]                          # a series code


def test_embed_images_stored_as_bin():
    assert dm.is_image("attachments/x.bin", {"kind": "thumbnail"}, "embed")
    assert dm.is_image("attachments/x.bin", {"content_type": "image/avif"}, "attachment")
    assert not dm.is_image("attachments/x.bin", {"content_type": "video/mp4"}, "attachment")


def test_art_alone_is_not_the_card():
    cards = {c: {"id": c} for c in ("a", "b")}
    art_hash = {"card": "a", "s": {"hash": 1.0, "file": 1.0}, "why": {"hash_part": "art"}}
    plain = dm.judge({"shape": "2:3", "ocr": True, "lines": [], "candidates": [art_hash]}, cards)
    assert plain["status"] == "art" and plain["art"] == ["a"] and plain["cards"] == []
    named = dm.judge({"shape": "2:3", "ocr": True, "lines": ["SOME OTHER TITLE", "Health", "Attack twice a turn"],
                      "candidates": [art_hash]}, cards)
    assert named["status"] == "ambiguous" and "draft" in named["reason"]           # a draft under another title
    blind = dm.judge({"shape": "2:3", "ocr": False, "candidates": [art_hash]}, cards)
    assert blind["status"] == "confident"                                         # no OCR: can't tell art apart
    whole = {"card": "a", "s": {"hash": 1.0}, "why": {"hash_part": "whole"}}
    assert dm.judge({"shape": "2:3", "ocr": True, "lines": [], "candidates": [whole]}, cards)["status"] == "confident"


def test_title_under_the_art():
    cards = {"the-last-melon-trader": {"id": "the-last-melon-trader", "name": "The Last Melon Trader",
                                       "key": "The Last Melon Trader"},
             "melon": {"id": "melon", "name": "Melon", "key": "Melon"},
             "cursed": {"id": "cursed", "name": "Cursed", "key": "Cursed"}}
    index = dm.CardIndex(cards, [])
    lines = [[text, 1.0, 0.3, y, 0.4, 0.03] for text, y in (("THF LAST", 0.35), ("MELON", 0.39), ("TRADER", 0.43),
                                                            ("HEALTH", 0.5), ("Cursed:", 0.6))]
    feat = {"shape": "1:2", "w": 600, "h": 1200, "title": [], "lines": lines}
    titles = dm.title_texts(feat, index.names)
    assert "THF LAST MELON TRADER" in titles and "Cursed:" not in titles
    twice = dm.title_texts({"shape": "1:2", "w": 400, "h": 800, "title": [["Melon", 1.0]],
                            "lines": [["Melon *", 1.0, 0.3, 0.01, 0.3, 0.03]]})
    assert not [t for t in twice if "Melon Melon" in t]                         # one line read twice: no join
    parts = dm.title_texts({"shape": "1:2", "w": 400, "h": 800, "title": [["Old Grey", 1.0], ["The", 1.0]],
                            "lines": [["The Old Grey", 1.0, 0.2, 0.01, 0.6, 0.03], ["DOG", 1.0, 0.4, 0.05, 0.2, 0.05]]})
    assert "The Old Grey DOG" in parts                                            # the fullest read of each line
    found = {c["card"]: c["s"].get("title", 0) for c in dm.score_image(index, feat)}
    assert found["the-last-melon-trader"] == 1.0 and found.get("melon", 0) < 0.3


def test_people_map_skips_ports_and_takes_the_most():
    def first(pid, channel, n):
        return [({"poster": {"id": pid, "username": pid}, "channel": channel}, credited) for credited in n]
    posts = first("u1", "port", ["B"] * 30) + first("u1", "cards", ["A"] * 11 + ["B"] * 9) + \
        first("u2", "cards", ["B"] * 5)
    found = dm.people_map(posts, ["A", "B"])
    assert found["u1"]["person"] == "A" and found["u1"]["port_first_posts"] == 30 and "guess" in found["u1"]["basis"]
    assert found["u2"]["person"] == "B"


def test_parse_rulings():
    assert dm.parse_rulings("A12 = Isma, A13 = none") == [("A12", "Isma"), ("A13", "none")]
    assert dm.parse_rulings("a3: #2; A4 = Salt & Pepper\nA5=Stone Golem + Bob") == [
        ("A3", "#2"), ("A4", "Salt & Pepper"), ("A5", "Stone Golem + Bob")]
    assert dm.parse_rulings("no rulings here") == []


def test_resolve_rulings():
    names = dm.Names(CARDS)
    entry = {"candidates": [{"card": "bob"}, {"card": "stone-golem"}]}
    assert dm.resolve("none", entry, names) == ([], "none")
    assert dm.resolve("?", entry, names)[0] is None
    assert dm.resolve("#2", entry, names) == (["stone-golem"], "candidate 2")
    assert dm.resolve("#7", entry, names)[1].startswith("error")
    assert dm.resolve("zealotry", entry, names) == (["indoctrination"], "card")
    assert dm.resolve("Salt & Pepper", entry, names) == (["salt-pepper"], "card")      # not split at &
    assert dm.resolve("Stone Golem + Bob", entry, names) == (["stone-golem", "bob"], "cards")
    assert dm.resolve("apex-predator", entry, names) == (["apex-predator"], "card")
    missing = dm.resolve("Stone Gollum Junior", entry, names)
    assert missing[0] is None and missing[1].startswith("error")


def candidate(card, **s):
    return {"card": card, "s": s, "why": {"title": "a long printed title"} if "title" in s else {}}


def test_judge_verdicts():
    cards = {c: {"id": c} for c in ("a", "b", "c")}
    cards["b2"] = {"id": "b2", "dup": "b"}
    sure = dm.judge({"shape": "2:3", "candidates": [candidate("a", hash=1.0, title=1.0), candidate("b", hash=0.2)]},
                    cards)
    assert sure["status"] == "confident" and sure["cards"] == ["a"]
    split = dm.judge({"shape": "2:3", "candidates": [candidate("a", file=1.0), candidate("b", title=1.0)]}, cards)
    assert split["status"] == "ambiguous" and "disagree" in split["reason"] and set(split["maybe"]) == {"a", "b"}
    dup = dm.judge({"shape": "2:3", "candidates": [candidate("b", hash=1.0), candidate("b2", hash=1.0, title=1.0)]},
                   cards)
    assert dup["status"] == "confident" and dup["cards"] == ["b2"]                    # the title decides a duplicate
    close = dm.judge({"shape": "2:3", "candidates": [candidate("a", hash=0.95), candidate("b", hash=0.9)]}, cards)
    assert close["status"] == "ambiguous" and "close second" in close["reason"]
    chat = dm.judge({"shape": None, "candidates": [candidate("a", text=0.6)]}, cards)
    assert chat["status"] == "mention"
    shot = dm.judge({"shape": None, "candidates": [candidate("a", line=1.0, anchor=1.0), candidate("b", line=1.0,
                                                                                                    text=0.6),
                                                   candidate("c", line=0.5)]}, cards)
    assert shot["status"] == "confident" and shot["cards"] == ["a", "b"] and shot["maybe"] == ["c"]
    short = dm.judge({"shape": "1:1", "candidates": [{"card": "a", "s": {"title": 1.0}, "why": {"title": "Bob"}}]},
                     cards)
    assert short["status"] == "ambiguous" and "short" in short["reason"]


def test_disagreements_settled_by_kind():
    cards = {c: {"id": c} for c in ("keeper", "ward", "frost", "ice", "oath", "creed")}

    def why(card, part="whole", **s):
        return {"card": card, "s": s, "why": {"title": "a long printed title", "hash_part": part}}
    # the ability moved to another card: its words (body) vs the card's own title, file and versions' text
    keeper = dm.judge({"shape": "1:2", "ocr": True, "lines": [], "candidates": [
        why("keeper", title=1.0, file=1.0, ocr=1.0), why("ward", body=0.99)]}, cards)
    assert keeper["status"] == "confident" and keeper["cards"] == ["keeper"] and keeper["settled"]
    # an old name: the whole card's hash and its versions' text vs a title and a file name naming another card
    frost = dm.judge({"shape": "2:3", "ocr": True, "lines": [], "candidates": [
        why("frost", hash=1.0, ocr=1.0), why("ice", title=1.0, file=1.0)]}, cards)
    assert frost["status"] == "confident" and frost["cards"] == ["frost"]
    # the versions' text beats a name alone, even from the runner-up
    creed = dm.judge({"shape": "2:3", "ocr": True, "lines": [], "candidates": [
        why("oath", title=1.0, file=1.0), why("creed", "art", hash=0.4, ocr=0.97, body=0.75)]}, cards)
    assert creed["status"] == "confident" and creed["cards"] == ["creed"] and creed["guess"] == "creed"
    # the ability's words alone don't beat a name
    named = dm.judge({"shape": "2:3", "ocr": True, "lines": [], "candidates": [
        why("oath", title=1.0, file=1.0), why("creed", body=0.95)]}, cards)
    assert named["status"] == "ambiguous" and "disagree" in named["reason"]       # neither: the owner rules


def test_body_needs_enough_words():
    cards = {"spin": {"id": "spin", "ability": "Spin wildly, gaining armor."},
             "golem": {"id": "golem", "ability": "Spin wildly, gaining armor permanently and inflicting sleepiness "
                                                 "upon nearby goblins, wolves, ravens."}}
    cards.update({f"x{i}": {"id": f"x{i}", "ability": f"Filler{i} words here"} for i in range(30)})
    bodies = dm.Bodies(cards)
    found = bodies.scores("SPIN WILDLY, GAINING ARMOR PERMANENTLY AND INFLICTING SLEEPINESS UPON NEARBY GOBLINS, "
                          "WOLVES, RAVENS")
    assert found["spin"] == (1.0, 4) and found["golem"][1] >= 8 and found["golem"][0] == 1.0
    spin = {"card": "spin", "s": {"body": 1.0}, "why": {"body_words": 4}}
    assert dm.few_words(spin) == 0.5                                               # four words: half
    golem = {"card": "golem", "s": {"title": 1.0, "file": 1.0}, "why": {"title": "Iron Golem"}}
    verdict = dm.judge({"shape": "2:3", "ocr": True, "lines": [], "candidates": [golem, spin]}, cards)
    assert verdict["status"] == "confident" and verdict["cards"] == ["golem"]     # no vote from four words


def test_titles_on_two_lines_and_other_names():
    cards = {"the-old-king": {"id": "the-old-king", "name": "The Old King", "key": "The Old King"},
             "heir": {"id": "heir", "name": "Heir of the Old King", "key": "Heir of the Old King"}}
    index = dm.CardIndex(cards, [])
    lines = [["Cape of the", 1.0, 0.23, 0.006, 0.34, 0.04], ["Old King", 1.0, 0.19, 0.046, 0.44, 0.04],
             ["Some long rules text on the card goes here", 1.0, 0.02, 0.68, 0.9, 0.036]]
    feat = {"shape": "2:3", "w": 400, "h": 600, "title": [["Cape of the", 1.0], ["Old King", 1.0]], "lines": lines}
    assert dm.title_blocks(feat) == ["Cape of the Old King"]
    found = {c["card"]: c["s"].get("title", 0) for c in dm.score_image(index, feat)}
    assert all(v < 0.3 for v in found.values())                                   # a retired card: no title
    verdict = dm.judge({"shape": "2:3", "ocr": True, "lines": [line[0] for line in lines],
                        "candidates": dm.score_image(index, feat)}, cards)
    assert verdict["status"] == "none"
    older = dm.CardIndex({"map": {"id": "map", "name": "Battle Map", "key": "Battle Map"}}, [])
    renamed = {"shape": "2:3", "w": 400, "h": 600, "title": [["Faded", 1.0], ["Battle Map", 1.0]],
               "lines": [["Faded", 1.0, 0.3, 0.006, 0.3, 0.04], ["Battle Map", 1.0, 0.2, 0.046, 0.5, 0.04]]}
    assert dm.score_image(older, renamed)[0]["s"]["title"] == 1.0                 # an older, longer name: kept
    assert dm.swapped("Fire Lance", ["flame", "lance"]) and not dm.swapped("Iron Golem.", ["iron", "golem"])
    assert not dm.swapped("SWORD", ["dark", "sword"]) and not dm.swapped("A Mr. Fixit", ["mr", "fixit"])
    assert dm.swapped("Ding of the Old King", ["heir", "of", "the", "old", "king"])   # ding ~ king: not compared


def test_names_read_with_nothing_behind_them():
    cards = {c: {"id": c, "name": n, "key": n} for c, n in (("wobbly", "Wobbly"), ("sticky", "Sticky"),
                                                             ("dizzy", "Dizzy"), ("soggy", "Soggy"),
                                                             ("fuzzy", "Fuzzy"))}
    index = dm.CardIndex(cards, [])
    meme = {"shape": None, "lines": [["WOBBLY", 1.0, 0.3, 0.4, 0.4, 0.1]]}
    assert dm.judge({"shape": None, "ocr": True, "lines": ["WOBBLY"],
                     "candidates": dm.score_image(index, meme)}, cards)["status"] == "none"
    chart = {"shape": None, "lines": [[n, 1.0, 0.1, 0.1 * i, 0.3, 0.05] for i, n in enumerate(
        ("Sticky", "Dizzy", "Soggy", "Fuzzy", "Wobbly"))]}
    verdict = dm.judge({"shape": None, "ocr": True, "lines": [line[0] for line in chart["lines"]],
                        "candidates": dm.score_image(index, chart)}, cards)
    assert verdict["status"] == "none"                                            # a list of names
    card = {"shape": None, "lines": [["Wobbly", 1.0, 0.3, 0.05, 0.3, 0.05], ["Health", 1.0, 0.1, 0.5, 0.2, 0.03],
                                     ["Attack", 1.0, 0.1, 0.6, 0.2, 0.03]]}
    found = dm.score_image(index, card)
    assert found[0]["card"] == "wobbly" and found[0]["s"]["line"] == 1.0         # a card's photo: kept


def test_a_sheet_is_read_cell_by_cell(tmp_path):
    names = (("alpha-wolf", "Alpha Wolf", 31), ("beta-bear", "Beta Bear", 32), ("gamma-ray", "Gamma Ray", 33),
             ("delta-wing", "Delta Wing", 34))
    cards = {cid: {"id": cid, "name": n, "key": n} for cid, n, _ in names}
    imgs = [art(seed, 300, 450) for _, _, seed in names]
    versions = [{"key": cid, "card": cid, "current": True, "ts": 1, "commit": "c1", "whole": dm.hashes(img),
                 "art": dm.hashes(img.crop(dm.card_box(300, 450, dm.ART)))} for (cid, _, _), img in zip(names, imgs)]
    index = dm.CardIndex(cards, versions)
    sheet = Image.new("RGB", (600, 900))
    for i, img in enumerate(imgs):
        sheet.paste(img, (300 * (i % 2), 450 * (i // 2)))
    path = tmp_path / "sheet.png"
    sheet.save(path)
    lines = [[n, 1.0, 0.1 + 0.5 * (i % 2), 0.01 + 0.5 * (i // 2), 0.3, 0.02] for i, (_, n, _) in enumerate(names)]
    lines += [["Some rules text that names no card", 1.0, 0.02, 0.4, 0.45, 0.02]]
    feat = {"shape": "2:3", "w": 600, "h": 900, "ocr": True, "lines": lines}
    assert dm.sheet_grid(feat, index.names) == (2, 2)
    one = {"shape": "2:3", "w": 600, "h": 900, "lines": [["Alpha Wolf", 1.0, 0.35, 0.01, 0.3, 0.03],
                                                          ["Beta Bear", 1.0, 0.05, 0.8, 0.2, 0.02],
                                                          ["Gamma Ray", 1.0, 0.05, 0.85, 0.2, 0.02]]}
    assert dm.sheet_grid(one, index.names) is None                     # names in a card's text: one column
    cells = []
    for cell in dm.sheet_cells(str(path), (2, 2)):
        part = dm.cell_feat(feat, cell)
        cells.append({**part, "candidates": dm.score_image(index, part)})
    verdict = dm.judge({"shape": "2:3", "ocr": True, "cells": cells}, cards)
    assert verdict["status"] == "confident" and verdict["cards"] == [cid for cid, _, _ in names]
    lone = dm.judge({"shape": "2:3", "ocr": True, "cells": cells[:1] + [{**c, "candidates": []} for c in cells[1:]]},
                    cards)
    assert lone["cards"] == ["alpha-wolf"]


def test_alias_rules():
    cards = {"grand-old-knight": {"id": "grand-old-knight", "name": "Grand Old Knight", "key": "Grand Old Knight"},
             "twin-wardens": {"id": "twin-wardens", "name": "Twin Wardens", "key": "Twin Wardens"},
             "long-pike": {"id": "long-pike", "name": "Long Pike", "key": "Long Pike"},
             "shadow-steel-swordsman": {"id": "shadow-steel-swordsman", "name": "Shadow Steel Swordsman",
                                        "key": "Shadow Steel Swordsman"},
             "pale-keeper": {"id": "pale-keeper", "name": "Pale Keeper", "key": "Pale Keeper"}}
    names = dm.Names(cards)

    def ok(cid, text, printed=False):
        return dm.alias_ok(text, cid, names, set(names.card_forms[cid]), printed)
    assert ok("grand-old-knight", "gok")                                          # initials
    assert not ok("twin-wardens", "wardans") and not ok("twin-wardens", "those two")
    assert ok("twin-wardens", "Twin Kings") and ok("long-pike", "Ironclad Lance", printed=True)
    assert not ok("long-pike", "steelhelmet")                                     # a swapped file name
    assert not ok("shadow-steel-swordsman", dm.typed_name("shadow_steel_sword_color.png"))
    assert not ok("pale-keeper", "p kpr")
    assert not ok("twin-wardens", "Grand Old Knight")                             # another card's name
    assert dm.typed_name("Pale_Wizard2.png") == "Pale Wizard" and dm.typed_name("PaleWizard.png") == "Pale Wizard"


def test_title_pieces():
    cards = {"watermelon": {"id": "watermelon", "name": "Watermelon", "key": "Watermelon"},
             "inconspicuous-watermelon": {"id": "inconspicuous-watermelon", "name": "Inconspicuous Watermelon",
                                          "key": "Inconspicuous Watermelon", "dup": "watermelon"},
             "dark-sword": {"id": "dark-sword", "name": "Dark Sword", "key": "Dark Sword"},
             "sword": {"id": "sword", "name": "Sword", "key": "Sword"}}
    index = dm.CardIndex(cards, [])
    feat = {"shape": "2:3", "w": 600, "h": 900, "title": [["INCONSPICUOUS", 1.0], ["WATERMELON", 1.0]]}
    found = {c["card"]: c["s"].get("title", 0) for c in dm.score_image(index, feat)}
    assert found["inconspicuous-watermelon"] == 1.0 and found.get("watermelon", 0) < 0.3   # a piece of the read
    credit = dm.ramp(index.names.part_ratio("dark-sword", ["SWORD"])[0], dm.TITLE_FLOOR, dm.TITLE_FULL)
    assert 0.3 <= credit < 1.0                                                            # the hash brings it in
    hashed = [{"card": "dark-sword", "s": {"hash": 1.0, "title": credit}, "why": {"title": "SWORD"}},
              {"card": "sword", "s": {"title": 1.0}, "why": {"title": "SWORD"}}]
    verdict = dm.judge({"shape": "2:3", "candidates": hashed}, cards)
    assert verdict["status"] == "confident" and verdict["cards"] == ["dark-sword"]      # the read fits both


def test_fake_ocr_title_scores(monkeypatch):
    monkeypatch.setattr(dm, "_vision", lambda: True)
    monkeypatch.setattr(dm, "ocr_lines", lambda img: [["Stone Golem", 0.99, 0.3, 0.2, 0.4, 0.5]]
                        if img.height < img.width / 2 else [])
    feat = dm.features(jpeg(art(5)), ocr=True)
    assert feat["title"] == [["Stone Golem", 0.99]] and feat["shape"] == "2:3"
    index = dm.CardIndex(CARDS, [])
    found = {c["card"]: c for c in dm.score_image(index, feat)}
    assert found["stone-golem"]["s"]["title"] == 1.0


OLD_TEXT = "BREAKFAST +2 MAX HP HEAL FOR 1 HP DRINK IT BEFORE THE FIGHT"
MID_TEXT = "BREAKFAST HEAL 3 HP INFLICT 2T2 REGEN AND 2T1 STUFFED DRINK IT"
NOW_TEXT = "BREAKFAST HEAL 3 HP INFLICT 2 REGEN FOR 2 TURNS INFLICT 1 STUFFED"


def text_versions(img):
    """One card, three versions: the same art and nearly the same hash, three texts."""
    whole, part = dm.hashes(img), dm.hashes(img.crop(dm.card_box(img.width, img.height, dm.ART)))
    return [{"key": key, "card": "breakfast", "current": key == "now", "ts": ts, "commit": f"c{ts}", "whole": whole,
             "art": part, "lines": [[text, 1.0, 0, 0, 1, 0.1]]}
            for key, ts, text in (("now", 3, NOW_TEXT), ("mid", 2, MID_TEXT), ("old", 1, OLD_TEXT))]


def test_version_picked_by_text_when_only_the_text_changed():
    img = art(21)
    index = dm.CardIndex({"breakfast": {"id": "breakfast", "name": "Breakfast"}}, text_versions(img))
    assert index.looks() == {"now": 0, "mid": 1, "old": 2}
    whole, part = dm.hashes(img), dm.hashes(img.crop(dm.card_box(img.width, img.height, dm.ART)))

    def pick(text, **feat):
        return index.pick_version("breakfast", {"shape": "2:3", "whole": whole, "art": part, **feat}, dm.squash(text))
    found = pick("BREAKFAST +2 MAX HP HEAL F0R 1 HP DRINK IT BEF0RE THE FIGHT")       # OCR noise, the oldest text
    assert found["by"] == "text" and found["keys"] == ["old"]
    far = dm.hashes(art(99))
    cropped = pick("HEAL 3 HP INFLICT 2 REGEN FOR 2 TURNS INFLICT 1 STUFFED", whole=far, art=far)
    assert cropped["by"] == "text" and cropped["keys"] == ["now"]                    # a crop: the text still tells
    bare = index.pick_version("breakfast", {"shape": "2:3", "whole": whole, "art": part}, "")
    assert bare["by"] == "hash" and set(bare["keys"]) == {"now", "mid", "old"}       # the hash can't: all tie
    draft = pick("BREAKFAST GAIN 5 SHIELD AND DRAW A CARD AT THE START OF YOUR TURN")
    assert draft["by"] == "draft" and draft["keys"] == []                            # in no git version
    lost = index.pick_version("breakfast", {"shape": "2:3", "whole": far, "art": far}, "")
    assert lost["by"] is None and lost["keys"] == []                                 # a crop without text


def test_scan_ties_go_to_the_nearest_whole_card():
    img = art(22)
    versions = text_versions(img)
    versions[0]["whole"], versions[1]["whole"] = dm.hashes(art(23)), dm.hashes(art(27))   # all: the same art
    index = dm.CardIndex({"breakfast": {"id": "breakfast"}}, versions)
    feat = {"whole": versions[2]["whole"], "art": versions[2]["art"]}
    assert index.versions[index.scan(feat)["breakfast"][1]]["key"] == "old"


def test_text_method_finds_a_crop():
    img = art(24)
    cards = {"breakfast": {"id": "breakfast", "name": "Breakfast", "key": "Breakfast"},
             "pancake": {"id": "pancake", "name": "Pancake", "key": "Pancake"}}
    versions = text_versions(img) + [{"key": "p", "card": "pancake", "current": True, "ts": 1, "commit": "c1",
                                      "whole": dm.hashes(art(25)), "art": dm.hashes(art(26)),
                                      "lines": [["PANCAKE FLIP A COIN TO GAIN 2 SPEED OR LOSE 1 DEF", 1.0]]}]
    index = dm.CardIndex(cards, versions)
    far = dm.hashes(art(98))
    feat = {"shape": "1:2", "w": 600, "h": 1200, "whole": far, "art": far,
            "lines": [["ast", 0.9, 0.4, 0.0, 0.2, 0.03],
                      ["HEAL 3 HP INFLICT 2 REGEN FOR 2 TURNS", 1.0, 0.1, 0.6, 0.8, 0.03],
                      ["INFLICT 1 STUFFED", 1.0, 0.2, 0.65, 0.6, 0.03]]}
    found = dm.score_image(index, feat)
    assert found[0]["card"] == "breakfast" and found[0]["s"]["ocr"] == 1.0
    assert found[0]["why"]["pick"]["keys"] == ["now"]


def write_build_archive(root: Path, posts: list, candidates: dict, versions: list, cards: dict,
                        extra: Optional[dict] = None):
    """An archive whose pairs.json is given: build alone, no images. extra: more fields per image (by post)."""
    for sub in ("messages", "attachments", "match"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    lines, images = [], {}
    for n, (when, shape, author) in enumerate(posts):
        mid = snowflake(when)
        lines.append({"guild_id": G, "channel_id": "100", "channel_name": "card-ideas", "thread": None, "id": mid,
                      "timestamp": dm.iso(when), "author": {"id": author, "username": author,
                                                           "bot": author == "bot"},
                      "content": "", "attachments": [{"id": str(n), "filename": "image.png", "sha256": f"{n:064x}",
                                                      "file": f"attachments/{n:064x}.png"}],
                      "embeds": [], "referenced_message_id": None, "jump_url": f"https://discord.com/x/{mid}"})
        images[f"{mid}:{n}"] = {"key": f"{mid}:{n}", "message": mid, "media": str(n), "file": f"attachments/{n}.png",
                                "sha256": f"{n:064x}", "filename": "image.png", "kind": "attachment", "shape": shape,
                                "whole": None, "text": "", "title": [], "lines": [], "candidates": candidates[n],
                                **(extra or {}).get(n, {})}
    (root / "messages" / "100.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    (root / "channels.json").write_text(json.dumps([{"id": "100", "name": "card-ideas", "type": 0}]))
    (root / "summary.json").write_text(json.dumps({"guild_id": G, "status": "done"}))
    (root / "match" / "pairs.json").write_text(json.dumps({"version": dm.PAIRS_VERSION, "cards": cards,
                                                           "versions": versions, "images": images}))


def test_build_revisions_current_since_and_confirm(tmp_path):
    cards = {"breakfast": {"id": "breakfast", "name": "Breakfast", "key": "Breakfast"},
             "shooting-star": {"id": "shooting-star", "name": "Shooting Star", "key": "Shooting Star"}}
    versions = [{"key": "now", "card": "breakfast", "commit": "c3", "ts": 1700000000, "current": True, "look": 0},
                {"key": "mid", "card": "breakfast", "commit": "c2", "ts": 1690000000, "current": False, "look": 0},
                {"key": "old", "card": "breakfast", "commit": "c1", "ts": 1680000000, "current": False, "look": 1}]

    def seen(keys, by="text", **s):
        return [{"card": "breakfast", "s": s or {"hash": 1.0, "title": 1.0}, "why": {"pick": {"keys": keys, "by": by}}}]
    t = 1600000000
    posts = [(t, "2:3", "maker"), (t + 86400, "2:3", "alice"), (t + 2 * 86400, "2:3", "alice"),
             (t + 3 * 86400, "2:3", "alice"), (t + 4 * 86400, None, "maker")]
    candidates = {0: seen(["old"]),                                    # the oldest look
                  1: seen(["old"], ocr=1.0, body=1.0),                 # a crop of it: no revision
                  2: seen(["now", "mid"]),                             # the current look (two versions, one look)
                  3: seen(["now", "old"], by="hash"),                  # looks tied: untold
                  4: [{"card": "shooting-star", "s": {"line": 0.227, "anchor": 0.625, "body": 1.0}, "why": {}}]}
    root = tmp_path / "archive"
    write_build_archive(root, posts, candidates, versions, cards)
    result = dm.build(str(root), str(tmp_path / "none"), log=lambda *a: None)
    d = result["cards"]["breakfast"]["discord"]
    assert d["first"]["version"]["shows"] == "older" and d["first"]["version"]["commits"] == ["c1"]
    assert [(r["date"][:10], r["shows"], r["commits"]) for r in d["revisions"]] == [("2020-09-15", "current",
                                                                                     ["c3", "c2"])]
    assert d["current_since"].startswith("2020-09-15") and d["looks_untold"] == 1
    star = result["cards"]["shooting-star"]["discord"]
    entries = json.loads((root / "ambiguous.json").read_text())["entries"]
    assert [e["confirm"] for e in entries] == [["shooting-star"]] and star["to_confirm"] == [entries[0]["id"]]
    assert star["first"]["date"].startswith("2020-09-17") and "confirm" in entries[0]["reason"]
    assert dm.rulings_main(str(root), [f"{entries[0]['id']} = none"]) == 0
    assert (json.loads((root / "matches.json").read_text())["cards"]["shooting-star"]["discord"] or {}).get(
        "first") is None


def test_twins_bots_and_suspects(tmp_path):
    cards = {"fang": {"id": "fang", "name": "Fang", "key": "Fang", "fs": 1700000000},
             "bob": {"id": "bob", "name": "Bob", "key": "Bob"}}
    short = [{"card": "fang", "s": {"title": 1.0}, "why": {"title": "Fang"}}]
    sure = [{"card": "fang", "s": {"hash": 1.0, "title": 1.0}, "why": {"title": "Fang", "hash_part": "whole"}}]
    bob = [{"card": "bob", "s": {"title": 0.7}, "why": {"title": "Bobb"}}]
    t = 1600000000
    posts = [(t, "1:2", "maker"), (t + 86400, "1:2", "maker"), (t + 2 * 86400, "1:2", "bot"),
             (t + 30 * 86400, "1:2", "maker")]
    root = tmp_path / "archive"
    keys = [f"{snowflake(when)}:{n}" for n, (when, _, _) in enumerate(posts)]
    untitled = [{"card": "fang", "s": {"body": 0.5}, "why": {"body_words": 8}}]
    posts += [(t - 86400, "1:2", "maker"), (t - 2 * 86400, "1:2", "maker")]
    keys = [f"{snowflake(when)}:{n}" for n, (when, _, _) in enumerate(posts)]
    write_build_archive(root, posts, {0: short, 1: short, 2: bob, 3: sure, 4: untitled, 5: untitled}, [], cards,
                        {0: {"twins": [[keys[3], 1.3, 0.5]]}, 1: {"twins": [[keys[3], 6.0, 9.0]]},
                         4: {"twins": [[keys[3], 6.0, 1.0]]}, 5: {"twins": [[keys[3], 2.0, 9.0]]}})
    result = dm.build(str(root), str(tmp_path / "none"), log=lambda *a: None)
    d = result["cards"]["fang"]["discord"]
    assert d["first"]["date"].startswith("2020-09-11") and d["posts"] == 4       # the same image: sure
    assert d["earlier_ambiguous"] == []                     # 1: alike, titled: sure; 4: alike, no title: not
    entries = json.loads((root / "ambiguous.json").read_text())["entries"]
    assert [e["post"]["date"][:10] for e in entries] == ["2020-09-12"]           # the bot's post isn't listed

    late = {"fang": {**cards["fang"], "fs": t - 90 * 86400}, "bob": cards["bob"]}
    root2 = tmp_path / "late"
    write_build_archive(root2, posts[:4], {0: short, 1: short, 2: bob, 3: sure}, [], late)
    d = dm.build(str(root2), str(tmp_path / "none"), log=lambda *a: None)["cards"]["fang"]["discord"]
    assert d["suspect"] and d["earliest_open"]["date"].startswith("2020-09-13")
    entries = json.loads((root2 / "ambiguous.json").read_text())["entries"]
    assert entries[0]["suspect"] == ["fang"]
    edits = json.loads((root2 / "proposal" / dm.EDITS_FILE).read_text())
    assert edits["pending"]["Fang"]["created_if_ruled"]["date"] == "2020-09-13"


def test_root_inside_git_is_refused(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    (tmp_path / "repo" / "archive").mkdir()
    with pytest.raises(SystemExit, match="inside the git repo"):
        dm.check_root(str(tmp_path / "repo" / "archive"))


def test_never_talks_to_discord():
    source = (REPO / "commands" / "discord_match.py").read_text()
    assert "discord.com/api" not in source and "Authorization" not in source and "discord_archive import" not in source


# ========================
# End to end
# ========================
CARD_SET = {"alpha-wolf": ("Alpha Wolf", 11), "beta-bear": ("Beta Bear", 12), "gamma-ray": ("Gamma Ray", 13)}


def make_cards_repo(base: Path) -> Path:
    repo = base / "cards"
    data = repo / "site" / "dist" / "data"
    data.mkdir(parents=True)
    cards = []
    for cid, (name, seed) in CARD_SET.items():
        png = f"Items/{name.replace(' ', '_')}.png"
        (repo / png).parent.mkdir(parents=True, exist_ok=True)
        art(seed).save(repo / png)
        cards.append({"id": cid, "name": name, "key": name, "png": png, "author": "MichaelJSr",
                      "fs": 1700000000 + seed})
    (data / "cards.json").write_text(json.dumps({"cards": cards}))
    (data / "versions.json").write_text(json.dumps({"available": False, "commits": [], "cards": {}}))
    (repo / "site" / "Card Authors.txt").write_text("[people]\nMichaelJSr = MichaelJSr\n[cards]\nBeta Bear = Croissant\n")
    return repo


def make_archive(base: Path) -> Path:
    root = base / "archive"
    for sub in ("messages", "attachments"):
        (root / sub).mkdir(parents=True)
    lines, n = [], 0

    def post(when, image=None, filename="image.png", content="", bot=False, author="maker"):
        nonlocal n
        n += 1
        mid = snowflake(when)
        attachments = []
        if image is not None:
            buf = root / "attachments" / f"tmp{n}.png"
            image.save(buf)
            digest = sha256(buf.read_bytes()).hexdigest()
            buf.rename(root / "attachments" / f"{digest}.png")
            attachments.append({"id": str(9000 + n), "filename": filename, "sha256": digest,
                                "file": f"attachments/{digest}.png"})
        lines.append({"guild_id": G, "channel_id": "100", "channel_name": "card-ideas", "thread": None, "id": mid,
                      "timestamp": dm.iso(when), "author": {"id": author, "username": author, "global_name": None,
                                                           "bot": bot}, "content": content,
                      "attachments": attachments, "embeds": [], "referenced_message_id": None,
                      "jump_url": f"https://discord.com/channels/{G}/100/{mid}"})
        return mid

    t = 1600000000                                    # 2020-09-13
    post(t - 400 * 86400, jpeg(art(11)), bot=True)                       # a bot's post: never creation evidence
    post(t - 100 * 86400, art(77), "Beta_Baer.png", "beta bear")         # weak: a near file name and a caption
    post(t, jpeg(art(11)), content="new card: Alpha Wolf")
    revised = art(11).copy()
    revised.paste(art(55).crop((0, 200, 600, 500)), (0, 600))            # same art, new text box
    post(t + 30 * 86400, jpeg(revised))
    post(t + 60 * 86400, jpeg(art(12)), "Beta_Bear.png", author="croissant")
    post(t + 90 * 86400, None, content="alpha wolf feels overpowered, needs a nerf")
    (root / "messages" / "100.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    (root / "channels.json").write_text(json.dumps([{"id": "100", "name": "card-ideas", "type": 0}]))
    (root / "summary.json").write_text(json.dumps({"guild_id": G, "status": "done", "totals": {"messages": n}}))
    return root


def test_score_build_and_rulings(tmp_path, capsys):
    repo, root = make_cards_repo(tmp_path), make_archive(tmp_path)
    result = dm.score(str(root), str(repo), site="", versions="none", ocr=False, workers=1, log=lambda *a: None)
    cards = result["cards"]
    wolf = cards["alpha-wolf"]["discord"]
    assert wolf["first"]["date"].startswith("2020-09-13") and wolf["first"]["tier"] == "high"
    assert wolf["first"]["creation_wording"] and wolf["bot_posts"] == 1
    assert len(wolf["revisions"]) == 1 and wolf["fs_check"]["verdict"] == "discord_earlier"
    assert any("nerf" in note["excerpt"] for note in wolf["notes"])
    bear = cards["beta-bear"]["discord"]
    assert bear["first"]["date"].startswith("2020-11-12") and bear["earlier_ambiguous"] == ["A1"]
    assert bear["poster_vs_author"]["poster"] == "croissant" and cards["beta-bear"]["credited_in_file"] == "Croissant"
    assert cards["gamma-ray"]["discord"] is None
    entries = json.loads((root / "ambiguous.json").read_text())["entries"]
    assert [e["id"] for e in entries] == ["A1"] and entries[0]["affects_earliest"] == ["beta-bear"]
    page = (root / "ambiguous.html").read_text()
    assert 'id=A1' in page and "Beta Bear" in page and "attachments/" in page
    assert "Coverage" in (root / "REPORT.md").read_text()

    assert dm.rulings_main(str(root), ["A1 = Beta Bear"]) == 0
    ruled = json.loads((root / "matches.json").read_text())["cards"]["beta-bear"]["discord"]
    assert ruled["first"]["date"].startswith("2020-06-05") and ruled["tier"] == "ruled"
    assert dm.rulings_main(str(root), ["A1 = none"]) == 0
    again = json.loads((root / "matches.json").read_text())["cards"]["beta-bear"]["discord"]
    assert again["first"]["date"].startswith("2020-11-12")
    assert dm.rulings_main(str(root), ["A9 = Bob"]) == 1                      # no such entry
    stored = (root / "match" / "rulings.json").read_text()
    assert dm.rulings_main(str(root), ["A1 = Beta Bear, A9 = Bob"]) == 1       # one bad: nothing stored
    assert (root / "match" / "rulings.json").read_text() == stored
    dm.build(str(root), str(repo), log=lambda *a: None)
    assert [e["id"] for e in json.loads((root / "ambiguous.json").read_text())["entries"]] == ["A1"]   # stable

    script = root / "apply_rulings.py"
    env = {**os.environ, "DISCORD_MATCH_REPO": str(REPO)}
    listed = subprocess.run([sys.executable, "-I", str(script), "--list"], capture_output=True, text=True, env=env)
    assert listed.returncode == 0 and "A1 = none" in listed.stdout
