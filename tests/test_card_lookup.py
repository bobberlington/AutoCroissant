"""/query's name lookup: exact names, titles and aliases; renamed cards' old names (their archive's renamed_to); guesses
that say "Closest match"; and the autocomplete list. A small registry with the names that collided on 2026-10-24."""
from collections import defaultdict
from types import SimpleNamespace

import pytest

import commands.card_models as cm
import commands.query_card as qc
import commands.utils as utils

CARDS = {                                               # name -> (PNG path, printed title)
    "The Forbidden One": ("Creatures/Other/5 Stars/The_Forbidden_One.png", None),
    "Spawn of the Forbidden One": ("Auxiliary/Minions/Spawn_of_the_Forbidden_One.png", None),
    "Erasure": ("Items/Support/2 Stars/Erasure.png", None),
    "Spice": ("Items/Heal/1 Stars/Spice.png", None),
    "Light": ("Types/Light.png", None),
    "Grodus": ("Creatures/Mario/4 Stars/Grodus.png", None),
    "Bobbery": ("Creatures/Mario/3 Stars/Bobbery.png", None),
    "Rope": ("Items/Equip/1 Stars/Rope.png", None),
    "Electro Probe": ("Auxiliary/Minions/Electro_Probe.png", None),
    "Cortez": ("Creatures/Other/3 Stars/Cortez.png", None),
    "The Core": ("Field/2 Stars/The_Core.png", None),
    "Shadow Chopper": ("Creatures/Mario/3 Stars/Shadow_Chopper.png", None),
    "Palace of Shadow": ("Field/4 Stars/Palace_of_Shadow.png", None),
    "Tainted Isaac": ("Creatures/Isaac/3 Stars/Tainted_Isaac.png", None),
    "Dark Esau": ("Auxiliary/Minions/Dark_Esau.png", "Dark ESAU"),
    "Bolts and Screws": ("Items/Heal/1 Stars/Bolts_and_Screws.png", None),
    "Infected Crossroads": ("Field/2 Stars/Infected_Crossroads.png", None),
    "Forgotten Crossroads": ("Field/1 Stars/Forgotten_Crossroads.png", None),
    "Nebula Star": ("Items/Support/1 Stars/Nebula_Star.png", None),
    "Nebulous Precipice": ("Field/3 Stars/Nebulous_Precipice.png", None),
    "Regen": ("Auxiliary/Status Effects/Buffs/Regen.png", None),
    "Tainted Eden": ("Creatures/Isaac/4 Stars/Tainted_Eden.png", None),
    "Frostmourne": ("Items/Equip/4 Stars/Frostmourne.png", None),
    "The Freezer": ("Creatures/Other/3 Stars/The_Freezer.png", None),
    "Mini Doomer": ("Auxiliary/Minions/Mini_Doomer.png", None),
    "Grand Doomer": ("Creatures/Other/5 Stars/Grand_Doomer.png", None),
    "Tomorrow I'll": ("Items/Support/2 Stars/Tomorrow_I'll.png", None),
    "WX-78": ("Creatures/DST/3 Stars/WX-78.png", None),
    "Clot Black": ("Auxiliary/Minions/Clot_Black.png", "Black Clot"),
    "Sans": ("Creatures/Undertale/5 Stars/Sans.png", None),
    "Mettaton EX": ("Auxiliary/Creatures/Mettaton_EX.png", None),
    "Mettaton NEO": ("Auxiliary/Creatures/Mettaton_NEO.png", None),
    "Pot Full of Spiders!": ("Creatures/Crypt/2 Stars/Pot_Full_of_Spiders!.png", None),
    "Magolor": ("Creatures/Kirby/3 Stars/Magolor.png", None),
    "Indoctrination": ("Auxiliary/Items/Indoctrination.png", None),
    "Dogma": ("Creatures/Isaac/5 Stars/Dogma.png", None),
    "Panda": ("Creatures/Kirby/3 Stars/Panda.png", None),        # Types/Panda.png too: an ambiguous name
}
RENAMED = {                                             # archived name -> (renamed_to, printed title)
    "Chestnut": ("The Forbidden One", None), "Chestnutter": ("Spawn of the Forbidden One", None),
    "Sacrifice": ("Erasure", None), "Probe": ("Electro Probe", None), "Core": ("The Core", None),
    "Shadow Palace": ("Palace of Shadow", None), "Tainted Esau": ("Dark Esau", None), "TaintedEsau": ("Dark Esau", None),
    "Nuts and Bolts": ("Bolts and Screws", None), "Crossroads": ("Forgotten Crossroads", None),
    "Plan Ahead": ("Tomorrow I'll", None), "Wx78": ("WX-78", None), "Magalor": ("Magolor", None),
    "Dogma (item)": ("Zealotry", "Dogma"), "Zealotry": ("Indoctrination", None),         # a chain; Dogma is a card too
    "Old Gone": ("Long Gone", None),                    # renamed to a card that left too
}
ALIASES = {"sans": ["snas."]}                           # Card Edits.json names, keyed as that file (lowercase)


@pytest.fixture
def repo(monkeypatch):
    stats = {n: SimpleNamespace(name=n, display_name=t, hidden=False, duplicate_of=None, series=None)
             for n, (_, t) in CARDS.items()}
    old = defaultdict(list)
    for n, (to, t) in RENAMED.items():
        old[n].append(SimpleNamespace(name=n, display_name=t, archived={"at": 1.0, "renamed_to": to}))
    monkeypatch.setattr(cm.stats_db, "stats", stats)
    monkeypatch.setattr(cm.stats_db, "old_stats", old)
    monkeypatch.setattr(qc, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))    # no website ids
    repo = qc.CardRepository()
    repo._edit_names = ("sha", ALIASES)
    tree = {"tree": [{"path": p, "sha": "s"} for p, _ in CARDS.values()] + [{"path": "Types/Panda.png", "sha": "s"},
                                                                             {"path": qc.EDITS_PATH, "sha": "sha"}]}
    repo._fetch_tree = lambda name: (tree, "", 200, "")
    repo.populate_files()
    repo.get_card_url = lambda key: f"https://img/{key}"
    monkeypatch.setattr(qc, "card_repo", repo)
    utils.dispatch_queue.clear()
    yield repo
    utils.dispatch_queue.clear()


# ========================
# The lookup table
# ========================
EXACT, RENAMED_, GUESS = "exact", "renamed", "guess"


@pytest.mark.parametrize("query, key, how", [
    # exact: file names, any case, spaces or underscores, .png; printed titles; website aliases
    ("spice", "spice.png", EXACT), ("Light", "light.png", EXACT), ("the freezer", "the_freezer.png", EXACT),
    ("THE FREEZER", "the_freezer.png", EXACT), ("Mini_Doomer", "mini_doomer.png", EXACT),
    ("mini doomer.png", "mini_doomer.png", EXACT), ("black clot", "black_clot.png", EXACT), ("snas.", "snas..png", EXACT),
    ("frostmourne", "frostmourne.png", EXACT), ("wx-78", "wx-78.png", EXACT), ("dogma", "dogma.png", EXACT),
    ("nebulous precipice", "nebulous_precipice.png", EXACT), ("Dark ESAU", "dark_esau.png", EXACT),
    ("  the   freezer ", "the_freezer.png", EXACT), ("Spice.PNG", "spice.png", EXACT), ("sacrifice ", "erasure.png", RENAMED_),
    # old names: the card they became, along chains; a name of today wins (dogma above)
    ("chestnut", "the_forbidden_one.png", RENAMED_), ("Chestnut", "the_forbidden_one.png", RENAMED_),
    ("chestnutter", "spawn_of_the_forbidden_one.png", RENAMED_), ("sacrifice", "erasure.png", RENAMED_),
    ("probe", "electro_probe.png", RENAMED_), ("core", "the_core.png", RENAMED_),
    ("shadow palace", "palace_of_shadow.png", RENAMED_), ("tainted esau", "dark_esau.png", RENAMED_),
    ("taintedesau", "dark_esau.png", RENAMED_), ("nuts and bolts", "bolts_and_screws.png", RENAMED_),
    ("crossroads", "forgotten_crossroads.png", RENAMED_), ("plan ahead", "tomorrow_i'll.png", RENAMED_),
    ("wx78", "wx-78.png", RENAMED_), ("magalor", "magolor.png", RENAMED_), ("zealotry", "indoctrination.png", RENAMED_),
    ("dogma (item)", "indoctrination.png", RENAMED_),
    # guesses: a name or a word in it starting so (shortest first), then the closest spelling
    ("nebulous", "nebulous_precipice.png", GUESS), ("eden", "tainted_eden.png", GUESS),
    ("freezer", "the_freezer.png", GUESS), ("doomer", "mini_doomer.png", GUESS), ("frost", "frostmourne.png", GUESS),
    ("mettaton", "mettaton_ex.png", GUESS), ("pot full", "pot_full_of_spiders!.png", GUESS),
    ("forbidden", "the_forbidden_one.png", GUESS), ("crossroad", "infected_crossroads.png", GUESS),
    ("frostmorne", "frostmourne.png", GUESS), ("mini domer", "mini_doomer.png", GUESS),
    ("the freezr", "the_freezer.png", GUESS), ("grand domer", "grand_doomer.png", GUESS),
    ("magolr", "magolor.png", GUESS), ("spawn of the forbiden one", "spawn_of_the_forbidden_one.png", GUESS),
    ("tainted isac", "tainted_isaac.png", GUESS), ("bowser", "bobbery.png", GUESS), ("sanz", "sans.png", GUESS),
    ("spcie", "spice.png", GUESS), ("pand", "panda.png", GUESS), ("panda", "panda.png", EXACT),   # not types/panda
    # nothing: .png no longer lifts every score (luigi/light was 0.78, is 0.60: at the ratio is not above it)
    ("luigi", None, ""), ("goku", None, ""), ("pikachu", None, ""), ("xyz", None, ""), ("zz", None, ""),
    ("old gone", None, ""), ("long gone", None, ""), ("", None, ""),
])
def test_lookup(repo, capsys, query, key, how):
    assert repo.lookup(query)[0:2] == (key, how)
    qc.query_name(utils.make_simple_fake_interaction(1), query)
    sent = [kw for _, kw in utils.dispatch_queue]
    logged = capsys.readouterr().out                    # misses and guesses: one console line each
    assert (f"Search miss: /query {query!r}" in logged) == (key is None)
    assert ("Search guess: /query" in logged) == (how == GUESS)
    if key is None:
        assert [kw["content"] for kw in sent] == ["No card found!"]
        return
    url, text, embed = f"https://img/{key}", sent[0]["content"] or "", sent[0].get("embed")     # a printed title: an embed
    assert [kw["content"].startswith("Ambiguous name found") for kw in sent[1:]] == ([True] if key in repo.ambiguous_names else [])
    assert embed.image.url == url if embed else text.endswith(url)
    text = text if embed else text[:len(text) - len(url)].rstrip("\n")
    label = qc._label(key)
    assert text == {EXACT: "", RENAMED_: f"**{label}** (renamed from {repo.lookup(query)[2]})",
                    GUESS: f"Closest match: **{label}**"}[how]


def test_autocomplete(repo):
    """Names starting with the text first, then a word that does, then holding it; old names say what they became."""
    names = lambda text: [shown for shown, _ in repo.name_choices(text)]
    assert names("mett") == ["Mettaton EX", "Mettaton NEO"]
    assert names("doomer") == ["Mini Doomer", "Grand Doomer"]
    assert names("chestnut") == ["Chestnut (now The Forbidden One)", "Chestnutter (now Spawn of the Forbidden One)"]
    assert repo.name_choices("black") == [("Black Clot", "Black Clot"), ("Clot Black (Black Clot)", "Clot Black")]
    assert ("snas. (Sans)", "snas.") in repo.name_choices("sna")
    assert len(repo.name_choices("")) == 25 and len(repo.name_choices("o")) == 25


def test_slash_surface(repo, capsys, monkeypatch):
    """Card-name options autocomplete; every slash command logs one line (values only for searches)."""
    import asyncio
    import main
    monkeypatch.setattr(main, "card_repo", repo)
    cmds = {c.name: c for c in main.tree.get_commands()}
    for name, option in (("query", "query"), ("view_old_metadata", "query"), ("frankenstein", "cards")):
        assert {p.name: p for p in cmds[name].parameters}[option].autocomplete, name
    picked = asyncio.run(main.card_list_names(None, "spice, mini do"))
    assert [(c.name, c.value) for c in picked] == [("Mini Doomer", "spice, Mini Doomer")]
    inter = SimpleNamespace(type=main.InteractionType.application_command, user=SimpleNamespace(id=7), channel_id=9,
                            data={"name": "query_ability", "options": [{"name": "query", "value": "chestnut"},
                                                                       {"name": "howmany", "value": True}]})
    capsys.readouterr()                                 # main's import prints the token line
    asyncio.run(main.on_interaction(inter))
    assert capsys.readouterr().out == "Command /query_ability by 7 in 9: query='chestnut', howmany\n"
