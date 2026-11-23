"""/query's name lookup: exact names, titles and aliases; renamed cards' old names (their archive's renamed_to); guesses
that say "Closest match"; and the autocomplete list. A small registry with the names that collided on 2026-10-24."""
from collections import defaultdict
from types import SimpleNamespace

import pytest
from pandas import DataFrame

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
NOT_CARDS = ["Types/Panda.png", "MDW/Linked/Nebula_Star - NebulaStar.png", "MDW/Old Rules/Very Old Rulebook/13_Minions.png"]


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
    paths = [p for p, _ in CARDS.values()] + NOT_CARDS + [qc.EDITS_PATH]
    tree = {"tree": [{"path": p, "sha": "sha" if p == qc.EDITS_PATH else "s"} for p in paths]}
    repo._fetch_tree = lambda name: (tree, "", 200, "")
    repo.populate_files()
    repo.get_card_url = lambda key: f"https://img/{key}"
    monkeypatch.setattr(qc, "card_repo", repo)
    return repo


# ========================
# The lookup table
# ========================
EXACT, RENAMED_, GUESS = "exact", "renamed", "guess"


LOOKUPS = [
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
]


def test_lookup(repo, capsys):
    """The table: the key and how it was found, the reply, and one console line per miss or guess."""
    for query, key, how in LOOKUPS:
        utils.dispatch_queue.clear()
        assert repo.lookup(query)[0:2] == (key, how), query
        qc.query_name(utils.make_simple_fake_interaction(1), query)
        sent = [kw for _, kw in utils.dispatch_queue]
        logged = capsys.readouterr().out                # misses and guesses: one console line each
        assert (f"Search miss: /query {query!r}" in logged) == (key is None), query
        assert ("Search guess: /query" in logged) == (how == GUESS), query
        if key is None:
            assert [kw["content"] for kw in sent] == ["No card found!"], query
            continue
        url, text, embed = f"https://img/{key}", sent[0]["content"] or "", sent[0].get("embed")     # a printed title: an embed
        assert [kw["content"].startswith("Ambiguous name found") for kw in sent[1:]] == (
            [True] if repo.other_cards(key) else []), query
        assert embed.image.url == url if embed else text.endswith(url), query
        text = text if embed else text[:len(text) - len(url)].rstrip("\n")
        label = qc._label(key)
        assert text == {EXACT: "", RENAMED_: f"**{label}** (renamed from {repo.lookup(query)[2]})",
                        GUESS: f"Closest match: **{label}**"}[how], query


def test_autocomplete(repo):
    """Names starting with the text first, then a word that does, then holding it; old names say what they became."""
    names = lambda text: [shown for shown, _ in repo.name_choices(text)]
    assert names("mett") == ["Mettaton EX", "Mettaton NEO"]
    assert names("doomer") == ["Mini Doomer", "Grand Doomer"]
    assert names("chestnut") == ["Chestnut (now The Forbidden One)", "Chestnutter (now Spawn of the Forbidden One)"]
    assert repo.name_choices("black") == [("Black Clot", "Black Clot"), ("Clot Black (Black Clot)", "Clot Black")]
    assert ("snas. (Sans)", "snas.") in repo.name_choices("sna")
    assert len(repo.name_choices("")) == 25 and len(repo.name_choices("o")) == 25
    # then without spaces and punctuation; nothing: near spellings, one per card
    assert names("wx 78") == ["WX-78", "Wx78 (now WX-78)"] and names("pot full of spiders") == ["Pot Full of Spiders!"]
    assert names("magolr") == ["Magolor"] and names("mettatn") == ["Mettaton EX", "Mettaton NEO"]


def test_curly_apostrophes(repo):
    """A phone's curly apostrophes find the straight ones of the names: /query, as_of and the autocomplete."""
    for query in ("Tomorrow I\u2019ll", "tomorrow i\u2018ll"):
        assert repo.lookup(query)[0:2] == ("tomorrow_i'll.png", EXACT), query
        assert qc.history_match(query) == ("tomorrow_i'll.png", "Tomorrow I'll"), query
    assert repo.name_choices("tomorrow i\u2019") == repo.name_choices("tomorrow i'") != []


def test_guesses_cards_only(repo, monkeypatch):
    """Guesses pick cards, not linked art or old pages (exact names still find them); a chapter's title or label, or the
    start of a word of its title no card name holds ('minion', with a guess's ability pointer; not 'minons', 'inion'),
    its chapter (before a Types/ icon); a section's title that is no card's or icon's, its section; the autocomplete
    lists chapters for /query only."""
    monkeypatch.setattr(cm.stats_db, "rulebook", [
        {"id": "minions", "num": "15", "title": "Minions", "text": "", "sections": [[2, 4, "SHADOW", "shadow"], [4, 6, "TIPS:", "tips"]]},
        {"id": "light", "num": "22", "title": "Light", "text": "", "sections": [[2, 4, "🕐TIPS", "tips"], [4, 6, "ROPE", "rope"],
                                                                                [6, 8, "PANDA", "panda"]]}])
    assert repo.lookup("NebulaStar")[0:2] == ("nebula_star.png", GUESS)       # not the linked art's "_-_nebulastar"
    assert qc.history_match("NebulaStar") == ("nebula_star.png", "Nebula Star")
    assert repo.lookup("13 minions")[0:2] == ("13_minions.png", EXACT)
    assert repo.other_cards("panda.png") == []                                # Types/Panda.png: no note
    repo._populate_renames()                                                  # /query lists chapters, after cards
    assert repo.name_choices("minions", chapters=True) == [("15. Minions (rulebook chapter)", "15. Minions")]
    assert repo.name_choices("mini", chapters=True)[-1][1] == "15. Minions" and repo.name_choices("minions") == [
        ("Mini Doomer", "Mini Doomer")]
    for query in ("minions", " MINIONS ", "15. Minions"):
        utils.dispatch_queue.clear()
        qc.query_name(utils.make_simple_fake_interaction(1), query)
        assert [kw["content"] for _, kw in utils.dispatch_queue] == [
            f"**15. Minions** is a rulebook chapter: {qc.SITE_URL}/#/rules/minions"], query
    repo.cards_dff = DataFrame({"ability": ["summon a minion."]}, index=["Rope"])
    utils.dispatch_queue.clear()
    qc.query_name(utils.make_simple_fake_interaction(1), "minion")               # a word's start: a guess's pointer too
    assert [(kw["content"], kw.get("links")) for _, kw in utils.dispatch_queue] == [(
        f'**15. Minions** is a rulebook chapter: {qc.SITE_URL}/#/rules/minions · 1 card has "minion" in its ability: '
        "/query_ability minion", [('Cards with "minion"', f"{qc.SITE_URL}/#/cards?q=minion")])]
    for query in ("minons", "inion"):                                            # a typo, inside a word: no chapter
        utils.dispatch_queue.clear()
        qc.query_name(utils.make_simple_fake_interaction(1), query)
        assert "rulebook chapter" not in str(utils.dispatch_queue), query
    utils.dispatch_queue.clear()
    qc.query_name(utils.make_simple_fake_interaction(1), "light")                # Types/Light.png: the chapter
    assert [kw["content"] for _, kw in utils.dispatch_queue] == [
        f"**22. Light** is a rulebook chapter: {qc.SITE_URL}/#/rules/light"]
    sent = []
    for query in ("Shadow", "tips", "rope", "panda"):              # Shadow Chopper a guess; Rope a card, Panda an icon too
        utils.dispatch_queue.clear()
        qc.query_name(utils.make_simple_fake_interaction(1), query)
        sent.append([kw.get("content") for _, kw in utils.dispatch_queue])
    assert sent[:2] == [[f"**SHADOW** is a section of **15. Minions**: {qc.SITE_URL}/#/rules/minions/shadow"],
                        [f"**TIPS:** is a section of 2 rulebook chapters:\n- **15. Minions**: {qc.SITE_URL}/#/rules/minions/tips\n"
                         f"- **22. Light**: {qc.SITE_URL}/#/rules/light/tips"]]
    assert not any("section" in str(c) for c in sent[2] + sent[3])


def test_ban_list_by_its_names(repo, monkeypatch):
    """The website's Constructed ban list answers to its names ('ban list', 'banlist'; not 'bans'): /query, its groups'
    cards (one reason's together) and page; the autocomplete (/query only); /query_rulebook: its notes and every group,
    a group's heading that group's cards."""
    repo.banlist = {"notes": ["One of each item."], "groups": [
        {"title": "5 star creatures", "entries": [{"name": "Marx", "reason": "antifun", "id": "marx"},
                                                  {"name": "Baba", "reason": "combos", "id": "baba"},
                                                  {"name": "Fawful", "reason": "antifun", "id": "fawful"}]},
        {"title": "4 star items", "entries": [{"name": "Mimic", "reason": "", "id": None}]}]}
    shown = ("Constructed ban list (rulebook)", "Constructed ban list")
    assert repo.name_choices("ban list", chapters=True) == [shown] and shown not in repo.name_choices("ban list")
    sent = []
    for query in ("ban list", "BANLIST", "Constructed ban list", "bans"):
        utils.dispatch_queue.clear()
        qc.query_name(utils.make_simple_fake_interaction(1), query)
        sent.append([kw.get("content") for _, kw in utils.dispatch_queue])
    assert sent[:3] == [[f"**Constructed ban list** is in the rulebook: {qc.SITE_URL}/#/rules/banlist\n"
                         "**5 star creatures**: Marx, Fawful (antifun) · Baba (combos)\n**4 star items**: Mimic"]] * 3
    assert "banlist" not in str(sent[3])
    repo.rulebook_dff = DataFrame({"ability": ["Use an item."]}, index=["13. Items"])
    monkeypatch.setattr(qc, "render_type_tokens", lambda text, escape=False: text)
    utils.dispatch_queue.clear()
    for text in ("banlist", "5 Star Creatures"):
        qc.query_rulebook(utils.make_simple_fake_interaction(1), text)
    assert [kw["content"] for _, kw in utils.dispatch_queue] == [
        "'banlist' is on 1 line of the Constructed ban list:\n\n**Constructed ban list**\n- One of each item.\n\n"
        "**Constructed ban list · 5 star creatures**\n- Marx - antifun\n- Baba - combos\n- Fawful - antifun\n\n"
        "**Constructed ban list · 4 star items**\n- Mimic",
        "'5 Star Creatures' is on 1 line of the Constructed ban list:\n\n**Constructed ban list · 5 star creatures**\n"
        "- Marx - antifun\n- Baba - combos\n- Fawful - antifun"]


def test_guess_points_to_abilities(repo, monkeypatch):
    """A guess whose text is in abilities says how many, as /query_ability counts them, with a website search button
    (real uses: 'draw', 'assume' then typed into /query_ability), a rulebook section's title too; exact names and other
    guesses as before."""
    repo.cards_dff = DataFrame({"ability": ["gain frost\narmor.", "deal 2 damage. frost them.", "*eden*'s  frost dmg"]},
                               index=["Rope", "Spice", "Regen"])

    def reply(query):
        utils.dispatch_queue.clear()
        qc.query_name(utils.make_simple_fake_interaction(1), query)
        return utils.dispatch_queue[0][1]["content"], utils.dispatch_queue[0][1].get("links")

    assert reply("Frost") == ('Closest match: **Frostmourne** · 3 cards have "Frost" in their ability: /query_ability Frost',
                              [('Cards with "Frost"', f"{qc.SITE_URL}/#/cards?q=Frost")])
    assert reply("eden") == ('Closest match: **Tainted Eden** · 1 card has "eden" in its ability: /query_ability eden',
                             [('Cards with "eden"', f"{qc.SITE_URL}/#/cards?q=eden")])
    assert repo.ability_hits("FROST  DAMAGE") == repo.ability_hits("frost armor") == 1 and repo.ability_hits("heal") == 0
    assert reply("frostmourne") == (None, None) and reply("freezer") == ("Closest match: **The Freezer**", None)
    monkeypatch.setattr(cm.stats_db, "rulebook", [{"id": "ice", "num": "9", "title": "Ice", "text": "",
                                                   "sections": [[2, 4, "FROST ARMOR", "frost-armor"]]}])
    assert reply("frost armor") == (f'**FROST ARMOR** is a section of **9. Ice**: {qc.SITE_URL}/#/rules/ice/frost-armor · 1 card '
                                    'has "frost armor" in its ability: /query_ability frost armor',
                                    [('Cards with "frost armor"', f"{qc.SITE_URL}/#/cards?q=frost%20armor")])


def test_series_name_gives_its_cards(repo, capsys):
    """A series' or subseries' name (as a search term takes it) says how many cards, with a website search button; a card
    named so is still the card (OMORI); a typo still guesses."""
    repo.cards_dff = DataFrame({"ability": [""] * 4, "series": ["The Binding of Isaac", "The Binding of Isaac", "Mario", "Mario"],
                                "subseries": [None, "Dogma", "Paper Mario: The Thousand-Year Door", None]},
                               index=["Tainted Isaac", "Dogma", "Grodus", "Bobbery"])

    def reply(query):
        utils.dispatch_queue.clear()
        qc.query_name(utils.make_simple_fake_interaction(1), query)
        return utils.dispatch_queue[0][1]["content"], utils.dispatch_queue[0][1].get("links")

    assert reply("binding of isaac") == ("**The Binding of Isaac** is a series: 2 cards", [
        ("Its cards on the website", f"{qc.SITE_URL}/#/cards?q=binding%20of%20isaac")])
    assert "Search series: /query 'binding of isaac'" in capsys.readouterr().out
    assert reply("Thousand-Year Door")[0] == "**Paper Mario: The Thousand-Year Door** is a subseries of Mario: 1 card"
    assert reply("dogma")[1] is None and reply("magolr") == ("Closest match: **Magolor**", None)
    # /query's autocomplete lists them as it does chapters: after the cards unless typed whole; never a card's name
    repo._populate_renames()
    names = lambda text: [shown for shown, _ in repo.name_choices(text, chapters=True)]
    assert repo.name_choices("Mario", chapters=True)[0] == ("Mario (series, 2 cards)", "Mario")
    assert names("isaac") == ["Tainted Isaac", "The Binding of Isaac (series, 2 cards)"]
    assert names("thousand") == ["Paper Mario: The Thousand-Year Door (subseries of Mario, 1 card)"]
    assert names("dogma") == ["Dogma"] and repo.name_choices("mario") == []
    for name, (path, _) in CARDS.items():                       # each card's name: its card first
        if not path.startswith(qc.NOT_CARDS):
            assert repo.name_choices(name, chapters=True)[0][1].lower() == name.lower(), name


def test_morgue_and_unfinished_cards_say_so(monkeypatch):
    """A Morgue card (cards.json morgue) says it is retired, with its page; an MDW/Unfinished/ one says it is not in play,
    its archived name finding it (Pandiety); a live card and cardback.png as before."""
    at = 1792658876.0                                   # 2026-10-22, Pacific
    moved = lambda psd, **kw: [SimpleNamespace(display_name=None, archived={"at": at, "commit": "b633073" + "0" * 33,
                                                                            "moved_to": psd, **kw})]
    monkeypatch.setattr(cm.stats_db, "stats", {"Spice": SimpleNamespace(name="Spice", series=None, display_name=None, hidden=False,
                                                                  duplicate_of=None)})
    monkeypatch.setattr(cm.stats_db, "old_stats", defaultdict(list, {
        "White Palace": moved("MDW/The Morgue/White_Palace.psd"),
        "Pandiety": moved("MDW/Unfinished/6 Stars/Pandeity.psd", backfill=True)}))
    site = {"cards": [{"id": "spice", "png": "Items/Heal/1 Stars/Spice.png"}],
            "morgue": [{"id": "white-palace", "png": "MDW/The Morgue/White_Palace.png", "retired": at}]}
    monkeypatch.setattr(qc, "get", lambda *a, **k: SimpleNamespace(status_code=200, headers={}, json=lambda: site,
                                                                   raise_for_status=lambda: None))
    paths = ["Items/Heal/1 Stars/Spice.png", "MDW/The Morgue/White_Palace.png", "MDW/The Morgue/cardback.png",
             "MDW/Unfinished/6 Stars/Pandeity.png"]
    repo = qc.CardRepository()
    repo._fetch_tree = lambda name: ({"tree": [{"path": p, "sha": "s"} for p in paths]}, "", 200, "")
    repo.populate_files()
    repo.get_card_url = lambda key: f"https://img/{key}"
    monkeypatch.setattr(qc, "card_repo", repo)
    page = f"{qc.SITE_URL}/c/%s/"
    for query, content, links in [
            ("white palace", "**White Palace** is retired: in MDW/The Morgue since 2026-10-22, out of play. Archived on "
             "2026-10-22 (PSD moved to MDW/The Morgue in b633073).\nhttps://img/white_palace.png",
             [("Card page", page % "white-palace")]),
            ("Pandiety", "**Pandeity** (renamed from Pandiety) is unfinished (MDW/Unfinished/6 Stars): not in play. "
             "Archived by 2026-10-22 (PSD moved to MDW/Unfinished/6 Stars in b633073).\nhttps://img/pandeity.png", None),
            ("spice", "https://img/spice.png", [("Card page", page % "spice")]),
            ("cardback", "https://img/cardback.png", None)]:
        utils.dispatch_queue.clear()
        qc.query_name(utils.make_simple_fake_interaction(1), query)
        assert [(kw["content"] or kw["embed"].image.url, kw.get("links")) for _, kw in utils.dispatch_queue] == \
            [(content, links)], query
    assert repo.name_choices("pandi") == []                      # not offered
    utils.dispatch_queue.clear()


def test_site_and_tts_pictures_are_no_cards(monkeypatch):
    """The website's own PNGs (site/favicon: favicons, header logo) and TTS/ ones are no cards: no exact name, no guess
    ('logo' was logo-mask); MDW/ and Types/ files are still found by exact name."""
    monkeypatch.setattr(cm.stats_db, "stats", {})
    monkeypatch.setattr(cm.stats_db, "old_stats", defaultdict(list))
    monkeypatch.setattr(qc, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    paths = ["Items/Heal/1 Stars/Spice.png", "Types/Light.png", "MDW/Assets/Markers/Red.png", "TTS/Rulebook/page_1.png",
             *(f"site/favicon/{n}.png" for n in ("favicon-16", "favicon-32", "apple-touch-icon", "logo-mask"))]
    repo = qc.CardRepository()
    repo._fetch_tree = lambda name: ({"tree": [{"path": p, "sha": "s"} for p in paths]}, "", 200, "")
    repo.populate_files()
    assert sorted(repo.git_files.values()) == sorted(paths[:3]) and "site" not in repo.tops
    assert [repo.lookup(q)[0] for q in ("logo", "apple", "favicon", "favicon-16", "logo-mask", "page 1")] == [None] * 6
    assert [repo.lookup(q)[0:2] for q in ("spice", "light", "red")] == [
        ("spice.png", "exact"), ("light.png", "exact"), ("red.png", "exact")]


def test_folders_and_paths(repo):
    """Guesses score names without their folder (creatures/panda, there for Types/Panda.png, took every 'creatures ...');
    a typed path (a top folder or .png) finds its file name, exact only in its own folder, never an icon from another;
    slash phrases stay misses; folder keys still exact; a miss in abilities points to them as a guess does."""
    repo.cards_dff = DataFrame({"ability": ["your creatures and stats rise."]}, index=["Spice"])
    assert [repo.lookup(q)[0] for q in ("creatures and stats", "your creatures", "creatures/spi")] == [None, None, "spice.png"]
    assert [repo.lookup(q)[0:2] for q in ("Items/Rope.png", "Creatures/Rope.png", "Items/Heal/Rope.png", "items/rop",
                                          "creatures/panda", "Types/Panda.png", "types/light", "dark/light",
                                          "creatures/light.png")] == [
        ("rope.png", EXACT), ("rope.png", GUESS), ("rope.png", GUESS), ("rope.png", GUESS), ("creatures/panda.png", EXACT),
        ("types/panda.png", EXACT), ("light.png", EXACT), (None, ""), (None, "")]
    utils.dispatch_queue.clear()
    qc.query_name(utils.make_simple_fake_interaction(1), "creatures and stats")
    assert [(kw["content"], kw["links"]) for _, kw in utils.dispatch_queue] == [(
        'No card found! · 1 card has "creatures and stats" in its ability: /query_ability creatures and stats',
        [('Cards with "creatures and stats"', f"{qc.SITE_URL}/#/cards?q=creatures%20and%20stats")])]


def test_lost_morgue_cards_answer_as_such(repo, monkeypatch):
    """A Morgue card lost before the repo (cards.json morgue, no PNG): its picture, page and the website's words, before
    a guess; a live card's name or typo, a renamed card's old name, a series and an in-joke's live card win; the
    autocomplete never offers one (as retired cards)."""
    entry = lambda name, i, **kw: {"id": i, "name": name, "png": None, "lost": "joke",
                                   "pic": f"history/_morgue/{i}/O.webp", **kw}
    raw = "https://raw.githubusercontent.com/r/h/abc/_morgue/void-sentinel/P.webp"      # history branch-hosted
    site = {"cards": [], "morgue": [
        entry("Chestnut The Forbidden One", "chestnut", first="2020-10-14", firstCh="ideas-discussion",
              also=[["2020-10-15", "meme-cards"]]),                    # one picture posted again (owner 11-21)
        entry("Void Sentinel", "void-sentinel", lost="played", first="2020-07-27", firstCh="update-01-port", pic=raw),
        entry("M'uru", "m-uru", lost="played", inGit=True, pic=None), entry("Magolor", "magolor-2"),
        entry("Sacrifice", "sacrifice"), entry("Mario", "mario"), entry("Rope", "rope-morgue", live="rope"),
        entry("Spider Queen", "spider-queen", lost="idea", first="2019-05-10", firstCh="general-discussion")]}
    monkeypatch.setattr(qc, "get", lambda *a, **k: SimpleNamespace(status_code=200, headers={}, json=lambda: site,
                                                                   raise_for_status=lambda: None))
    typed = ("forbidden one", "chestnut the forbiden", "void", "muru", "mario")
    offered = {text: repo.name_choices(text, chapters=True) for text in typed}
    repo._populate_site_ids()
    repo.cards_dff = DataFrame({"ability": [""], "series": ["Mario"]}, index=["Grodus"])
    page = f"{qc.SITE_URL}/c/%s/"
    for query, content, title, image, link in [
            ("chestnut the forbidden one", "**Chestnut The Forbidden One** is in The Morgue: Joke card · first posted "
             "2020-10-14 in #ideas-discussion · also posted 2020-10-15 in #meme-cards; out of play.",
             "Chestnut The Forbidden One",
             f"{qc.SITE_URL}/history/_morgue/chestnut/O.webp", page % "chestnut"),
            ("void sentinel", "**Void Sentinel** is in The Morgue: Played, lost before git · first posted 2020-07-27 "
             "in #update-01-port; out of play.", "Void Sentinel", raw, page % "void-sentinel"),
            ("MURU", "**M'uru** is in The Morgue: Played, lost before this repo; out of play.", "M'uru", None,
             page % "m-uru"),
            ("spider queen", "**Spider Queen** is in The Morgue: Idea, never played · first posted 2019-05-10 in "
             "#general-discussion; out of play.", "Spider Queen", f"{qc.SITE_URL}/history/_morgue/spider-queen/O.webp",
             page % "spider-queen"),
            ("Magolor", None, "Magolor", "https://img/magolor.png", None),
            ("Sacrifice", "**Erasure** (renamed from Sacrifice)", "Erasure", "https://img/erasure.png", None),
            ("Rope", None, "Rope", "https://img/rope.png", None),
            ("magolr", "Closest match: **Magolor**", "Magolor", "https://img/magolor.png", None)]:
        utils.dispatch_queue.clear()
        qc.query_name(utils.make_simple_fake_interaction(1), query)
        [(_, kw)] = utils.dispatch_queue
        embed = kw["embed"]
        assert (kw["content"], embed.title, embed.image.url, embed.url, kw.get("links")) == \
            (content, title, image, link, [("Card page", link)] if link else None), query
    utils.dispatch_queue.clear()
    qc.query_name(utils.make_simple_fake_interaction(1), "mario")
    assert utils.dispatch_queue[0][1]["content"] == "**Mario** is a series: 1 card"
    assert {text: repo.name_choices(text, chapters=True) for text in typed} == offered
    assert offered["chestnut the forbiden"] and offered["muru"] == []
    utils.dispatch_queue.clear()
