"""/query_ability's query language on a table of website-shaped cards, and /query_rulebook's lines."""

import json
import re
from subprocess import run as run_process

import pytest

import commands.card_models as cm
import commands.query_card as qc
from conftest import SITE_SRC


# website-shaped cards: name (as shown), aliases, ability, fields
CARDS = [
    dict(name="Kirby", type="creature", series="kirby", hp=5, defense=3, atk=4, spd=6, stars=3, types=["wind"],
         ability="Deal 2 DMG to the enemy."),
    dict(name="Meta Knight", type="creature", series="kirby", hp=7, defense=5, atk=7, spd=8, stars=4,
         aliases=["mk", "Galaxia Knight"], ability="Deals double damage.\nDie at the end of the turn."),
    dict(name="sans.", file="Sans", type="creature", series="undertale", hp=1, defense=1, atk=1, spd=9, stars=5,
         ability="Dodge every attack. +1 SPD."),
    dict(name="King Dedede", type="creature", series="kirby", subseries="Kirby's Return to Dream Land", hp=4, defense=4,
         atk=4, spd=2, stars=4, ability="Hammer time."),
    dict(name="Mettaton EX", file="Mettaton", type="creatures", hp=6, defense=6, atk=6, spd=6, ability="Gain +2 DEF."),
    dict(name="Fire Bomb", type="item", subtype="attack", stars=2, ability="Deal 3 damage to a [fire] creature."),
    dict(name="Panda Shield", type="item", subtype="equip", stars=1, aliases=["buckler"], ability="+1 defense."),
    dict(name="Bloon", type="minion", hp=2, defense=1, atk=1, spd=3, ability="(2U)(OYT) Pop: heal 2 health."),
    dict(name="Burning", type="debuff", stars=1, ability="Burning: 3 DMG, can't affect [fire]. On the PF: 1 CDMG, lose 1 SP."),
    dict(name="Power Ring", type="item", series="mario", subseries="Mario & Luigi: Superstar Saga",
         ability="+1 ATK to every friendly minion."),
    dict(name="Enfeebled", type="debuff", ability="-1 ATK."),
    dict(name="Mom's Knife", type="item", series="The Binding of Isaac", ability="Throw it."),
    dict(name="Twin Emperors", type="creature", hp=7, defense=4, atk=5, spd=6, stars=4, ability="Two emperors as one.",
         creatures=[(7, 4, 5, 6), (7, 10, 4, 4)]),                # creature 1 is the card's stats
]
ERROR = "error"                                          # the bot answers with an error

QUERIES = [                                              # query, the cards found
    ("damage", {"Meta Knight", "Fire Bomb", "Kirby", "Burning"}),
    ("dmg", {"Kirby", "Burning", "Meta Knight", "Fire Bomb"}),
    ("defense", {"Panda Shield", "Mettaton EX"}),
    ("health", {"Bloon"}), ("speed", {"sans."}),
    ("counter damage", {"Burning"}), ("primary field: 1 cdmg, lose 1 star points", {"Burning"}),   # SP, PF, CDMG
    ('"damage"', {"Meta Knight", "Fire Bomb"}), ('"(2u)"', {"Bloon"}),
    ("!type==creature", {"Mettaton EX", "Fire Bomb", "Panda Shield", "Bloon", "Burning", "Power Ring", "Enfeebled",
                         "Mom's Knife"}),
    ("type==creature", {"Kirby", "Meta Knight", "sans.", "King Dedede", "Twin Emperors"}),
    ("type~creature", {"Kirby", "Meta Knight", "sans.", "King Dedede", "Mettaton EX", "Twin Emperors"}),
    ("hp>5", {"Meta Knight", "Mettaton EX", "Twin Emperors"}), ("hp > 5", {"Meta Knight", "Mettaton EX", "Twin Emperors"}),
    ("total>=24", {"Meta Knight", "Mettaton EX", "Twin Emperors"}), ("spd>=8", {"Meta Knight", "sans."}),
    ("stars<=2", {"Fire Bomb", "Panda Shield", "Burning"}),
    ("fire", {"Fire Bomb", "Burning"}), ("dedede", {"King Dedede"}), ("mk", {"Meta Knight"}),
    ("sans.", {"sans."}), ("name~knight", {"Meta Knight"}), ("types~wind", {"Kirby"}),
    ("types==wind", {"Kirby"}),
    ("buckler", {"Panda Shield"}), ("alias~mk", {"Meta Knight"}), ("aliases==buckler", {"Panda Shield"}),
    ("alias==mettaton", {"Mettaton EX"}), ("has:alias", {"Meta Knight", "Panda Shield", "Mettaton EX"}),   # the file name too
    ("display_name~sans", {"sans."}),  # any column of the bot's
    ("+1 atk", {"Power Ring"}), ("+1 atk to every friendly minon.", {"Power Ring"}),   # near text only when none has it
    ("!+1 atk to every friendly minon.", {c["name"] for c in CARDS}),     # inside !, | and &: exact text (search.js)
    ("fire | +1 atk to every friendly minon.", {"Fire Bomb", "Burning"}),
    ("+1 atk to every friendly minon. & type==item", set()),
    ("die & series==kirby", {"Meta Knight"}), ("fire | hp>6", {"Fire Bomb", "Burning", "Meta Knight", "Twin Emperors"}),
    ("!die", {c["name"] for c in CARDS} - {"Meta Knight"}),
    ("series==kirby & !hp>5", {"Kirby", "King Dedede"}), ("subtype==equip", {"Panda Shield"}),
    ("atack>3", ERROR), ("total~5", ERROR), ("hp>=x", ERROR), ("type>3", ERROR),
    ("type!=creature", ERROR),
    ("fire || ice", {"Fire Bomb", "Burning"}), ("fire |", {"Fire Bomb", "Burning"}),   # blank parts dropped (search.js)
    ("|", ERROR), ("&", ERROR), ("series~", ERROR), ("types~", ERROR),
    ('"atack>3"', set()),
    ("can\u2019t", {"Burning"}), ("\u201cdeal 2\u201d", {"Kirby"}),        # a phone's curly quotes
    ("series==\u201ckirby\u201d & \u201cdie\u201d", {"Meta Knight"}),
]


def _repo(cards):
    stats = {}
    for c in cards:
        name = c.get("file", c["name"])
        stats[name] = cm.CardInfo(name=name, card_type=c["type"], path=f"X/{name}.psd", ability=c["ability"],
                                  stars=c.get("stars"), series=c.get("series"), subseries=c.get("subseries"),
                                  subtype=c.get("subtype"),
                                  types=c.get("types", []), display_name=c["name"] if "file" in c else None,
                                  stats=cm.CardStats(c.get("hp", -1), c.get("defense", -1), c.get("atk", -1), c.get("spd", -1)),
                                  creatures=[cm.CardStats(*x) for x in c["creatures"]] if "creatures" in c else None)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cm.stats_db, "stats", stats)
        repo = qc.CardRepository()
        repo._edit_names = ("sha", {qc.CardAuthors.key(c.get("file", c["name"])): c["aliases"]   # as Card Edits.json
                                    for c in cards if c.get("aliases")})
        repo.prep_dataframes()
    return repo


@pytest.fixture(scope="module")
def repo():
    return _repo(CARDS)


def _shown(repo, names) -> set:
    titles = repo.cards_dff["display_name"] if "display_name" in repo.cards_dff else {}
    return {titles[n] if isinstance(titles.get(n), str) else n for n in names}


def test_bot(repo):
    """The table: each query's cards, or a QueryError."""
    for query, bot in QUERIES:
        if bot == ERROR:
            with pytest.raises(qc.QueryError):
                repo.ability_search_engine(query)
            continue
        assert _shown(repo, repo.ability_search_engine(query).index) == bot, query


def test_website_synonyms_are_the_bots():
    """search.js's SYNONYMS is the bot's table."""
    js = (SITE_SRC / "site" / "assets" / "search.js").read_text(encoding="utf-8")
    table = re.search(r"const SYNONYMS = \{(.*?)\};", js, re.S).group(1)
    assert dict(re.findall(r'"?([a-z][a-z ]*)"?: "([a-z ]+)"', table)) == qc.SYNONYMS


def test_plain_terms_find_series_and_bare_names(repo, tmp_path):
    """A plain term also finds a series' or subseries' cards and a name typed without its punctuation, series== and
    series~ a subseries too, a known field's lone = is ==, a name typed with & finds its cards (& inside quotes is
    text), alias== any one alias and a 'single-quoted' term is whole words too, on the bot and the website."""
    rows = [("kirby", {"Kirby", "Meta Knight", "King Dedede"}), ("binding of isaac", {"Mom's Knife"}),   # without The
            ("moms knife", {"Mom's Knife"}), ("msk", set()),           # 3 letters or fewer: a whole name only
            ("kirby's return to dream land", {"King Dedede"}), ("series==kirby's return to dream land", {"King Dedede"}),
            ("series~dream", {"King Dedede"}), ("series=undertale", {"sans."}), ("hp = 6", {"Mettaton EX"}),
            ("mario & luigi: superstar saga", {"Power Ring"}), ("mario & luigi", {"Power Ring"}),
            ('subseries~"mario & luigi"', {"Power Ring"}), ("alias==mk", {"Meta Knight"}),   # any one alias
            ("alias==galaxia knight", {"Meta Knight"}),
            ("'damage' & !'die'", {"Fire Bomb"}), ("‘can't affect’", {"Burning"}),   # 'single' quotes: "double"
            ("'s", {"Mom's Knife"}), ("'kirby'", {"Kirby", "Meta Knight", "King Dedede"}),   # still a series
            ("'moms knife'", {"Mom's Knife"})]
    for query, cards in rows:
        assert _shown(repo, repo.ability_search_engine(query).index) == cards, query
    assert _website(tmp_path, [q for q, _ in rows]) == [cards for _, cards in rows]


def test_any_creature_meets_a_stat(repo, tmp_path):
    """A card with several creatures meets a stat condition when any one of its creatures does (total per creature),
    on the bot and the website; ! on the whole card."""
    rows = [("def>=10", {"Twin Emperors"}), ("total>=25", {"Meta Knight", "Twin Emperors"}),
            ("spd<=4", {"King Dedede", "Bloon", "Twin Emperors"}), ("!def>=10", {c["name"] for c in CARDS} - {"Twin Emperors"})]
    for query, cards in rows:
        assert _shown(repo, repo.ability_search_engine(query).index) == cards, query
    assert _website(tmp_path, [q for q, _ in rows]) == [cards for _, cards in rows]


def _website(tmp_path, queries, cards=CARDS) -> list:
    """search.js on cards as cards.json has them: each query's card names."""
    stats = lambda x: dict(zip(("hp", "def", "atk", "spd"), x))
    keys = ("name", "aliases", "ability", "type", "subtype", "types", "series", "subseries", "hp", "atk", "spd", "stars")
    shaped = [{**{k: c.get(k) for k in keys}, "def": c.get("defense"),
               "creatures": [stats(x) for x in c.get("creatures", ())]} for c in cards]
    script = tmp_path / "search.mjs"
    script.write_text(f"""const S = await import({json.dumps((SITE_SRC / "site" / "assets" / "search.js").as_uri())});
const cards = {json.dumps(shaped)};
process.stdout.write(JSON.stringify({json.dumps(queries)}.map(q => cards.filter(S.compile(q)).map(c => c.name))));""",
                      encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return [set(names) for names in json.loads(done.stdout)]


# the 9 items printing two item-type symbols (data 903e6fd1): kind, subtype (folder or left-most icon), icons
DUAL = [("Bolts and Screws", "item", "heal", ["food", "infliction"]), ("Cake Mix", "item", "heal", ["attack", "food"]),
        ("Hadal Stew", "item", "heal", ["food", "infliction"]), ("Salt", "aux item", "heal", ["food", "infliction"]),
        ("Inconspicuous Cake", "aux item", "heal", ["food", "infliction"]), ("Pie", "item", "attack", ["attack", "food"]),
        ("Magnum Opus", "aux item", "active", ["active", "field"]), ("Life Shroom", "item", "heal", ["food", "fragile equip"]),
        ("Electric Dart", "nme", "attack", ["nme", "attack", "food"])]


def test_items_count_at_each_item_type(tmp_path):
    """An item printing two item-type symbols counts at both (as TTS's decks), a field card's own icon at none: the bot
    and search.js subtype==, grid.js's Item type exclusion; one map (card_models, grid.js)."""
    cards = [dict(name=n, type=k, subtype=s, types=t, ability="Use it.") for n, k, s, t in DUAL] + [
        dict(name="Nexus", type="field", types=["field"], ability="A field."),
        dict(name="Potion", type="item", subtype="heal", types=["food"], ability="Heal.")]
    rows = [("subtype==infliction", {"Bolts and Screws", "Hadal Stew", "Inconspicuous Cake", "Salt"}),
            ("subtype==attack", {"Cake Mix", "Pie", "Electric Dart"}), ("subtype~equip", {"Life Shroom"}),
            ("subtype==field", {"Magnum Opus"}), ("!subtype==heal", {"Magnum Opus", "Nexus"})]
    repo = _repo(cards)
    assert [set(repo.ability_search_engine(q).index) for q, _ in rows] == [names for _, names in rows]
    assert _website(tmp_path, [q for q, _ in rows], cards) == [names for _, names in rows]
    script = tmp_path / "grid.mjs"
    script.write_text(f"""const G = await import({json.dumps((SITE_SRC / "site" / "assets" / "grid.js").as_uri())});
const cards = {json.dumps(cards)};
const kept = G.dropExcluded(cards, new URLSearchParams("xsub=infliction,equip")).map(c => c.name);
process.stdout.write(JSON.stringify([G.ICON_ITEM_TYPES, kept]));""",
                      encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [cm.AUX_ITEM_TYPES, ["Cake Mix", "Pie", "Magnum Opus", "Electric Dart", "Nexus",
                                                           "Potion"]]


def test_space_finds_line_break(repo, monkeypatch):
    """Text typed on one line finds an ability across its line break by the text, not the near-text fallback."""
    monkeypatch.setattr(qc.CardRepository, "_fuzzy_labels", lambda self, ability: pytest.fail(ability))
    assert list(repo.ability_search_engine("double damage.  die at").index) == ["Meta Knight"]
    assert list(repo.ability_search_engine('"damage. die"').index) == ["Meta Knight"]


def test_old_names_and_second_pictures(monkeypatch):
    """A text term finds a card by a name renamed into it (first, as search.js exactFirst), and a second picture by its
    own name, listed itself as the website does; never by its ability (counted once)."""
    card = lambda name, path, **kw: cm.CardInfo(name=name, card_type="item", path=f"X/{path}.psd", **kw)
    monkeypatch.setattr(cm.stats_db, "stats", {
        "Magolor": card("Magolor", "Magolor", ability="Float."),
        "Kirby": card("Kirby", "Kirby", ability="Beat Magalor."),
        "A Snack": card("A Snack", "A_Snack", ability="Heal 3 HP."),
        "A Snack2": card("A Snack2", "A_Snack", ability="Heal 3 HP.", display_name="Breakfast", duplicate_of="A Snack",
                         variant_of="A Snack")})
    monkeypatch.setattr(cm.stats_db, "old_stats", {"Magalor": [card("Magalor", "Magalor", archived={"renamed_to": "Magolor"})]})
    repo = qc.CardRepository()
    repo.prep_dataframes()
    monkeypatch.setattr(qc, "card_repo", repo)
    find = lambda q: list(repo.ability_search_engine(q).index)
    assert [n for n, _ in qc._site_order(repo.ability_search_engine("magalor"), "magalor")] == ["Magolor", "Kirby"]
    assert find("breakfast") == find('"breakfast"') == find("break-fast") == ["A Snack2"]
    assert find("heal 3 hp") == ["A Snack"]


def test_near_matches_say_so(repo, monkeypatch):
    """No card has the text: the near abilities, closest first, called near (reply and howmany)."""
    import commands.card_grid as cg
    import commands.utils as utils
    monkeypatch.setattr(qc, "card_repo", repo)
    monkeypatch.setattr(cg, "grids", lambda cards: [b"jpg"])
    query = "Deal 3 damage to the enemy."                 # A-Z would be Fire Bomb, Kirby, Meta Knight
    qc.query_ability(utils.make_simple_fake_interaction(1), query)
    qc.query_ability_num_occur(utils.make_simple_fake_interaction(1), query)
    out = [kw["content"] for _, kw in utils.dispatch_queue]
    assert out[0].split("\n")[:4] == [f"No card has the text '{query}'; 3 cards have a similar ability, closest first:",
                                      "**1.** Kirby", "**2.** Fire Bomb", "**3.** Meta Knight"]
    assert out[-1] == f"No card has '{query}'; 3 cards have a similar ability."


def test_rulebook_lines(monkeypatch):
    """/query_rulebook: the matching lines under their page names (word starts, then most matches), whole sections per
    message; with db/rulebook.json's sections, under them: a section titled the text brings its first lines, then every
    later line holding it, "…" only where lines are left out; the page's "# Title" is no line."""
    import commands.utils as utils
    from pandas import DataFrame
    pages = {"08 Global Turn": "THE GLOBAL TURN\nEach slot takes its turn.\nThen the next turn.",
             "02 Abbreviations": "(OYT) = On your turn.\n- (BOT) = Beginning of turn.\n(CD) = Cooldown.",
             "30 Long": "x " * 300 + "turn " + "y " * 300, "40 None": "Nothing here.", "50 Bricks": "1 per global\nturn."}
    repo = qc.CardRepository()
    repo.rulebook_dff = DataFrame({"ability": list(pages.values())}, index=list(pages))
    monkeypatch.setattr(qc, "card_repo", repo)
    monkeypatch.setattr(qc, "render_type_tokens", lambda text, escape=False: text)
    found = qc.rulebook_lines(repo.rulebook_search_engine("Turn"), "Turn")
    assert [(n, count) for n, _, count in found] == [("08 Global Turn", 3), ("02 Abbreviations", 2), ("30 Long", 1),
                                                     ("50 Bricks", 1)]
    assert found[1][1] == [("", ["(OYT) = On your turn.", "(BOT) = Beginning of turn."])]   # its own bullet dropped
    long = found[2][1][0][1][0]
    assert len(long) <= qc.SNIPPET + 2 and "turn" in long and long.startswith("…")
    assert qc._bold("A <:defensive_minion:12> minion, <a:x_1:12> 1", "minion") == "A <:defensive_minion:12> **minion**, <a:x_1:12> 1"
    assert qc._bold("<:x_1:12> 1", "1") == "<:x_1:12> **1**"                                   # emoji markup kept whole
    assert qc.rulebook_lines(repo.rulebook_search_engine("global  turn"), "global turn") == [
        ("08 Global Turn", [("", ["THE GLOBAL TURN"])], 1), ("50 Bricks", [("", ["1 per global turn."])], 1)]   # over a line break
    qc.query_rulebook(utils.make_simple_fake_interaction(1), "turn")
    sent = [kw for _, kw in utils.dispatch_queue]
    utils.dispatch_queue.clear()
    assert len(sent) == 1 and "links" in sent[0]
    assert sent[0]["content"].startswith("'turn' is on 7 lines of 4 rulebook chapters:\n\n**08 Global Turn**\n- THE GLOBAL **TURN**")
    repo.rulebook_dff = DataFrame({"ability": ["During a turn.\nAll during it.", "The ring (1CD).", "It's \u201cfree\u201d.",
                                               'A "sneak attack".']}, index=["A", "B", "C", "D"])
    assert [n for n, *_ in qc.rulebook_lines(repo.rulebook_search_engine("ring"), "ring")] == ["B", "A"]   # word starts first
    for text in ("it\u2019s", "\u201cfree\u201d", '"free"', "\u201csneak attack\u201d"):   # either quotes find both
        qc.query_rulebook(utils.make_simple_fake_interaction(1), text)
    sent = [kw["content"] for _, kw in utils.dispatch_queue]
    assert sent == ["'it's' is on 1 line of 1 rulebook chapter:\n\n**C**\n- **It's** \u201cfree\u201d.",
                    "'\u201cfree\u201d' is on 1 line of 1 rulebook chapter:\n\n**C**\n- It's **\u201cfree\u201d**.",
                    "'\"free\"' is on 1 line of 1 rulebook chapter:\n\n**C**\n- It's **\u201cfree\u201d**.",
                    "'\u201csneak attack\u201d' is on 1 line of 1 rulebook chapter:\n\n**D**\n- A **\"sneak attack\"**."]
    line = "y " * 50 + "[defensive_minion] " + "x" * 56 + " turn " + "z" * 120 + " [defensive_minion]: end."
    assert qc._cut(line, "turn") == "\u2026[defensive_minion] " + "x" * 56 + " turn " + "z" * 120 + " [defensive_minion]\u2026"
    rules = [f"Rule {i} " + "x" * 90 for i in range(6)]
    text = "\n".join(["Creature Types", "", "Every type has a function.", "", "MACHINE", "Immune to Sickness.", "", "WIND", *rules[:5],
                      "Wind again " + "x" * 90, rules[5], "Wind last.", "", "After."])
    repo.rulebook_dff = DataFrame({"ability": [text], "title": ["Creature Types"], "sections": [
        [[4, 7, "MACHINE", "machine"], [7, 17, "WIND", "wind"]]]}, index=["20. Creature Types"])
    lines = lambda q: qc.rulebook_lines(repo.rulebook_search_engine(q), q)
    assert lines("immune") == [("20. Creature Types", [("MACHINE", ["Immune to Sickness."])], 1)]
    assert lines("wind") == [("20. Creature Types", [("WIND", rules[:4] + ["\u2026", "Wind again " + "x" * 90, "Wind last."])], 3)]
    assert lines("rule 5") == [("20. Creature Types", [("WIND", [rules[5]])], 1)]
    assert lines("creature types") == [("20. Creature Types", [("", ["Every type has a function."])], 1)]   # the page's title
    utils.dispatch_queue.clear()
    qc.query_rulebook(utils.make_simple_fake_interaction(1), "wind")
    sent = [kw["content"] for _, kw in utils.dispatch_queue]
    utils.dispatch_queue.clear()
    assert sent[0].startswith("'wind' is on 3 lines of 1 rulebook chapter:\n\n**20. Creature Types \u00b7 WIND**\n- Rule 0 x")


def test_rulebook_timing_tag_finds_its_x_row(monkeypatch):
    """/query_rulebook: a numbered timing tag the book never prints whole ((2PG); 2pg is not (2PGT)'s) quotes its X row,
    saying so; a printed one ((1PG)) its row's lines first, then its own (with sections: the row's sections and lines
    first, a heading holding the tag too); no row in the book ((1/2CD)): as before."""
    import commands.utils as utils
    from pandas import DataFrame
    repo = qc.CardRepository()
    repo.rulebook_dff = DataFrame({"ability": ["(XCD) | Cooldown: needs X turn(s) of charge\n(1/2CD) | Starts with 1 of 2\n"
                                               "(XPG) | Can only be activated X time(s) per game\n(2PGT) | Now (1GCD)(2U)",
                                               "3. (AAT)(1PG) Deal 10 damage."]}, index=["23. Timing Tags", "11. Abilities"])
    monkeypatch.setattr(qc, "card_repo", repo)
    monkeypatch.setattr(qc, "render_type_tokens", lambda text, escape=False: text)
    for text in ("(2PG)", "(3CD)", "2pg", "(1PG)", "1/2cd"):
        qc.query_rulebook(utils.make_simple_fake_interaction(1), text)
    sent = [(kw["content"], kw["links"][0][1]) for _, kw in utils.dispatch_queue]
    utils.dispatch_queue.clear()
    pg = ("'(2PG)' is not printed in the rulebook; its row (XPG), with X = 2, is on 1 line of 1 rulebook chapter:\n\n"
          "**23. Timing Tags**\n- **(XPG)** | Can only be activated X time(s) per game", qc.search_link("rules", "(XPG)"))
    assert sent[0] == sent[2] == pg
    assert sent[1][0].startswith("'(3CD)' is not printed in the rulebook; its row (XCD), with X = 3, is on 1 line")
    assert sent[3] == ("'(1PG)' (its row (XPG), with X = 1) is on 2 lines of 2 rulebook chapters:\n\n**11. Abilities**\n"
                       "- 3. (AAT)**(1PG)** Deal 10 damage.\n\n**23. Timing Tags**\n- **(XPG)** | Can only be activated X "
                       "time(s) per game", qc.search_link("rules", "(1PG)"))
    assert sent[4][0].startswith("'1/2cd' is on 1 line")
    assert qc.x_row("-1/3cd", ["(XCD)"]) == ("(-1/3CD)", "(Y/XCD)", "Y = -1 and X = 3", False) and qc.x_row(
        "(2PGT)", []) == ("(2PGT)", "(XPGT)", "X = 2", False) and qc.x_row("-3cd", []) is None
    assert qc.x_row("1pg", ["(AAT)(1PG)"])[3] and not qc.x_row("2pg", ["(2PGT) | Now"])[3]
    pages = DataFrame({"ability": ["Tags\n(1CD) in the panel.\nCOOLDOWN (1CD)\n(1CD) first\n(XCD) | Cooldown"],
                       "title": ["Tags"], "sections": [[[2, 5, "COOLDOWN (1CD)", "cooldown-1cd"]]]}, index=["23. Timing Tags"])
    assert qc.rulebook_lines(pages, ("(XCD)", "(1CD)")) == [("23. Timing Tags", [
        ("COOLDOWN (1CD)", ["(XCD) | Cooldown", "(1CD) first"]), ("", ["(1CD) in the panel."])], 4)]


def test_rulebook_printed_timing_tags_say_what_they_mean(monkeypatch):
    """The real book: (1CD), (1GCD), (1PG), (2U) and (3PG) (178 cards) quote their X row's meaning line first in 23. Timing
    Tags (its section first), then their own lines, in one message; (2PG), never printed, its row alone as before."""
    import commands.utils as utils
    from pandas import DataFrame
    from commands import rulebook
    from global_config import RULEBOOK_SRC
    book = rulebook.chapters({f"{RULEBOOK_SRC}/{p.name}": p.read_text(encoding="utf-8")
                              for p in (SITE_SRC / "site" / "rulebook").glob("*.md")})
    repo = qc.CardRepository()
    repo.rulebook_dff = DataFrame({"ability": [c["text"] for c in book], "title": [c["title"] for c in book],
                                   "sections": [c.get("sections") for c in book]}, index=[rulebook.label(c) for c in book])
    monkeypatch.setattr(qc, "card_repo", repo)
    monkeypatch.setattr(qc, "render_type_tokens", lambda text, escape=False: text)
    rows = {"(1CD)": "(XCD)", "(1GCD)": "(XGCD)", "(1PG)": "(XPG)", "(2U)": "(XU)", "(3PG)": "(XPG)", "(2PG)": "(XPG)"}
    for tag in rows:
        qc.query_rulebook(utils.make_simple_fake_interaction(1), tag)
    sent = [kw["content"] for _, kw in utils.dispatch_queue]
    utils.dispatch_queue.clear()
    timing = [line.strip() for line in next(c["text"] for c in book if c["file"].startswith("23-")).split("\n")]
    assert len(sent) == len(rows)                       # one message each
    for (tag, row), reply in zip(rows.items(), sent):
        meaning = next(line for line in timing if line.startswith(f"{row} | "))
        said = (f"'{tag}' is not printed in the rulebook; its row {row}, with X = {tag[1]}, is" if tag == "(2PG)" else
                f"'{tag}' (its row {row}, with X = {tag[1]}) is")
        assert reply.startswith(said) and (f"**{tag}**" in reply) == (tag != "(2PG)"), reply
        first = reply.split("**23. Timing Tags", 1)[1].split("**\n- ", 1)[1]     # its first section's first line
        assert first.startswith(meaning.replace(row, f"**{row}**", 1)), reply


def _app_book(tmp_path, tail: str):
    """app.js's rulebook search (bookFold to the deck builder: bookRe, highlight...) in node, its marks into out; then
    tail, which writes JSON."""
    assets, app = SITE_SRC / "site" / "assets", (SITE_SRC / "site" / "assets" / "app.js").read_text(encoding="utf-8")
    code = app[app.index("const bookFold"):app.index("// ---- deck builder")]
    script = tmp_path / "book.mjs"
    script.write_text(f"""import fs from "fs";
const {{ straight, loose }} = await import({json.dumps((assets / "search.js").as_uri())});
const RB = await import({json.dumps((assets / "rulebook.js").as_uri())});
const NodeFilter = {{ SHOW_TEXT: 4 }}, out = [];
const document = {{ createTreeWalker: el => {{ let i = -1; return {{ nextNode: () => ++i < el.length, get currentNode() {{ return el[i]; }} }}; }},
  createDocumentFragment: () => ({{ append: x => x.textContent && out.push(x.textContent) }}), createElement: () => ({{}}) }};
const marks = (text, q) => {{ out.length = 0; highlight([[{{ nodeValue: text, replaceWith() {{}} }}]], bookFold(q)); return [...out]; }};
{code}
{tail}""", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_website_rulebook_search_takes_curly_quotes(tmp_path):
    """The website's rulebook search (app.js bookFold, highlight): a phone's curly ' and " list the same chapters and
    marks as plain ones, the marks keeping the printed characters."""
    found = _app_book(tmp_path, f"""const dir = {json.dumps(str(SITE_SRC / "site" / "rulebook"))}, texts = fs.readdirSync(dir).filter(f => f.endsWith(".md")).sort()
  .map(f => [f, bookFold(RB.plainText(RB.parse(fs.readFileSync(`${{dir}}/${{f}}`, "utf8"))))]);
const seen = q => [texts.filter(([, t]) => bookRe(bookFold(q)).test(t)).map(([f]) => f), marks("It's “On Death”: can't.", q)];
process.stdout.write(JSON.stringify(["can't", "creature's hp", '"on death"', "it's"].map(q =>
  [seen(q), seen(q.replace(/'/g, "\\u2019").replace(/"([^"]*)"/, "\\u201c$1\\u201d"))])));""")
    assert all(plain == curly and plain[0] for plain, curly in found), found
    assert [marks for (_, marks), _ in found] == [["can't"], [], ["\u201cOn Death\u201d"], ["It's"]]


def test_website_rulebook_timing_tag_finds_its_x_row(tmp_path):
    """The website's Rulebook search (app.js bookTag): a numbered timing tag the printed chapters never print whole
    searches its X row instead, as the bot; (3U) only in the website-only Stoogedings chapter counts as not printed; a
    printed one ((1PG)) the tag or its row, both marked."""
    found = _app_book(tmp_path, f"""const dir = {json.dumps(str(SITE_SRC / "site" / "rulebook"))}, chapters = fs.readdirSync(dir)
  .filter(f => /^\\d\\d-.*\\.md$/.test(f)).sort().map(f => {{ const ast = RB.parse(fs.readFileSync(`${{dir}}/${{f}}`, "utf8"));
    return {{ file: f, ast, text: bookFold(`${{ast.title}}\\n${{RB.plainText(ast)}}`) }}; }});
out.length = 0; highlight([[{{ nodeValue: "(XPG) | Per game. (1PG)(AAT)", replaceWith() {{}} }}]], ["(1pg)", "(xpg)"]);
process.stdout.write(JSON.stringify([["(2PG)", "2pg", "(3CD)", "(3U)", "(1CD)", "(1/2CD)", "(XPG)", "(2/3U)"].map(q =>
  bookTag(bookFold(q), chapters)), out]));""")
    found, marked = found
    assert found[:4] == [{"tag": "(2PG)", "row": "(xpg)", "says": "X = 2", "printed": False}] * 2 + [
        {"tag": "(3CD)", "row": "(xcd)", "says": "X = 3", "printed": False},
        {"tag": "(3U)", "row": "(xu)", "says": "X = 3", "printed": False}]
    assert found[4:6] == [{"tag": "(1CD)", "row": "(xcd)", "says": "X = 1", "printed": True},
                          {"tag": "(1/2CD)", "row": "(y/xcd)", "says": "Y = 1 and X = 2", "printed": True}]
    assert found[6:] == [None] * 2 and marked == ["(XPG)", "(1PG)"]     # not numbered, or no (Y/XU) row
    app = (SITE_SRC / "site" / "assets" / "app.js").read_text(encoding="utf-8")
    rules = app[app.index("function renderRules(r) {"):app.index("// a deep link's place, again once")]
    assert "x = typed && bookTag(typed, chapters)" in rules and "is not printed in the rulebook; its row" in rules
    assert "x.printed ? [x.tag.toLowerCase(), x.row] : x.row" in rules


def test_old_rulebook_page_links_open_their_chapter(tmp_path):
    """The bot's 10-25 changelog posts link old Photoshop rulebook pages (#/rules?q=11%20Abilities4): app.js bookPage
    opens the chapter of that number and title; any other search still searches."""
    pages = {"contents": ["00 Table of Contents"], "getting-started": ["01 Getting Started"], "table": ["04 Table3", "04 Table5"],
             "shops": ["05 Shops"], "reading-cards": ["06 Reading Cards"], "stats": ["07 Stats1"],
             "the-global-turn": ["08 Global Turn"], "taking-your-turn": ["09 TakingYourTurn"],
             "attacking": ["10 Attacking", "10 Attacking2"], "abilities": [f"11 Abilities{n}" for n in ("", 2, 3, 4, 6, 7)],
             "reactions": [f"12 Reactions{n}" for n in ("", 2, 3, 4, 5)], "items": ["13 Items"], "item-types": ["14 Item Types2"],
             "death": ["17 Death"], "field-cards": ["18 Field Cards"], "status-effects": ["19 Status Effects", "19 Status Effects2"]}
    names = [n for ns in pages.values() for n in ns]
    others = ["10 HP", "12 Abilities4", "11 abilities are", "11Abilities", "1 Abilities", "abilities", "(2PG)", "08 Turn"]
    found = _app_book(tmp_path, f"""const dir = {json.dumps(str(SITE_SRC / "site" / "rulebook"))}, chapters = fs.readdirSync(dir)
  .filter(f => /^\\d\\d-.*\\.md$/.test(f)).sort().map(f => ({{ id: f.slice(3, -3), num: f.slice(0, 2), title: RB.parse(fs.readFileSync(`${{dir}}/${{f}}`, "utf8")).title }}));
process.stdout.write(JSON.stringify({json.dumps(names + others)}.map(q => bookPage(q, chapters)?.id ?? null)));""")
    assert found == [c for c, ns in pages.items() for _ in ns] + [None] * len(others)
    app = (SITE_SRC / "site" / "assets" / "app.js").read_text(encoding="utf-8")
    assert "history.replaceState(history.state, \"\", h); renderView(parseHash(), h)" in app[app.index("function renderRules(r) {"):]


def test_page_search_boxes_close_a_phones_keyboard(tmp_path):
    """The Rulebook's and the published decks' own search boxes (app.js searchBox, no form): Enter searches at once and
    a phone's Search key closes the keyboard, also past the search's 200 ms re-focus; a desktop keeps the redrawn box
    focused; an IME's Enter and other keys change nothing."""
    app = (SITE_SRC / "site" / "assets" / "app.js").read_text(encoding="utf-8")
    for sel in ("#rules-q", "#decks-q"):
        assert re.search(rf'const input = \$\("{sel}"\);\n\s*searchBox\(input, debounce\(', app), sel
    at = app.index("function searchBox(")
    code = re.search(r"^function debounce\(.*$", app, re.M)[0] + "\n" + app[at:app.index("\n}\n", at) + 2]
    script = tmp_path / "box.mjs"
    script.write_text(code + r"""
let active = null, cur = null, coarse = true, redraws = 0;
const document = { get activeElement() { return active; } }, matchMedia = () => ({ matches: coarse }), $ = () => cur;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const draw = () => { const on = {}, b = { id: "rules-q", on, addEventListener: (t, h) => { on[t] = h; }, setSelectionRange() {},
  focus() { active = b; }, blur() { if (active === b) active = null; } };
  cur = b; searchBox(b, debounce(() => { redraws++; draw(); $().focus(); }, 200)); };   // setParams redraws the box
const seen = () => [redraws, active && active.id];
async function press(phone, key, waited = false) {
  coarse = phone; redraws = 0; draw(); cur.focus(); cur.on.input();
  if (waited) await sleep(300);
  cur.on.keydown(key); const now = seen(); await sleep(300); return [now, seen()];
}
const enter = { key: "Enter" };
process.stdout.write(JSON.stringify([await press(true, enter), await press(false, enter), await press(true, enter, true),
  await press(true, { key: "Enter", isComposing: true }), await press(true, { key: "Enter", keyCode: 229 }), await press(true, { key: "a" })]));
""", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    phone, desk, later, *typing = json.loads(done.stdout)
    assert phone == [[1, None], [1, None]]                # searched now, keyboard closed, no re-focus after
    assert desk == [[1, "rules-q"], [1, "rules-q"]]
    assert later == [[1, None], [1, None]]                # the search had run: only the keyboard closes
    assert typing == [[[0, "rules-q"], [1, "rules-q"]]] * 3


def test_words_typed_without_punctuation(tmp_path, monkeypatch):
    """A word typed without ' or . also finds it printed with them after a letter (cant: can't, nme: N.M.E.), never a
    sentence's end between two words; digits never (05 is not 0.5); quoted or typed with them: exact. Cards and
    rulebook, the bot and the website alike."""
    from pandas import DataFrame
    cards = [dict(name="Gag", type="debuff", ability="Can't attack."), dict(name="Tilt", type="item", ability="A cant."),
             dict(name="Shop", type="item", ability="Buy an N.M.E. shop item."),
             dict(name="Half", type="item", ability="Heal 0.5 HP."),
             dict(name="Clot", type="minion", ability="Its parent creature. Attacks hit the owners' foes.")]
    rows = [("cant", {"Gag", "Tilt"}), ("can't", {"Gag"}), ('"cant"', {"Tilt"}), ('"can\'t"', {"Gag"}),
            ("nme shop", {"Shop"}), ("05", set()), ("0.5", {"Half"}), ("creature attacks", set()),
            ("owners foes", {"Clot"})]
    repo = _repo(cards)
    assert [_shown(repo, repo.ability_search_engine(q, near=False).index) for q, _ in rows] == [c for _, c in rows]
    assert _website(tmp_path, [q for q, _ in rows], cards) == [c for _, c in rows]
    assert repo.ability_hits("cant") == 2
    book = {"4. Table": "4. N.M.E. Items for the N.M.E. Shop.\nThe next turn.", "5. Shops": "N.M.E SHOP\nIt can't fit 0.5.",
            "6. Bans": "NME stock", "7. Fights": "A creature. Attacks hit the owners' foes."}
    repo.rulebook_dff = DataFrame({"ability": list(book.values())}, index=list(book))
    lines = lambda q: qc.rulebook_lines(repo.rulebook_search_engine(q), q)
    assert lines("nme") == [("4. Table", [("", ["4. N.M.E. Items for the N.M.E. Shop."])], 1), ("5. Shops", [("", ["N.M.E SHOP"])], 1),
                            ("6. Bans", [("", ["NME stock"])], 1)]       # the lines, not the whole chapter
    queries = ["nme", "nme shop", "cant", "05", "turn", "creature attacks", "owners foes"]
    bot = [[n for n, *_ in lines(q)] for q in queries]
    assert bot == [["4. Table", "5. Shops", "6. Bans"], ["4. Table", "5. Shops"], ["5. Shops"], [], ["4. Table"], [],
                   ["7. Fights"]]
    shown = "4. N.M.E. Items for the N.M.E. Shop. The next turn."
    bolded = [qc._bold(shown, q) for q in ("nme", "nme shop", "turn")]
    assert bolded == ["4. **N.M.E.** Items for the **N.M.E.** Shop. The next turn.",
                      "4. N.M.E. Items for the **N.M.E. Shop**. The next turn.",
                      "4. N.M.E. Items for the N.M.E. Shop. The next **turn**."]   # an abbreviation's last '.' only
    site = _app_book(tmp_path, f"""const book = {json.dumps(book)};
process.stdout.write(JSON.stringify([{json.dumps(queries)}.map(q => Object.keys(book).filter(n => bookRe(bookFold(q)).test(bookFold(book[n])))),
  ["nme", "nme shop", "turn"].map(q => marks({json.dumps(shown)}, q))]));""")
    assert site == [bot, [["N.M.E.", "N.M.E."], ["N.M.E. Shop"], ["turn"]]]


def test_replies_ping_nobody(monkeypatch):
    """A reply quotes the typed text: '@everyone' in it pings nobody."""
    import commands.utils as utils
    from discord import AllowedMentions
    from pandas import DataFrame
    repo = qc.CardRepository()
    repo.rulebook_dff = DataFrame({"ability": ["Each slot takes its turn."]}, index=["08 Global Turn"])
    monkeypatch.setattr(qc, "card_repo", repo)
    qc.query_rulebook(utils.make_simple_fake_interaction(1), "@everyone")
    [(_, kw)] = utils.dispatch_queue
    assert kw["content"] == "No rulebook lines match '@everyone'."
    assert kw["allowed_mentions"].to_dict() == AllowedMentions.none().to_dict()


def test_ban_list(monkeypatch):
    """The website's ban list: a section after the chapters (its own link), a banned card's line (a duplicate's too)."""
    import commands.utils as utils
    from pandas import DataFrame
    from types import SimpleNamespace
    site = {"cards": [{"id": "marx", "png": "Creatures/Kirby/Marx.png"}, {"id": "marx2", "png": "Creatures/Kirby/Marx2.png",
                                                                          "dup": "marx"}],
            "thumbs": {"240": "aaa111", "360": "bbb222"},
            "banlist": {"notes": ["One of each item."], "groups": [
                {"title": "5 star creatures", "entries": [{"name": "Marx", "reason": "antifun", "id": "marx"}]},
                {"title": "4 star items", "entries": [{"name": "Mimic", "reason": "", "id": None}]}]}}
    reply = SimpleNamespace(status_code=200, headers={}, raise_for_status=lambda: None, json=lambda: site)
    monkeypatch.setattr(qc, "get", lambda url, **kw: reply)
    repo = qc.CardRepository()
    repo._populate_site_ids()
    assert repo.banned == {"marx": "antifun", "marx2": "antifun"}
    assert repo.thumb_tag == "bbb222"                                          # card grids' pictures
    assert qc.ban_list_text(repo.banlist) == "One of each item.\n\n5 star creatures:\nMarx - antifun\n\n4 star items:\nMimic"
    repo.rulebook_dff = DataFrame({"ability": ["Use an item."]}, index=["13. Items"])
    monkeypatch.setattr(qc, "card_repo", repo)
    monkeypatch.setattr(qc, "render_type_tokens", lambda text, escape=False: text)
    for text in ("item", "antifun"):
        qc.query_rulebook(utils.make_simple_fake_interaction(1), text)
    sent = [kw for _, kw in utils.dispatch_queue]
    assert sent[0]["content"] == ("'item' is on 3 lines of 1 rulebook chapter and the Constructed ban list:\n\n**13. Items**\n"
                                  "- Use an **item**.\n\n**Constructed ban list**\n- One of each **item**.\n- 4 star **item**s:")
    assert [label for label, _ in sent[0]["links"]] == ["Rulebook on the website", "Constructed ban list"]
    assert sent[1]["content"].startswith("'antifun' is on 1 line of the Constructed ban list:")
    assert sent[1]["links"] == [("Constructed ban list", f"{qc.SITE_URL}/#/rules/banlist")]
    posted = []
    monkeypatch.setattr(qc, "queue_message", lambda inter, content, **kw: posted.append(kw))
    repo.git_files = {"marx2.png": "Creatures/Kirby/Marx2.png"}
    monkeypatch.setattr(cm.stats_db, "stats", {"Marx2": SimpleNamespace(name="Marx2", series="Kirby")})
    qc._post_card(SimpleNamespace(), "marx2.png")
    assert posted[0]["embed"].description == "Kirby\nBanned in constructed: antifun"
