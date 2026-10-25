"""/query_ability's query language on a table of website-shaped cards, and /query_rulebook's lines."""

import pytest

import commands.card_models as cm
import commands.query_card as qc


# website-shaped cards: name (as shown), aliases, ability, fields
CARDS = [
    dict(name="Kirby", type="creature", series="kirby", hp=5, defense=3, atk=4, spd=6, stars=3, types=["wind"],
         ability="Deal 2 DMG to the enemy."),
    dict(name="Meta Knight", type="creature", series="kirby", hp=7, defense=5, atk=7, spd=8, stars=4, aliases=["mk"],
         ability="Deals double damage. Die at the end of the turn."),
    dict(name="sans.", file="Sans", type="creature", series="undertale", hp=1, defense=1, atk=1, spd=9, stars=5,
         ability="Dodge every attack. +1 SPD."),
    dict(name="King Dedede", type="creature", series="kirby", hp=4, defense=4, atk=4, spd=2, stars=4, ability="Hammer time."),
    dict(name="Mettaton EX", type="creatures", hp=6, defense=6, atk=6, spd=6, ability="Gain +2 DEF."),
    dict(name="Fire Bomb", type="item", subtype="attack", stars=2, ability="Deal 3 damage to a [fire] creature."),
    dict(name="Panda Shield", type="item", subtype="equip", stars=1, aliases=["buckler"], ability="+1 defense."),
    dict(name="Bloon", type="minion", hp=2, defense=1, atk=1, spd=3, ability="Pop: heal 2 health."),
    dict(name="Burning", type="debuff", stars=1, ability="Burning: 3 DMG, can't affect [fire]."),
]
ERROR = "error"                                          # the bot answers with an error

QUERIES = [                                              # query, the cards found
    ("damage", {"Meta Knight", "Fire Bomb", "Kirby", "Burning"}),
    ("dmg", {"Kirby", "Burning", "Meta Knight", "Fire Bomb"}),
    ("defense", {"Panda Shield", "Mettaton EX"}),
    ("health", {"Bloon"}), ("speed", {"sans."}),
    ('"damage"', {"Meta Knight", "Fire Bomb"}),
    ("!type==creature", {"Mettaton EX", "Fire Bomb", "Panda Shield", "Bloon", "Burning"}),
    ("type==creature", {"Kirby", "Meta Knight", "sans.", "King Dedede"}),
    ("type~creature", {"Kirby", "Meta Knight", "sans.", "King Dedede", "Mettaton EX"}),
    ("hp>5", {"Meta Knight", "Mettaton EX"}), ("hp > 5", {"Meta Knight", "Mettaton EX"}),
    ("total>=24", {"Meta Knight", "Mettaton EX"}), ("spd>=8", {"Meta Knight", "sans."}),
    ("stars<=2", {"Fire Bomb", "Panda Shield", "Burning"}),
    ("fire", {"Fire Bomb", "Burning"}), ("dedede", {"King Dedede"}), ("mk", {"Meta Knight"}),
    ("sans.", {"sans."}), ("name~knight", {"Meta Knight"}), ("types~wind", {"Kirby"}),
    ("buckler", {"Panda Shield"}), ("alias~mk", {"Meta Knight"}), ("aliases==buckler", {"Panda Shield"}),
    ("display_name~sans", {"sans."}),  # any column of the bot's
    ("die & series==kirby", {"Meta Knight"}), ("fire | hp>6", {"Fire Bomb", "Burning", "Meta Knight"}),
    ("!die", {c["name"] for c in CARDS} - {"Meta Knight"}),
    ("series==kirby & !hp>5", {"Kirby", "King Dedede"}), ("subtype==equip", {"Panda Shield"}),
    ("atack>3", ERROR), ("total~5", ERROR), ("hp>=x", ERROR), ("type>3", ERROR),
    ("type!=creature", ERROR),
    ('"atack>3"', set()),
]


@pytest.fixture(scope="module")
def repo():
    stats = {}
    for c in CARDS:
        name = c.get("file", c["name"])
        stats[name] = cm.CardInfo(name=name, card_type=c["type"], path=f"X/{name}.psd", ability=c["ability"],
                                  stars=c.get("stars"), series=c.get("series"), subtype=c.get("subtype"),
                                  types=c.get("types", []), display_name=c["name"] if "file" in c else None,
                                  stats=cm.CardStats(c.get("hp", -1), c.get("defense", -1), c.get("atk", -1), c.get("spd", -1)))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cm.stats_db, "stats", stats)
        repo = qc.CardRepository()
        repo._edit_names = ("sha", {qc.CardAuthors.key(c.get("file", c["name"])): c["aliases"]   # as Card Edits.json
                                    for c in CARDS if c.get("aliases")})
        repo.prep_dataframes()
    return repo


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


def test_rulebook_lines(monkeypatch):
    """/query_rulebook: the matching lines under their page names (most matches first), whole pages per message."""
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
    assert [(n, len(lines)) for n, lines in found] == [("08 Global Turn", 3), ("02 Abbreviations", 2), ("30 Long", 1),
                                                       ("50 Bricks", 1)]
    assert found[1][1] == ["(OYT) = On your turn.", "(BOT) = Beginning of turn."]          # its own bullet dropped
    assert len(found[2][1][0]) <= qc.SNIPPET + 2 and "turn" in found[2][1][0] and found[2][1][0].startswith("…")
    assert qc._bold("A <:defensive_minion:12> minion, <a:x_1:12> 1", "minion") == "A <:defensive_minion:12> **minion**, <a:x_1:12> 1"
    assert qc._bold("<:x_1:12> 1", "1") == "<:x_1:12> **1**"                                   # emoji markup kept whole
    assert qc.rulebook_lines(repo.rulebook_search_engine("global  turn"), "global turn") == [
        ("08 Global Turn", ["THE GLOBAL TURN"]), ("50 Bricks", ["1 per global turn."])]          # over a line break
    utils.dispatch_queue.clear()
    qc.query_rulebook(utils.make_simple_fake_interaction(1), "turn")
    sent = [kw for _, kw in utils.dispatch_queue]
    utils.dispatch_queue.clear()
    assert len(sent) == 1 and "links" in sent[0]
    assert sent[0]["content"].startswith("'turn' is on 7 lines of 4 rulebook pages:\n\n**08 Global Turn**\n- THE GLOBAL **TURN**")
