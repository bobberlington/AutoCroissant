"""history_reread: a version's stored values against today's re-read, and the PNG's say on stats."""
from commands.card_models import CardInfo, CardStats
from commands.history_reread import picture, verdict


def card(parser=7, **kw):
    return CardInfo(name="X", card_type="creature", parser_version=parser, **kw)


def test_verdicts():
    old, new = card(None, stars=None, types=["darkmatter"], ability="Can’t affect darkmatter."), \
        card(stars=5, types=["debuff"], ability="Can't affect [darkmatter].")
    assert [verdict(old, new, key) for key in ("stars", "types", "ability")] == ["fill", "differs", "tokens"]
    assert verdict(card(None, types=["undead"]), card(types=[]), "types") == "lost"
    assert verdict(card(None, types=["fire"]), card(types=["fire"]), "types") == "confirm"
    assert verdict(card(ability="A  \nb’s"), card(ability="A\nb's"), "ability") == "format"
    assert verdict(card(stats=CardStats(-1, -1, -1, -1)), card(stats=CardStats(1, 2, 3, 4)), "stats") == "fill"
    assert verdict(card(stats=CardStats(5, 4, 5, 5)), card(stats=CardStats(2, 2, 2, 2)), "stats") == "differs"


def test_picture_judges_only_the_differing_stats():
    old, new = card(stats=CardStats(5, 4, 5, 5)), card(stats=CardStats(5, 4, 2, 5))
    row = lambda *could: [{"best": could[0], "could": list(could)}]
    reads = {"hp": row(9), "def": row(9), "atk": row(2), "spd": row(9)}     # only ATK differs: the rest has no say
    assert picture({"style": "new", "reads": reads}, old, new) == "adopt"
    assert picture({"style": "new", "reads": {**reads, "atk": row(5)}}, old, new) == "keep"
    assert picture({"style": "new", "reads": {**reads, "atk": row(2, 5)}}, old, new) == "review"
    unread = {**reads, "hp": [{"best": None, "could": None}]}
    assert picture({"style": "custom", "reads": unread}, old, new) == "review"
    assert picture({"style": "new", "reads": reads}, card(stars=6, stats=old.stats), new) == "review"
    assert picture(None, old, new) == "review"
