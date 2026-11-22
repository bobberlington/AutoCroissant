"""The psd_analyzer split: the classifier guard."""


import commands.psd_parser as pp


def test_classify_shallow_paths_are_unknown():
    unknown = {"type": "unknown"}
    assert pp.CardClassifier.classify("Card.psd") == unknown
    assert pp.CardClassifier.classify("Items/Card.psd") == unknown            # no subtype folder
    assert pp.CardClassifier.classify("Creatures/Card.psd") == unknown
    assert pp.CardClassifier.classify("Items/Heal/2 Stars/B.psd") == {"type": "item", "subtype": "heal", "stars": 2}
    assert pp.CardClassifier.classify("N.M.E/Star_Jar.psd") == {"type": "nme"}
    assert pp.CardClassifier.classify("Auxiliary/Minions/Mini_Doomer.psd")["type"] == "minion"


def test_a_new_creature_takes_its_folders_one_series(monkeypatch):
    """No record: the series every other card in its folder carries, else none (11-22: Beholster's 'other'); a
    stored folder name lowercased (the old default) is none too, a website edit's value stays."""
    from contextlib import nullcontext
    from commands.card_models import CardInfo, StatsDatabase
    monkeypatch.setattr(pp, "variant_view", lambda psd, n: nullcontext())
    monkeypatch.setattr(pp.PSDParser, "_extract_from_layers", lambda *a, **k: None)
    db = StatsDatabase()
    put = lambda name, path, series, **kw: db.stats.setdefault(name, CardInfo(name=name, card_type="creature",  # noqa: E731
                                                                              path=path, series=series, **kw))
    put("Isaac", "Creatures/Isaac/2 Stars/Isaac.psd", "The Binding of Isaac")
    put("Eden", "Creatures/Isaac/3 Stars/Eden.psd", "The Binding of Isaac")
    put("Lilith", "Creatures/Isaac/3 Stars/Lilith.psd", "The Binding of Isaac", parsed_values={"series": "isaac"})
    put("Ness", "Creatures/Other/3 Stars/Ness.psd", "MOTHER")
    put("Leshy", "Creatures/Other/4 Stars/Leshy.psd", "Inscryption")
    put("Beholster", "Creatures/Other/5 Stars/Beholster.psd", "other")
    parse = lambda path, existing=None: pp.PSDParser([], db).parse(object(), path, existing)    # noqa: E731
    assert parse("Creatures/Isaac/4 Stars/Cain.psd").series == "The Binding of Isaac"
    assert parse("Creatures/Other/2 Stars/Kid.psd").series is None                   # Other: two series
    assert parse("Creatures/Kirby/2 Stars/Waddle_Dee.psd").series is None             # no card there yet
    assert parse("Creatures/Other/5 Stars/Beholster.psd", db.stats["Beholster"]).series is None
    assert parse("Creatures/Isaac/3 Stars/Lilith.psd", db.stats["Lilith"]).series == "The Binding of Isaac"
    assert parse("Creatures/Other/3 Stars/Ness.psd", db.stats["Ness"]).series == "MOTHER"
