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
