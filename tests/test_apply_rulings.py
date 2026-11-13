"""scripts/apply_rulings.py (by hand): the review page's exported answers checked, stored and withdrawn; the page's
radios and the Morgue list with the answers (commands/timeline_review.py)."""
import importlib.util
import json
from pathlib import Path

import commands.timeline_review as tr

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "apply_rulings.py"


def load():
    spec = importlib.util.spec_from_file_location("apply_rulings", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def items():
    field = tr.item("field", "chain-f1", "matches", "fields", "Bolt: stats", card="bolt")
    field["choices"] = [tr.choice("yes", "yes: 5/2/6/4"), tr.choice("no", "no")]
    field["other"] = {"c": "value", "kind": "stats", "label": "the stats it shows"}
    card = tr.item("card not told", "match:P-1", "matches", "card", "2020-07-22", picture="P-1")
    card["choices"] = [tr.choice("card", "Bolt", "bolt"), tr.choice("none", "none of these")]
    card["other"] = {"c": "card", "kind": "card", "label": "another card"}
    made = tr.item("created earlier", "chain-c1", "created", "earlier", "Bolt", card="bolt")
    made["choices"] = [tr.choice("ours", "ours"), tr.choice("file", "the website's")]
    made["other"] = {"c": "date", "kind": "date", "label": "another date"}
    hint = tr.item("unsure fact", "R1", "matches", "facts", "Bolt", fact="f9")
    hint["choices"] = [tr.choice("yes", "right, in points", "points"), tr.choice("no", "wrong")]
    answer = tr.item("owner's answer", "chain-q1", "answers", None, "Q1 = C", read_only=True)
    return [field, card, made, hint, answer]


def test_answers_are_checked_stored_and_withdrawn(tmp_path):
    ar = load()
    review = tmp_path / "derived" / "review"
    review.mkdir(parents=True)
    (review / "items.json").write_text(json.dumps({"format": tr.FORMAT, "cards": {"bolt": "Bolt", "pike": "Pike"},
                                                   "items": items()}))
    export = lambda **answers: {"format": tr.FORMAT, "rulings": answers}
    stored = lambda: json.loads((review / "rulings.json").read_text())["rulings"]
    logs = []
    run = lambda e: ar.apply(e, str(tmp_path), build=False, log=logs.append)
    bad = {"chain-x": {"choice": "yes"}, "chain-q1": {"choice": "yes"}, "chain-f1": {"choice": "value", "value": "5/2"},
           "chain-c1": {"choice": "date", "value": "1999-01-01"}, "match:P-1": {"choice": "card", "value": "Zed"},
           "R1": {"choice": "yes", "value": "inches"}}
    assert run(export(**bad)) == 1 and not (review / "rulings.json").exists()
    assert sum(" refused" in line for line in logs) == 1 and len(logs) == 7          # every error named, none kept
    assert run({"rulings": {}}) == 1                                                 # not the page's export
    good = {"chain-f1": {"choice": "value", "value": " 5 / 2 / 6 / 5 "}, "match:P-1": {"choice": "card",
                                                                                         "value": "pike"},
            "chain-c1": {"choice": "date", "value": "before 2020"}, "R1": {"choice": "yes", "value": "points"}}
    assert run(export(**good)) == 0
    got = stored()
    assert got["chain-f1"]["value"] == {"hp": 5, "def": 2, "atk": 6, "spd": 5} and got["chain-f1"]["kind"] == "field"
    assert (got["match:P-1"]["value"], got["chain-c1"]["value"], got["R1"]["item"]["fact"]) == (
        "pike", "Before 2020", "f9")
    assert run(export(**{"chain-f1": {"choice": "yes"}, "R1": {"choice": "yes", "value": "points"},
                         "chain-c1": {"choice": "clear"}})) == 0
    got = stored()
    assert "chain-c1" not in got and got["chain-f1"] == {**got["chain-f1"], "choice": "yes"} and \
        "value" not in got["chain-f1"]
    assert "1 new or changed, 1 unchanged, 1 cleared" in logs[-1]
    (review / "items.json").write_text(json.dumps({"format": tr.FORMAT, "cards": {}, "items": []}))
    assert run(export(**{"R1": {"choice": "no"}})) == 0 and stored()["R1"]["choice"] == "no"   # taken in: still


def test_the_page_offers_the_answers_and_the_morgue_list(tmp_path):
    html = tr.item_html(items()[0])
    assert "data-c='yes'" in html and "data-c='value' data-other" in html and "data-kind='stats'" in html
    assert "type=radio" not in tr.item_html(items()[-1])                             # read only
    cell = tr.pic_html(tr.pic("../matcher/x.webp", "cell", [0.5, 0, 1, 1], 0.5))
    assert "background-size:200.00% 100.00%" in cell and "background-position:100.00% 0.00%" in cell
    (tmp_path / "derived" / "morgue").mkdir(parents=True)
    (tmp_path / "derived" / "review").mkdir(parents=True)
    entry = lambda n: {"id": f"LC-{n}", "name": f"Card {n}"}
    (tmp_path / "derived" / "morgue" / "candidates.json").write_text(json.dumps(
        {"candidates": [entry(1), entry(2)], "not_candidates": [entry(4)]}))
    (tmp_path / "derived" / "morgue" / "ideas.json").write_text(json.dumps({"ideas": [entry(3)]}))
    (tmp_path / "derived" / "review" / "rulings.json").write_text(json.dumps({"rulings": {
        "morgue:LC-1": {"choice": "no"}, "morgue:LC-3": {"choice": "yes"}}}))
    got = tr.morgue_list(str(tmp_path))
    assert [(e["id"], e.get("owner")) for e in got["entries"]] == [("LC-2", None), ("LC-3", "yes")]
