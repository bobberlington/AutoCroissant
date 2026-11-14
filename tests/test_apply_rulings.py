"""scripts/apply_rulings.py (by hand): the review page's exported answers checked, stored and withdrawn; the page's
radios and the Morgue list with the answers (commands/timeline_review.py). A 'yes' carries the value it confirms."""
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
    field["choices"] = [tr.choice("yes", "yes: 5/2/6/4", {"hp": 5, "def": 2, "atk": 6, "spd": 4}),
                        tr.choice("no", "no")]
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
    bad = {"chain-x": {"choice": "yes"}, "chain-q1": {"choice": "yes"}, "chain-f1": {"choice": "yes"},   # no value
           "chain-c1": {"choice": "date", "value": "1999-01-01"}, "match:P-1": {"choice": "card", "value": "Zed"},
           "R1": {"choice": "yes", "value": "inches"}}
    assert run(export(**bad)) == 1 and not (review / "rulings.json").exists()
    assert sum(" refused" in line for line in logs) == 1 and len(logs) == 7          # every error named, none kept
    assert run(export(**{"chain-f1": {"choice": "value", "value": "5/2"}})) == 1
    assert run({"rulings": {}}) == 1                                                 # not the page's export
    good = {"chain-f1": {"choice": "value", "value": " 5 / 2 / 6 / 5 "}, "match:P-1": {"choice": "card",
                                                                                         "value": "pike"},
            "chain-c1": {"choice": "date", "value": "before 2020"}, "R1": {"choice": "yes", "value": "points"}}
    assert run(export(**good)) == 0
    got = stored()
    assert got["chain-f1"]["value"] == {"hp": 5, "def": 2, "atk": 6, "spd": 5} and got["chain-f1"]["kind"] == "field"
    assert (got["match:P-1"]["value"], got["chain-c1"]["value"], got["R1"]["item"]["fact"]) == (
        "pike", "Before 2020", "f9")
    yes = {"choice": "yes", "value": {"spd": 4, "atk": 6, "def": 2, "hp": 5}}        # the value it confirms
    assert run(export(**{"chain-f1": yes, "R1": {"choice": "yes", "value": "points"},
                         "chain-c1": {"choice": "clear"}})) == 0
    got = stored()
    assert "chain-c1" not in got and got["chain-f1"]["choice"] == "yes" and \
        got["chain-f1"]["value"] == {"hp": 5, "def": 2, "atk": 6, "spd": 4}
    assert "1 new or changed, 1 unchanged, 1 cleared" in logs[-1]
    (review / "items.json").write_text(json.dumps({"format": tr.FORMAT, "cards": {}, "items": []}))
    assert run(export(**{"R1": {"choice": "no"}})) == 0 and stored()["R1"]["choice"] == "no"   # taken in: still


def test_the_page_offers_the_answers_and_the_morgue_list(tmp_path):
    html = tr.item_html(items()[0])
    assert "data-c='yes'" in html and "data-c='value' data-other" in html and "data-kind='stats'" in html
    assert "type=radio" not in tr.item_html(items()[-1])                             # read only
    cell = tr.pic_html(tr.pic("../matcher/x.webp", "cell", [0.5, 0, 1, 1], 0.5))
    assert "background-size:200.00% 100.00%" in cell and "background-position:100.00% 0.00%" in cell
    how = {"halves": {"Lorn": "top", "Pike": "bottom"}, "matched_by": "today", "eye": "the bar colours",
           "readings": {"today": {"top": "Lorn", "bottom": "Pike"},
                        "first old version": {"top": "Pike", "bottom": "Lorn", "days": ["2023-10-16"]}}}
    said = tr.split_evidence(how)                  # a split's halves: what picked them, what reads the other way
    assert said[0] == "Top rows Lorn, bottom rows Pike. Picked by today's stats and the bar colours."
    assert len(said) == 2 and "first old version (2023-10-16)" in said[1] and tr.split_evidence(None) == []
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


def test_the_page_s_items_from_the_parts(tmp_path):
    """Built from the parts' outputs: a shared name needs the id (typed and listed), stats typed as hp/def or two
    creatures, a map to a card that isn't live takes no 'yes', the rival card a reason names is a radio, a picture the
    preview's eye leaves out isn't asked about, Morgue hints only before the Created date in use (no meme ones), an
    answer whose item left the page: taken in or changed, and a look answered by its version's key (a stored number no
    radio takes: not applied, kept as stored)."""
    ar = load()
    derived = tmp_path / "derived"
    put = lambda parts, data: ((derived.joinpath(*parts)).parent.mkdir(parents=True, exist_ok=True),
                               derived.joinpath(*parts).write_text(data if isinstance(data, str) else json.dumps(data)))
    lines = lambda rows: "".join(json.dumps(r) + "\n" for r in rows)
    cards = {"bolt": {"key": "Bolt", "name": "Bolt"}, "kit": {"key": "Kit", "name": "Kit"},
             "kit2": {"key": "Kit 2", "name": "Kit"}, "tentacle": {"key": "Tentacle", "name": "Tentacle"},
             "blooper-tentacle": {"key": "Blooper Tentacle", "name": "Blooper Tentacle"}}
    key = lambda c: "b" + c * 20
    put(("matcher", "match", "pairs.json"), {"cards": cards, "versions": [
        {"key": key("a"), "card": "bolt", "ts": 1.6e9, "look": 1, "thumb": "t/a.webp"},
        {"key": key("b"), "card": "bolt", "ts": 1.7e9, "look": 0, "thumb": "t/b.webp"}]})
    put(("match", "verdicts.jsonl"), lines([{"id": "P-1", "verdict": "unsure", "maybe": ["tentacle"], "first": {
        "date": "2020-07-18T10:31:57+00:00", "channel": "ideas"}, "reason": "named only, its text reads "
        "blooper-tentacle's better (0.79 vs 0.18)"}]))
    put(("oldrows", "review.jsonl"), lines([{"id": "oldrows-a", "kind": "unsure map", "card": "Gone", "status":
                                            "archived", "rows": []},
                                           {"id": "oldrows-b", "kind": "unsure map", "card": "Bolt", "status": "live",
                                            "rows": []}]))
    put(("chain", "review.jsonl"), lines([{"id": "chain-f1", "kind": "field", "card": "bolt", "entry": "D-x",
                                           "field": "stats", "proposed": {"hp": 6, "def": 1}},
                                          {"id": "chain-f2", "kind": "field", "card": "kit", "entry": "D-y",
                                           "field": "stats", "proposed": {"creatures": [{"hp": 7, "def": 4, "atk": 6,
                                                                                        "spd": 6}] * 2}},
                                          {"id": "chain-l1", "kind": "look not told", "card": "bolt", "picture": "P-1",
                                           "near": 1},
                                          {"id": "chain-f3", "kind": "field", "card": "bolt", "entry": "D-ai",
                                           "field": "stars", "proposed": 9}]))
    put(("preview", "eye.json"), {"leave_out": {"bolt|D-ai": "an AI restyle"}})
    put(("chain", "created.json"), {"cards": {
        "Bolt": {"card": "bolt", "status": "later", "date": "2020-05-01", "sort": "2020-05-01"},
        "Kit": {"card": "kit", "status": "owner", "date": "2020-01-01", "sort": "2020-01-01",
                "owner": {"base": "2020-09-01"}}}})
    hint = lambda n, card, date, **x: {"entry": f"LC-{n}", "card": card, "date": date, "channel": "port", **x}
    put(("morgue", "candidates.json"), {"candidates": [], "not_candidates": [
        {"id": f"LC-{n}", "name": "x", "pictures": [{"id": f"P-m{n}", "date": "2020-04-01T00:00:00+00:00"}]}
        for n in range(4)], "created_hints": [
        hint(0, "bolt", "2020-06-01T00:00:00+00:00"), hint(1, "bolt", "2020-04-01T00:00:00+00:00"),
        hint(2, "bolt", "2019-01-01T00:00:00+00:00", meme=True), hint(3, "kit", "2020-08-01T00:00:00+00:00")]})
    stale = lambda rid, **x: {"kind": "field", "choice": "yes", "item": tr.item("field", rid, "matches", "fields", rid,
                                                                              **x)}
    number = {"kind": "look not told", "choice": "look", "value": 1}         # an older page's look number
    put(("review", "rulings.json"), {"rulings": {"chain-old1": stale("chain-old1", stale=True),
                                                 "chain-old2": stale("chain-old2"),
                                                 "chain-l1": number,                 # a yes on what it proposed then
                                                 "chain-f1": {"kind": "field", "choice": "yes",
                                                              "value": {"hp": 6, "def": 2}}}})
    put(("chain", "ruled.json"), {"used": ["chain-old1"]})
    inputs = tr.Inputs(str(tmp_path))
    got = {i["id"]: i for i in tr.gather(inputs)}
    assert [c["c"] for c in got["oldrows-a"]["choices"]] == ["no"] and got["oldrows-a"]["other"]["c"] == "card"
    assert "chain-f3" not in got                                                   # no version (the preview's eye)
    assert [(c["c"], c.get("value")) for c in got["oldrows-b"]["choices"]] == [("yes", "Bolt"), ("no", None)]
    assert [c.get("value") for c in got["match:P-1"]["choices"]] == ["tentacle", "blooper-tentacle", None]
    assert sorted(i for i in got if i.startswith("hint:")) == ["hint:LC-1:bolt", "hint:LC-3:kit"]
    assert got["hint:LC-1:bolt"]["picture"] == "P-m1"
    assert (got["chain-old1"]["gone"], got["chain-old2"]["gone"]) == ("taken", "changed")
    assert "taken in by the rerun" in tr.item_html(got["chain-old1"]) and \
        "Not applied" in tr.item_html(got["chain-old2"])
    assert "changed" not in tr.item_html(got["chain-old1"]).split(">")[0]        # taken: its stored copy's mark gone
    page = tr.page(list(got.values()), inputs, "now")
    assert "<option value='kit2' label='Kit (kit2)'>" in page and "<option value='Bolt'>" in page
    assert "1 answered and applied, 3 answered but not applied" in page
    names = {cid: c["name"] for cid, c in cards.items()}
    assert (ar.parse("card", "kit2", names), ar.parse("card", "Tentacle", names)) == ("kit2", "tentacle")
    try:
        ar.parse("card", "Kit", names)
        raise AssertionError("a shared name taken")
    except ValueError as e:
        assert "type the card's id" in str(e)
    assert (got["chain-f1"]["other"]["shape"], got["chain-f2"]["other"]["shape"]) == ("hp/def", "creatures")
    good, _, errors = ar.check({"chain-f1": {"choice": "value", "value": "6/2"},
                                "chain-f2": {"choice": "value", "value": "7/4/6/6 | 7/0/4/4"}}, got, {}, names)
    assert not errors and good["chain-f1"]["value"] == {"hp": 6, "def": 2} and \
        good["chain-f2"]["value"]["creatures"][1] == {"hp": 7, "def": 0, "atk": 4, "spd": 4}
    assert ar.check({"chain-f2": {"choice": "value", "value": "6/2"}}, got, {}, names)[2]       # one creature, 2 stats
    # a look's radio is its first version's key; a stored number no radio takes is not applied, kept as stored
    assert [c.get("value") for c in got["chain-l1"]["choices"]] == [key("a"), key("b"), "bolt", "bolt"]
    assert got["chain-l1"]["stale"] and "Not applied" in tr.item_html(got["chain-l1"])
    assert not any(i.get("stale") for i in got.values() if i["id"] not in ("chain-l1", "chain-f1"))
    # a field's yes carries its value: the rerun proposes 6/1 now, so the stored 6/2 is not applied
    assert got["chain-f1"]["choices"][0] == {"c": "yes", "label": "yes: 6/1/?/?", "value": {"hp": 6, "def": 1}}
    assert got["chain-f1"]["stale"] and "data-v='{&quot;hp&quot;: 6, &quot;def&quot;: 1}'" in \
        tr.item_html(got["chain-f1"])
    assert tr.offered(got["chain-f1"], {"choice": "yes", "value": {"def": 1, "hp": 6}})
    answer = {"chain-l1": {"choice": "look", "value": 1}}
    assert ar.check(answer, got, {"chain-l1": number}, names)[0]["chain-l1"] == number and ar.check(answer, got, {},
                                                                                                    names)[2]
    assert not ar.check({"chain-l1": {"choice": "look", "value": key("a")}}, got, {"chain-l1": number}, names)[2]


def test_the_owner_s_later_answers_show_read_only():
    from types import SimpleNamespace
    answers = {"a1": {"id": "a1", "when": "2026-11-13 14:30", "title": "Jokes", "answer": "their own", "ruling": "x"},
               "a2": {"id": "a2", "when": "2026-11-13 23:15", "title": "Meme", "answer": "not sure", "ruling": "y"}}
    inputs = SimpleNamespace(answers=answers, owner={"read_only": {"morgue:LC-1": "a2", "morgue:LC-2": "a1"},
                                                     "notes": {"chain-1": {"answer": "a2", "note": "left open"}}})
    assert tr.owner_answer(inputs, "morgue:LC-1")["id"] == "a2"
    assert tr.owner_answer(inputs, "morgue:LC-2", {"kind": "joke", "nearest": {"by": "eye"}})["id"] == "a1"
    assert tr.owner_answer(inputs, "morgue:LC-3", {"kind": "joke", "nearest": {"by": "eye"}}) is None   # not listed
    items = [{"id": "chain-1", "meta": []}, {"id": "chain-2", "meta": []}]
    tr.owner_notes(inputs, items)
    assert items[0]["meta"] == ["The owner's answer (2026-11-13 23:15): Meme = not sure: left open"] and \
        items[1]["meta"] == []
