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
    assert sum(" refused" in line for line in logs) == 1 and len(logs) == 6          # every error named, none kept
    assert "chain-q1" not in " ".join(logs)                                          # read only: skipped, no error
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
    assert [tr.place(x) for x in ({"channel": "port"}, {"channel": "old repo", "source": "old repo"}, {})] == [
        "#port", "(old repo)", ""]                                                    # the old repo: a source
    assert [tr.nearest_text({"nearest": n}) for n in ({"name": "Bolt", "bits": 6.0}, {"name": "Bolt", "by": "eye"})] == [
        "nearest live card: Bolt (6.0 bits)", "nearest live card: Bolt (by eye)"]
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
        "blooper-tentacle's better (0.79 vs 0.18)"},
        {"id": "P-6", "verdict": "draft", "card": "bolt", "file": "attachments/6.png", "thumb": "t/6.webp"}]))
    from PIL import Image
    (tmp_path / "attachments").mkdir()
    shot = Image.new("RGB", (200, 100), (40, 40, 40))
    shot.paste(Image.new("RGB", (60, 90), (200, 30, 30)), (120, 5))
    shot.putpixel((0, 0), (255, 255, 255))                                       # no uniform border: not trimmed
    shot.save(tmp_path / "attachments" / "6.png")
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
                                           "field": "stars", "proposed": 9},
                                          {"id": "chain-d1", "kind": "dates only", "card": "bolt", "picture": "P-5",
                                           "date": "2020-07-19T00:00:00+00:00", "channel": "ideas",
                                           "why": "a sheet of two cards", "also": ["kit"], "read_only": True},
                                          {"id": "chain-d2", "kind": "dates only", "card": "kit", "picture": "P-7",
                                           "date": "2019-11-01T00:00:00+00:00", "channel": "ideas",
                                           "why": "bare art", "read_only": True},
                                          {"id": "chain-k1", "kind": "cropped", "card": "bolt", "picture": "P-6",
                                           "date": "2020-07-20T00:00:00+00:00", "channel": "ideas",
                                           "why": "a TTS table: the card cut out", "read_only": True}]))
    put(("match", "eye.json"), {"dates_only": {"P-5": "a sheet of two cards", "P-7": "bare art"},
                                "crops": {"P-6": {"box": [0.6, 0.05, 0.9, 0.95], "why": "a TTS table"}}})
    put(("preview", "eye.json"), {"leave_out": {"bolt|D-ai": "an AI restyle"}})
    put(("chain", "created.json"), {"cards": {
        "Bolt": {"card": "bolt", "status": "later", "date": "2020-05-01", "sort": "2020-05-01"},
        "Kit": {"card": "kit", "status": "owner", "date": "2020-01-01", "sort": "2020-01-01",
                "owner": {"base": "2020-09-01", "ruled": ["chain-h1"], "source": {"day": "2020-01-01", "from":
                                                                                    "changelog line"}},
                "ours": {"day": "2019-11-01", "picture": "P-7", "dates_only": True}}}})
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
                                                 "chain-x5": stale("chain-x5", card="bolt", entry="D-P-5"),
                                                 "chain-f1": {"kind": "field", "choice": "yes",
                                                              "value": {"hp": 6, "def": 2}},
                                                 "chain-h1": {"kind": "changelog earlier", "choice": "yes",
                                                              "source": "triage (owner-delegated)"}}})
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
    assert got["chain-d1"]["read_only"] and got["chain-d1"]["sub"] == "dates" and not got["chain-d1"]["choices"]
    assert got["chain-d1"]["meta"][2:] == [
        "Created: not set by this picture: Bolt's Created, 2020-05-01, comes from the website's date.",
        "Also printed on it (by eye), so it dates them too: Kit.",                # each card it dates, its own line
        "Created: not set by this picture: Kit's Created, 2020-01-01, comes from the triage's yes on a changelog "
        "line (chain-h1, 2020-01-01)."] and got["chain-d2"]["meta"][-1] == "Created: NOT set by this picture: it " \
        "gives 2019-11-01, but the triage's yes on a changelog line (chain-h1, 2020-01-01) sets it to 2020-01-01."
    assert got["chain-x5"]["gone"] == "taken" and "dates only" in got["chain-x5"]["settled"]   # no card on its own
    assert got["chain-k1"]["read_only"] and got["chain-k1"]["sub"] == "crops" and \
        got["chain-k1"]["pics"][1]["src"] == "crops/P-6.webp"                    # a crop by eye: shown, read only
    assert tr.crop_thumbs(str(tmp_path), inputs) == 1
    with Image.open(derived / "review" / "crops" / "P-6.webp") as im:
        assert im.size == (60, 90)                                                 # the card alone, as the preview
    assert tr.picture_page(str(tmp_path), inputs, list(got.values()), "now") == 3
    shown = (derived / "review" / tr.PICTURES_PAGE).read_text()
    assert shown.count("data:image/webp;base64,") == 2 and "Cropped: " in shown and "Dates only: " in shown and \
        "NOT set by this picture" in shown and "so it dates them too: Kit." in shown and \
        "Answers on this picture (identification only): yes (owner)" in shown   # P-5 has no file here
    assert "taken in by the rerun" in tr.item_html(got["chain-old1"]) and \
        "Not applied" in tr.item_html(got["chain-old2"])
    assert "changed" not in tr.item_html(got["chain-old1"]).split(">")[0]        # taken: its stored copy's mark gone
    page = tr.page(list(got.values()), inputs, "now")
    assert "<option value='kit2' label='Kit (kit2)'>" in page and "<option value='Bolt'>" in page
    assert "3 answered and applied, 3 answered but not applied" in page        # chain-h1: Kit's hint, taken
    names = {cid: c["name"] for cid, c in cards.items()}
    assert (ar.parse("card", "kit2", names), ar.parse("card", "Tentacle", names)) == ("kit2", "tentacle")
    try:
        ar.parse("card", "Kit", names)
        raise AssertionError("a shared name taken")
    except ValueError as e:
        assert "type the card's id" in str(e)
    assert (got["chain-f1"]["other"]["shape"], got["chain-f2"]["other"]["shape"]) == ("hp/def", "creatures")
    good, _, errors, _ = ar.check({"chain-f1": {"choice": "value", "value": "6/2"},
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


def test_the_photo_created_dates_show_read_only_and_no_paper_items(tmp_path):
    """Owner Q6 (11-14 22:15/22:20): the physical card photos are no review item (no paper part); the Created
    'Before 2020' of the three shows read only with the answer, no radio; a typed 'Before 2020' elsewhere still parses."""
    assert "paper" not in {part[0] for parts in tr.SUBS.values() for part in parts}
    inputs = tr.Inputs(str(tmp_path))
    inputs.cards = {"pep-drink": {"key": "Pep Drink", "name": "Pep Drink"}}
    made = {"id": "chain-c9", "kind": "created earlier", "card": "pep-drink", "file": {"date": "2020-07-22",
                                                                                      "from": "discord"},
            "ours": {"date": None, "day": "Before 2020", "source": "physical card photo", "owner": "Q6"},
            "read_only": True, "answer": "Q6", "why": "the owner's ruling"}
    it = tr.created_item(inputs, made)
    assert it["read_only"] and not it["choices"] and it["other"] is None and not any(it["pics"])   # no photo
    assert "ours: Before 2020 from physical card photo" in it["meta"][0]
    assert any(m.startswith("The owner's answer Q6 (2026-11-14 22:20): Physical card photos") for m in it["meta"])
    assert "type=radio" not in tr.item_html({**it, "pics": []})                # gather drops the empty pictures
    asked = tr.created_item(inputs, {**made, "read_only": False})             # no ruling: asked as before
    assert [c["c"] for c in asked["choices"]] == ["ours", "file"] and not asked["read_only"]
    assert load().parse("date", "before 2020", {}) == "Before 2020"
    assert load().parse("types", '["Machine", "equip"]', {}) == ["equip", "machine"]   # a JSON-like list typed


def test_a_rebuild_s_settled_answers_skip_and_the_owner_s_win(tmp_path):
    """BRIEF 11-15 20:50 and the APPLY ORDER (owner > triage > its guesses): an answer whose item a rebuild settled or
    dropped (an id the page showed before: ids.json), one on an item now read only and one a stale export's value
    no radio offers now are listed and skipped, never an error; a mistyped id still refuses the file. The triage's
    source is kept and never replaces the owner's answer."""
    ar = load()
    review = tmp_path / "derived" / "review"
    review.mkdir(parents=True)
    its = items()
    its[2]["read_only"] = True                                     # Created: the owner's answer now
    (review / "items.json").write_text(json.dumps({"format": tr.FORMAT, "cards": {"bolt": "Bolt"}, "items": its}))
    tr.register(str(tmp_path), [*its, tr.item("look not told", "chain-gone", "matches", "looks", "Bolt: 2020",
                                              card="bolt")], "2026-11-16T03:46:55+00:00")
    logs = []
    run = lambda e: ar.apply(e, str(tmp_path), build=False, log=logs.append)
    stored = lambda: json.loads((review / "rulings.json").read_text())["rulings"]
    export = lambda source=None, **answers: {"format": tr.FORMAT, "page": "2026-11-16T03:46:55+00:00",
                                             "rulings": answers, **({"source": source} if source else {})}
    assert run(export(**{"chain-typo": {"choice": "no"}})) == 1                     # never shown: an error
    stale = {"chain-gone": {"choice": "draft", "value": "bolt"}, "chain-c1": {"choice": "ours"},
             "chain-f1": {"choice": "yes", "value": {"hp": 9, "def": 2, "atk": 6, "spd": 4}},
             "match:P-1": {"choice": "card", "value": "bolt"}}
    assert run(export(**stale)) == 0
    assert set(stored()) == {"match:P-1"} and "source" not in stored()["match:P-1"]
    told = " ".join(logs)
    assert "3 answer(s) skipped" in told and "chain-gone: left the page (last shown 2026-11-16T03:46:55+00:00: " \
        "Bolt: 2020)" in told and "chain-c1: read only now" in told and "chain-f1: 'yes' now offers another value" in told
    triage = "triage (owner-delegated 2026-11-16, two judges agreed)"
    assert run(export(triage, **{"match:P-1": {"choice": "none"}, "R1": {"choice": "no"}})) == 0
    got = stored()
    assert got["match:P-1"]["choice"] == "card" and got["R1"]["source"] == triage    # the owner's answer kept
    assert "match:P-1: kept: owner's answer" in " ".join(logs)
    assert run(export("triage guess (owner-delegated)", **{"R1": {"choice": "yes", "value": "points"}})) == 0
    assert stored()["R1"]["choice"] == "no"                                          # the triage's over its guess
    assert run(export(**{"R1": {"choice": "no"}})) == 0 and "source" not in stored()["R1"]  # the same: owner's now
    ids = json.loads((review / "ids.json").read_text())["ids"]
    tr.register(str(tmp_path), [tr.item("look not told", "chain-gone", "matches", "looks", "Pike: 2020",
                                        card="pike")], "2026-11-16T12:10:04+00:00")
    assert json.loads((review / "ids.json").read_text())["ids"]["chain-gone"] == ids["chain-gone"]  # meaning kept


def test_an_answer_on_a_dates_only_picture_is_kept_as_identification(tmp_path):
    """Owner 11-16 23:10: an answer naming what a picture with no card on its own shows (its look, a draft, a field)
    is stored as identification only even when its item left the page for that (match/eye.json dates_only); the
    page shows it taken, as identification only. Other left items still skip."""
    ar, dc = load(), tr.dc
    review = tmp_path / "derived" / "review"
    review.mkdir(parents=True)
    (tmp_path / "derived" / "match").mkdir()
    (tmp_path / "derived" / "match" / "eye.json").write_text(json.dumps({"dates_only": {"P-7": "a TTS table"}}))
    (tmp_path / "derived" / "match" / "verdicts.jsonl").write_text(json.dumps(
        {"id": "P-7", "thumb": "match/thumbs/p-7.webp"}) + "\n")
    (review / "items.json").write_text(json.dumps({"format": tr.FORMAT, "cards": {"bolt": "Bolt"}, "items": items()}))
    look, field = dc.review_id("look not told", "P-7"), dc.review_id("field", "bolt|D-P-7|stars")
    tr.register(str(tmp_path), [tr.item("look not told", look, "matches", "looks", "Bolt: 2020", card="bolt"),
                                tr.item("field", field, "matches", "fields", "Bolt: stars", card="bolt"),
                                tr.item("look not told", "chain-gone", "matches", "looks", "Bolt: 2021",
                                        card="bolt")], "2026-11-16T03:46:55+00:00")
    logs = []
    answers = {look: {"choice": "look", "value": "b" * 21}, field: {"choice": "value", "value": "4"},
               "chain-gone": {"choice": "draft", "value": "bolt"}}
    assert ar.apply({"format": tr.FORMAT, "rulings": answers}, str(tmp_path), build=False, log=logs.append) == 0
    got = json.loads((review / "rulings.json").read_text())["rulings"]
    assert set(got) == {look, field} and "1 answer(s) skipped" in " ".join(logs)
    assert (got[look]["item"]["picture"], got[look]["item"]["identification"], got[field]["item"]["field"]) == \
        ("P-7", True, "stars")
    assert got[look]["item"]["pics"] == [{"src": "../matcher/match/thumbs/p-7.webp", "cap": "the picture"}]
    inputs = tr.Inputs.__new__(tr.Inputs)
    inputs.dates_only = {"P-7": "a TTS table"}
    assert "identification only" in tr.off_timeline(inputs, got[look])


def test_changelog_lines_take_a_card(tmp_path, monkeypatch):
    """Owner D9: an 'ambiguous card' line offers its candidates, none and a typed card; the owner's own answer
    (owner.json lines) shows read only. An item the owner answered outside the page (read_only) is never asked."""
    inputs = tr.Inputs(str(tmp_path))
    inputs.cards = {"metal-shield": {"key": "Metal Shield", "name": "Metal Shield"},
                    "lightning-bolt": {"key": "Lightning Bolt", "name": "Lightning Bolt"}}
    inputs.by_key = {c["key"]: cid for cid, c in inputs.cards.items()}
    inputs.answers = {"d9": {"id": "d9", "when": "2026-11-15 21:50", "title": "Lines", "answer": "a picker",
                             "ruling": "As told."}}
    inputs.owner = {"lines": {"R2": {"card": "lightning-bolt", "line": "5:img14", "answer": "d9"}},
                    "read_only": {"chain-c": "d9"}}
    row = lambda rid, cands, line: {"id": rid, "kind": "ambiguous card", "candidates": cands, "line": line,
                                    "date": "2019-03-31T01:47:37+00:00", "channel": "general-discussion"}
    inputs.fact_review = [row("R1", ["Mechanical Shield", "Metal Shield"], "5:doc10"), row("R2", [], "5:img14")]
    asked, told = tr.fact_items(inputs)
    assert [(c["c"], c.get("value")) for c in asked["choices"]] == [("card", "metal-shield"), ("none", None)]
    assert asked["other"]["kind"] == "card" and asked["line"] == "5:doc10" and not asked["read_only"]
    assert told["read_only"] and any("Lightning Bolt" in m for m in told["meta"])
    made = tr.item("created earlier", "chain-c", "created", "earlier", "Gust of Trees", card="gust-of-trees")
    made["choices"] = [tr.choice("ours", "ours: 2020-08-24", "2020-08-24")]
    inputs.chain, inputs.old_review, inputs.verdicts, inputs.morgue, inputs.ideas = [], [], {}, {}, []
    inputs.fact_review = []
    monkeypatch.setattr(tr, "match_items", lambda i: [made])    # one item from a part, the owner's answer over it
    got = tr.gather(inputs)
    assert got[0]["read_only"] and not got[0]["choices"] and "The owner's answer (2026-11-15 21:50)" in got[0]["meta"][-1]


def test_an_answer_the_rerun_gives_the_same_way_is_settled_not_orange():
    """s3 gap 2 fix: an answer whose item left the page but whose version still shows it (from the git look's own
    answer, a repost's entry, the same old blob's row) is taken in, not 'answer it again'; so are a picture's card, a
    Created date in use, a Created hint no earlier than it and a Morgue 'no'. Another value or a gone entry stays not
    applied."""
    from types import SimpleNamespace
    sure = lambda v: {"value": v, "confidence": "sure", "method": "triage"}
    ns = SimpleNamespace(
        timelines={"bolt": {"kind": "creature", "entries": [
            {"id": "D-P-1", "pictures": [{"id": "P-1"}, {"id": "P-2"}], "fields": {"ability": sure("Heal 2 HP.")}},
            {"id": "O-x", "kind": "old repo", "blob": "b1", "fields": {"stats": sure({"hp": 6, "def": 5})}}]}},
        verdicts={"P-9": {"card": "bolt"}}, created={"Bolt": {"date": "2020-01-01"}}, cards={"bolt": {"key": "Bolt"}},
        rulings={"chain-n": {"kind": "look not told", "choice": "no", "item": {"card": "bolt", "picture": "P-6"}}})
    field = lambda entry, value, c="value": {"kind": "field", "choice": c, "value": value, "item": {
        "kind": "field", "card": "bolt", "entry": entry, "field": "ability", "choices": [{"c": "yes", "value": "Heal 3 HP."}]}}
    assert tr.settled(ns, field("D-P-2", "heal 2 hp")) and tr.settled(ns, field("D-P-1", None, "no"))
    assert not tr.settled(ns, field("D-P-2", "Heal 3 HP.")) and not tr.settled(ns, field("D-P-7", "Heal 2 HP."))
    assert "not this card" in tr.settled(ns, field("D-P-6", None, "no"))     # its picture answered 'not this card'
    ns.verdicts["P-8"] = {"card": "bolt", "version": {"commits": ["c1"]}}           # a git look's picture, gone to its row
    ns.timelines["bolt"]["entries"].append({"id": "O-c1", "kind": "old repo", "commit": "c1",
                                            "fields": {"ability": sure("Heal 4 HP.")}})
    assert tr.settled(ns, field("D-P-8", "Heal 4 HP.")) and tr.settled(ns, field("O-c1", "heal 4 hp"))
    old = {"kind": "old field", "choice": "value", "value": {"hp": 6, "def": 5}, "item": {
        "kind": "old field", "card": "bolt", "field": "stats", "pics": [{"src": "../oldgit/thumbs/b1.webp"}]}}
    assert tr.settled(ns, old)
    assert tr.settled(ns, {"kind": "card not told", "choice": "card", "value": "bolt",
                           "item": {"kind": "card not told", "picture": "P-9"}})
    assert tr.settled(ns, {"kind": "created earlier", "choice": "file", "value": "2020-01-01",
                           "item": {"kind": "created earlier", "card": "bolt"}})
    ns.created_before = lambda cid: tr.Inputs.created_before(ns, cid)
    hint = lambda day: {"kind": "changelog earlier", "choice": "yes", "value": day,
                        "item": {"kind": "changelog earlier", "card": "bolt", "date": day + "T10:28:10+00:00"}}
    assert "no longer asked" in tr.settled(ns, hint("2020-07-21")) and not tr.settled(ns, hint("2019-07-21"))
    assert tr.settled(ns, {"kind": "morgue not candidate", "choice": "no", "item": {"kind": "morgue not candidate"}})
    assert not tr.settled(ns, {"kind": "morgue idea", "choice": "yes", "item": {"kind": "morgue idea"}})
    ns.dates_only = {"P-7": "two cards in one screenshot"}           # no card on its own: any answer on it taken
    assert "dates only" in tr.settled(ns, field("D-P-7", "Heal 2 HP.")) and \
        "dates only" in tr.settled(ns, field("D-P-7", None, "no")) and \
        not tr.off_timeline(ns, {"kind": "look not told", "choice": "no", "item": {"picture": "P-7"}})
