"""
Card ratings (TTSCardMaker site/Card Ratings.json): the TTS import (commands/tts_ratings.py: the board mapping, the
skips, an idempotent merge that keeps website entries), the file's layout (card_edits.CardRatings, dump_ratings) against
edit.js ratingsFile (node), the bot never reading it, and the website's side: build_site.py parse_ratings,
apply_ratings, rating_now and tier_of, ratings.js's helpers and chart. Loads the site from $AC_SITE_SRC (a TTSCardMaker
checkout) or the clone; those tests skip without it, before the ratings code, or without node.
"""
import importlib.util
import json
import shutil
import subprocess
from os import environ
from os.path import expanduser
from pathlib import Path

import pytest

import commands.card_sync as card_sync
import commands.tts_ratings as tr
from commands.card_edits import CardEdits, CardRatings, dump_ratings
from conftest import CLONE
from global_config import RATINGS_PATH

SITE = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE)))
BUILD_SITE = SITE / "site" / "build" / "build_site.py"
EDIT_JS = SITE / "site" / "assets" / "edit.js"
RATINGS_JS = SITE / "site" / "assets" / "ratings.js"
G, S = "group", "TTS session"


def _e(day, scale, value, by=G, source=S):
    return CardRatings.entry(day, scale, value, by, source)


def _row(save, board, x, category, card="Alpha", z=0.0, **kw):
    scale = tr.SCALE_OF[board]
    return {"save": save, "scale": scale, "board": board, "x": x, "z": z, "category": category, "card": card,
            "nick": card or kw.get("archived"), "png": f"Creatures/Other/2 Stars/{card or kw.get('archived')}.png",
            "match": "png", "match_note": "", "card_stars": 2, "card_author": "Maker", **kw}


@pytest.fixture(scope="module")
def bs():
    if not BUILD_SITE.is_file():
        pytest.skip(f"no card repo build_site at {BUILD_SITE}")
    spec = importlib.util.spec_from_file_location("site_build_site_ratings", BUILD_SITE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "parse_ratings"):
        pytest.skip(f"{BUILD_SITE} has no ratings: land TTSCardMaker work/r3-ratings")
    return module


def _node(tmp_path, body: str, src: Path = EDIT_JS):
    if not src.is_file() or shutil.which("node") is None:
        pytest.skip(f"{src.name} or node missing")
    if "ratings" not in src.read_text(encoding="utf-8"):
        pytest.skip(f"{src} has no ratings")
    script = tmp_path / "run.mjs"
    script.write_text(f"const m = await import({json.dumps(src.as_uri())});\n{body}\n", encoding="utf-8")
    done = subprocess.run(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# ========================
# The import
# ========================
def test_score_bands():
    even = (0, 10, 20, 30, 40)                    # labels centred in their bands: exactly 2i+1.5
    assert [tr.score(x, even)[1] for x in even] == [1.5, 3.5, 5.5, 7.5, 9.5]
    labels = tr.LABELS["centre"]
    assert [tr.score(x, labels) for x in labels] == [(0, 1.5), (1, 3.4), (2, 5.5), (3, 7.6), (4, 9.5)]
    mids = [(labels[i] + labels[i + 1]) / 2 for i in range(4)]
    assert [tr.score(m, labels)[1] for m in mids] == [2.5, 4.5, 6.5, 8.5]
    assert tr.score(-99, labels) == (0, 1.0) and tr.score(99, labels) == (4, 10.0)
    sweep = [tr.score(labels[0] - 6 + i * 0.1, labels)[1] for i in range(int((labels[4] - labels[0] + 12) * 10))]
    assert sweep == sorted(sweep)
    assert tr.score(-1.16, labels) == (2, 5.5)
    assert [tr.tier_of(v) for v in (1, 2.4, 2.5, 4.5, 6.5, 8.4, 8.5, 10)] == [0, 0, 1, 2, 3, 3, 4, 4]


def test_favorability_columns():
    """The reversi columns (37.72 + 4.567 per column) on the upper board."""
    got = [tr.score(37.72 + 4.567 * (c - 1), tr.LABELS["like-top"]) for c in range(1, 9)]
    assert [v for _, v in got] == [1.5, 2.7, 3.9, 5.0, 6.0, 7.1, 8.2, 9.5]
    assert [b for b, _ in got] == [0, 1, 1, 2, 2, 3, 3, 4]


def test_placements_skips_and_dates():
    raw = {"ratings": [
        _row("TS_Save_3", "like-bottom", 32.76, "Double Pain", card="Double Pain"),       # the joke column
        _row("TS_Save_3", "centre", -1.16, "Balens"),
        _row("TS_Save_3", "centre", -1.16, "Balens", card="Beta"),                       # same stack, same value
        _row("TS_Save_8", "centre", 9.71, "Above Average", z=-13.28),
        _row("TS_Save_10", "centre", 9.9, "Above Average", z=-13.0),                     # left where it was
        _row("TS_Save_8", "centre", 9.71, "Above Average", card="Beta", z=-13.28),
        _row("TS_Save_10", "centre", 4.57, "Balens", card="Beta", z=-13.16),             # moved
        _row("TS_Save_10", "like-top", 42.29, "Dislike", card=None, archived="Cho'Gath"),
    ], "forbidden": [{"save": "TS_Save_3", "card": "Kalecgos"}], "skipped": {}}
    entries, report = tr.placements(raw)
    got = [(e["card"], e["date"], e["scale"], e["value"]) for e in entries]
    assert got == [("Alpha", "2023-04-28", "power", 5.5), ("Beta", "2023-04-28", "power", 5.5),
                   ("Alpha", "2025-03-02", "power", 7.1), ("Beta", "2025-03-02", "power", 7.1),
                   ("Beta", "2025-04-28", "power", 6.3), ("Cho'Gath", "2025-04-28", "favorability", 2.7)]
    assert all(e["by"] == G and e["source"] == S for e in entries)
    assert [r["card"] for r in report["dropped"]] == ["Double Pain"] and len(report["carried"]) == 1
    assert report["forbidden"] == raw["forbidden"] and report["cards"] == 3
    with pytest.raises(ValueError, match="Balens"):
        tr.placements({"ratings": [_row("TS_Save_3", "centre", 9.71, "Balens")]})


def test_merge_is_idempotent_and_keeps_website_entries():
    text = dump_ratings({"about": CardRatings.ABOUT, "cards": {"alpha": [
        _e("2023-04-28", "power", 2, G, S),                     # stale: replaced
        _e("2026-10-24", "power", 6, "someone", "website"),
        _e("2024-01-01", "favorability", 4, G, S)]}})          # another date: kept
    entries = [{"card": "Alpha", **_e("2023-04-28", "power", 5.5)}, {"card": "Gamma", **_e("2025-04-28", "favorability", 7.1)}]
    once = tr.merge(text, entries)
    cards = json.loads(once)["cards"]
    assert list(cards) == ["alpha", "Gamma"]
    assert cards["alpha"] == [_e("2023-04-28", "power", 5.5), _e("2024-01-01", "favorability", 4),
                              _e("2026-10-24", "power", 6, "someone", "website")]
    assert tr.merge(once, entries) == once
    assert json.loads(tr.merge(None, entries))["about"] == CardRatings.ABOUT


def test_dump_ratings_layout():
    data = {"about": "x", "cards": {"Zeta": [_e("2023-04-28", "power", 7.0)],
                                    "Éclair": [_e("2025-04-28", "favorability", 8.2, "Ünï", "website")]}}
    assert dump_ratings(data) == (
        '{\n  "about": "x",\n  "cards": {\n'
        '    "Éclair": [{"date":"2025-04-28","scale":"favorability","value":8.2,"by":"Ünï","source":"website"}],\n'
        '    "Zeta": [{"date":"2023-04-28","scale":"power","value":7,"by":"group","source":"TTS session"}]\n  }\n}\n')
    parsed = CardRatings.parse(dump_ratings(data))
    assert parsed.cards["zeta"] == ("Zeta", data["cards"]["Zeta"])
    bad = [{**_e("2023-04-28", "power", 5), **change} for change in
           ({"value": 0}, {"value": 10.5}, {"value": True}, {"date": "2023-02-30"}, {"scale": "taste"}, {"source": "Discord"}, {"by": ""})]
    assert CardRatings.parse(json.dumps({"cards": {"A": bad + [_e("2023-04-28", "power", 5)], "B": "no"}})).cards == \
        {"a": ("A", [_e("2023-04-28", "power", 5)])}
    assert [CardRatings.number(v) for v in (7.25, 7.0, 8.2, 1.04)] == [7.3, 7, 8.2, 1]


def test_bot_ignores_ratings():
    assert RATINGS_PATH not in card_sync.CARD_FILES and not card_sync.card_input(RATINGS_PATH)
    shaped = dump_ratings({"cards": {"Alpha": [_e("2023-04-28", "power", 5)]}})
    assert CardEdits.parse(shaped).cards == {}
    workflows = SITE / ".github" / "workflows"
    if not workflows.is_dir():
        pytest.skip(f"no workflows at {workflows}")
    for name in ("card-check", "card-sync", "card-edits", "card-guards", "tts-save"):
        assert "Card Ratings.json" not in (workflows / f"{name}.yml").read_text(encoding="utf-8"), name


# ========================
# The website build
# ========================
def test_parse_ratings_validates(bs, tmp_path, monkeypatch, capsys):
    (tmp_path / "site").mkdir()
    monkeypatch.setattr(bs, "REPO", str(tmp_path))
    assert bs.parse_ratings() == {}
    assert "no ratings" in capsys.readouterr().out
    good = [_e("2023-04-28", "power", 5.55), _e("2023-04-28", "favorability", 7)]
    (tmp_path / RATINGS_PATH).write_text(json.dumps({"cards": {"Al_pha": good + [{"value": 11}], "Beta": "no"}}), encoding="utf-8")
    got = bs.parse_ratings()
    assert got == {"al pha": ("Al_pha", [_e("2023-04-28", "power", 5.6), _e("2023-04-28", "favorability", 7)])}
    out = capsys.readouterr().out
    assert "entries skipped on 2 cards" in out and "Al_pha, Beta" in out
    (tmp_path / RATINGS_PATH).write_text("{", encoding="utf-8")
    assert bs.parse_ratings() == {} and "isn't valid JSON" in capsys.readouterr().out


def test_apply_ratings(bs):
    cards = [{"key": "Alpha", "id": "alpha"}, {"key": "Beta", "id": "beta"}]
    archived = [{"card": "Cho'Gath"}, {"card": "Old Beta", "renamed_to": "Beta", "renamedId": "beta"}]
    ratings = {"alpha": ("Alpha", [_e("2025-04-28", "power", 6, "somelogin", "website"), _e("2023-04-28", "power", 5)]),
               "old beta": ("Old Beta", [_e("2023-04-28", "favorability", 3)]),
               "beta": ("Beta", [_e("2025-03-02", "power", 7)]),
               "cho'gath": ("Cho'Gath", [_e("2023-04-28", "power", 5.5)]),
               "nobody": ("Nobody", [_e("2023-04-28", "power", 1)])}
    people, by_login = {"somelogin": "Some Person"}, {}
    got = bs.apply_ratings(cards, ratings, archived, people, by_login)
    assert got == {"entries": 5, "cards": 2, "archived": 1, "unmatched": ["Nobody"]}
    assert cards[0]["ratings"] == [_e("2023-04-28", "power", 5), {**_e("2025-04-28", "power", 6, "Some Person", "website"), "login": "somelogin"}]
    assert cards[1]["ratings"] == [_e("2023-04-28", "favorability", 3), _e("2025-03-02", "power", 7)]
    assert archived[0]["ratings"] == [_e("2023-04-28", "power", 5.5)] and "ratings" not in archived[1]
    by_login = {"somelogin": {"login": "somelogin", "author": "Profile Name"}}
    cards = [{"key": "Alpha", "id": "alpha"}]
    bs.apply_ratings(cards, {"alpha": ratings["alpha"]}, [], people, by_login)
    assert [e["by"] for e in cards[0]["ratings"]] == [G, "Profile Name"]
    cards = [{"key": "Kit", "id": "kit"}, {"key": "Kit2", "id": "kit-2", "dup": "kit"}]          # a rated duplicate PNG
    got = bs.apply_ratings(cards, {"kit2": ("Kit2", [_e("2023-04-28", "power", 5)])}, [], people, {})
    assert cards[0]["ratings"] == [_e("2023-04-28", "power", 5)] and "ratings" not in cards[1] and got["cards"] == 1


def test_rating_now(bs):
    entries = [_e("2023-04-28", "power", 7), _e("2024-01-01", "power", 5, "Ann", "website"),
               _e("2025-04-28", "power", 6.3), _e("2025-05-01", "power", 6, "Ann", "website"),
               _e("2025-05-01", "favorability", 9, "Ann", "website")]
    assert bs.rating_now(entries, "power") == {"value": 6.2, "raters": 2, "date": "2025-05-01"}     # (6.3 + 6) / 2 = 6.15
    assert bs.rating_now(entries, "power", day="2024-06-01") == {"value": 6, "raters": 2, "date": "2024-01-01"}
    assert bs.rating_now(entries, "power", skip="Ann") == {"value": 6.3, "raters": 1, "date": "2025-04-28"}
    assert bs.rating_now(entries[:1], "power") == {"value": 7, "raters": 1, "date": "2023-04-28"}
    assert bs.rating_now(entries, "favorability", skip=["Ann"]) is None and bs.rating_now([], "power") is None
    assert [bs.tier_of(v) for v in (1, 2.4, 2.5, 4.5, 6.5, 8.4, 8.5, 10)] == [0, 0, 1, 2, 3, 3, 4, 4]


def test_the_import_matches_the_sites_tiers(bs):
    """Every category of the import is the site's tierOf of its value (amendment M7)."""
    assert [bs.tier_of(v) for v in range(1, 11)] == [tr.tier_of(v) for v in range(1, 11)]
    assert bs.RATINGS_PATH == RATINGS_PATH and tuple(bs.RATING_SCALES) == CardRatings.SCALES
    assert tuple(bs.RATING_SOURCES) == CardRatings.SOURCES


# ========================
# The website (node)
# ========================
def test_ratings_file_matches_python(tmp_path):
    data = {"about": CardRatings.ABOUT, "cards": {
        "Alpha": [_e("2023-04-28", "power", 5.5), _e("2026-10-24", "power", 4, "tester", "website"),
                  _e("2026-10-24", "power", 8, "other", "website")],
        "Éclair": [_e("2025-04-28", "favorability", 8.2)]}}
    text = dump_ratings(data)
    ops = [{"key": "alpha", "set": {}, "unset": [], "rate": {"power": 6.04, "favorability": 7.5}},
           {"key": "New Card", "set": {}, "unset": [], "rate": {"power": 2}}]
    got = _node(tmp_path, f"""
const text = {json.dumps(text)}, ops = {json.dumps(ops)};
let bad = ""; try {{ m.ratingsFile(text, [{{ key: "Alpha", rate: {{ power: 11 }} }}], "2026-10-24", "tester"); }} catch (e) {{ bad = e.message; }}
let broken = ""; try {{ m.ratingsFile("{{", ops, "2026-10-24", "tester"); }} catch (e) {{ broken = e.message; }}
process.stdout.write(JSON.stringify({{ same: m.ratingsFile(text, [], "2026-10-24", "tester"), out: m.ratingsFile(text, ops, "2026-10-24", "Tester"),
  fresh: m.ratingsFile(null, ops.slice(1), "2026-10-24", "tester"), about: m.RATINGS_ABOUT, bad, broken }}));""")
    assert got["same"] == text and got["about"] == CardRatings.ABOUT and got["bad"] == "Ratings go from 1 to 10"
    assert "isn't valid JSON" in got["broken"]
    out = json.loads(got["out"])["cards"]
    assert out["Alpha"] == [_e("2023-04-28", "power", 5.5), _e("2026-10-24", "power", 8, "other", "website"),
                            _e("2026-10-24", "favorability", 7.5, "Tester", "website"),
                            _e("2026-10-24", "power", 6, "Tester", "website")]
    assert out["New Card"] == [_e("2026-10-24", "power", 2, "Tester", "website")]
    want = {**data, "cards": {**data["cards"], "Alpha": out["Alpha"], "New Card": out["New Card"]}}
    assert got["out"] == dump_ratings(want)
    assert json.loads(got["fresh"])["about"] == CardRatings.ABOUT


def test_rating_only_batch_leaves_card_edits(tmp_path):
    edits = '{\n  "cards": {\n    "Alpha": {"hp":3,"_by":"x","_at":"y"}\n  }\n}\n'
    got = _node(tmp_path, f"""
const op = {{ id: "alpha", key: "Alpha", name: "Alpha", set: {{}}, unset: [], rate: {{ power: 6 }} }};
process.stdout.write(JSON.stringify({{ edits: m.editsFile({json.dumps(edits)}, [op], "2026-10-24T00:00:00Z", "tester"),
  empty: m.opEmpty(op), items: m.describeItems(op), without: m.withoutItem(op, "rate-power", "Alpha") }}));""")
    assert got["edits"] == edits and got["empty"] is False
    assert [(x["item"], x["text"], x["short"], x["psd"]) for x in got["items"]] == [("rate-power", "Power: rated 6", "Power 6", False)]
    assert "rate" not in got["without"]


def test_ratings_js_helpers(tmp_path):
    entries = [_e("2023-04-28", "power", 7), _e("2023-04-28", "favorability", 3.9), _e("2025-03-02", "power", 7.1),
               _e("2025-04-28", "power", 6.3), _e("2026-10-24", "power", 6, "Ann", "website")]
    got = _node(tmp_path, f"""
const list = {json.dumps(entries)};
const one = m.chartSvg(list.slice(1, 2), "favorability"), two = m.chartSvg(list, "power");
process.stdout.write(JSON.stringify({{
  now: m.current(list, "power"), asOf: m.current(list, "power", {{ day: "2025-03-02" }}), skip: m.current(list, "power", {{ skip: "Ann" }}),
  latest: [...m.latestBy(list, "power").keys()], dates: m.ratingDates(list, "power"), none: m.current(list, "nope"),
  personal: [m.hasPersonal([{{ ratings: list }}]), m.hasPersonal([{{ ratings: list.slice(0, 4) }}]), m.hasPersonal([{{ ratings: list }}], "favorability")],
  tiers: [1, 2.4, 2.5, 4.5, 6.5, 8.4, 8.5, 10].map(m.tierOf), labels: m.TIERS,
  one: [(one.match(/class="end"/g) || []).length, one.includes('class="line"'), one.includes('class="mid"')],
  two: [two.includes('class="line"'), two.includes('class="mid"'), (two.match(/class="person"/g) || []).length],
  empty: m.blockHtml({{ id: "x" }}), editing: m.blockHtml({{ id: "x" }}, {{ editing: true }}),
  block: m.blockHtml({{ id: "x", ratings: list }}), day: m.dayLabel("2025-04-28") }}));""", RATINGS_JS)
    assert got["now"] == {"value": 6.2, "raters": 2, "date": "2026-10-24"}       # (6.3 + 6) / 2 = 6.15
    assert got["asOf"] == {"value": 7.1, "raters": 1, "date": "2025-03-02"} and got["skip"]["value"] == 6.3
    assert got["latest"] == ["group", "Ann"] and got["dates"] == ["2023-04-28", "2025-03-02", "2025-04-28", "2026-10-24"]
    assert got["none"] is None and got["personal"] == [True, False, False]
    assert got["tiers"] == [0, 0, 1, 2, 3, 3, 4, 4] and got["labels"]["power"][0] == "Plz buff"
    assert got["one"] == [1, False, False] and got["two"] == [True, True, 1]
    assert got["empty"] == "" and "Not rated yet." in got["editing"] and 'data-action="rate-add"' in got["editing"]
    assert "<h3>Ratings</h3>" in got["block"] and "All ratings" in got["block"] and "rate-add" not in got["block"]
    assert "2025" in got["day"]


def test_a_re_rating_counts_its_rater_once(tmp_path):
    """An unpublished entry ("you") with rater = the person's published name replaces their earlier one (edit.js applyOp)."""
    entries = [_e("2025-04-28", "power", 7), _e("2026-10-20", "power", 3, "tester", "website"),
               {**_e("2026-10-24", "power", 9, "you", "website"), "rater": "tester", "_staged": True}]
    got = _node(tmp_path, f"""
const list = {json.dumps(entries)};
process.stdout.write(JSON.stringify([m.current(list, "power"), m.current(list, "power", {{ day: "2026-10-21" }}), m.current(list.slice(0, 2), "power")]));""", RATINGS_JS)
    assert got == [{"value": 8, "raters": 2, "date": "2026-10-24"}, {"value": 5, "raters": 2, "date": "2026-10-20"},
                   {"value": 5, "raters": 2, "date": "2026-10-20"}]
