"""discord_facts on made-up cards and lines: field changes, ability marks without text, cautious names, build."""
import json
from collections import Counter

import commands.discord_facts as F

TYPES = {"fire", "water", "machine", "protective"}


def names() -> F.Names:
    live = {"Iron Golem": ["Iron Golem"], "Sun Cookie": ["Sun Cookie"], "Gloom": ["Gloom"], "Mega Bot": ["Mega Bot"],
            "Bolt": ["Bolt"], "Twin Pike": ["Twin Pike"], "Pike": ["Pike"], "Shard": ["Shard"],
            "Crowned Mega Bot": ["Crowned Mega Bot"]}
    old = {"Mega Bot": ["Mecha Bot"], "Twin Pike": ["Pike"]}       # an old name another card has today
    return F.Names(live, old, ["Old Relic"], with_stats=set(live) - {"Sun Cookie", "Gloom"},
                   dictionary={"bolt", "gloom", "pike", "shard"}, statuses={"Gloom"}, no_ability={"Shard"})


def facts(line: str, source: str = "changelog", carry=None) -> list:
    got = F.parse_line(line, names(), TYPES, source, carry)
    return [{k: v for k, v in f.items() if v is not None and k != "doubt"} for f in got["facts"]]


def test_stat_changes():
    assert facts("Iron Golem: DEF 3 -> 5, SPD +1") == [
        {"card": "Iron Golem", "field": "def", "from": 3, "to": 5, "sure": True},
        {"card": "Iron Golem", "field": "spd", "by": 1, "sure": True}]
    assert facts("Nerf Iron Golem ATK by 2") == [{"card": "Iron Golem", "field": "atk", "by": -2, "sure": True}]
    assert [f["to"] for f in facts("Iron Golem (4/3/2/6 now, summoned by Somebody)")] == [4, 3, 2, 6]
    assert facts("Iron Golem - 3 Star") == [{"card": "Iron Golem", "field": "stars", "to": 3, "sure": True}]


def test_stars_types_and_names():
    assert facts("Sun Cookie now has 4 stars, not 5") == [
        {"card": "Sun Cookie", "field": "stars", "from": 5, "to": 4, "sure": True}]
    assert facts("Remove fire type from Iron Golem") == [
        {"card": "Iron Golem", "field": "type", "to": "fire", "op": "remove", "sure": True}]
    assert facts("Rename Mecha Bot to Mega Bot") == [
        {"card": "Mega Bot", "field": "name", "from": "Mecha Bot", "to": "Mega Bot", "sure": True}]
    assert facts("Mecha Bot: +2 HP") == [{"card": "Mega Bot", "field": "hp", "by": 2, "sure": True}]
    assert F.commit_cards("crowned mecha bot", names()) == [] and F.commit_cards("mecha bot", names()) == ["Mega Bot"]


def test_ability_marks_carry_no_text():
    line = "Sun Cookie: reword its ability to heal two more"
    got = facts(line)
    assert got == [{"card": "Sun Cookie", "kind": "ability changed", "sure": True}]
    assert not {w.lower() for w in line.split()} & {str(v).lower() for f in got for v in f.values()} - {"sun"}
    assert facts("Sun Cookie: +5 DEF") == [{"card": "Sun Cookie", "kind": "ability changed", "sure": True}]


def test_ability_marks_only_sure_on_the_subject():
    unsure = lambda card: [{"card": card, "kind": "ability changed", "sure": False}]
    summon = "Each turn it may pay 4 points to summon one Mega Bot."
    assert facts(summon) == unsure("Mega Bot")
    assert facts("Lantern now heals 2 HP and will give them 2 Gloom") == unsure("Gloom")
    assert facts('let Robo specify "Sun Cookie" as its target') == unsure("Sun Cookie")
    assert facts("cut the point cost for generating one sun cookie") == unsure("Sun Cookie")
    assert facts('Shard: "(BOT) summons one shard behind the enemy."') == unsure("Shard")   # no ability
    sure = lambda card: [{"card": card, "kind": "ability changed", "sure": True}]
    assert facts("Double the charge on Sun Cookie's ability") == sure("Sun Cookie")
    assert facts("Clarify that Iron Golem ignores shields") == sure("Iron Golem")
    assert facts("Sun Cookie: 4 uses instead of 5") == sure("Sun Cookie")
    assert F.parse_line(summon, names(), TYPES, "changelog")["facts"][0]["doubt"] == F.DOUBT["position"]


def test_effect_numbers_and_notes():
    assert facts("Iron Golem: +2 DEF (OYT), not (EOT)") == [
        {"card": "Iron Golem", "kind": "ability changed", "sure": False}]      # beside a timing tag
    assert facts("Iron Golem: make its +2 spd gain be +1 HP") == [
        {"card": "Iron Golem", "kind": "ability changed", "sure": True}]
    assert facts("The shield it's ability gives lasts one more turn (wrong it's)", carry=[["Iron Golem"]]) == []
    assert facts("Iron Golem should use the old art") == []
    assert facts("Iron Golem: it doesn't protect Bolt, who summons it, any more (non-protective minion)") == []
    assert facts("Pike: +3 ATK -1 DEF", "chat") == []                # a shared name picks nothing in chat
    assert facts("Whenever it is hit, Iron Golem gains +1 SPD", "chat") == []


def test_cautious_names():
    assert facts("lower Iron Golem's minion stars from 6 to 5 in the gloom") == []
    assert facts("inflict gloom") == [] and facts("Bolt: +1 ATK") == [{"card": "Bolt", "field": "atk", "by": 1,
                                                                         "sure": True}]
    assert facts("what if Iron Golem had 7 SPD?", "chat") == []
    got = F.parse_line("Pike: SPD 2 -> 3", names(), TYPES, "changelog")
    assert got["facts"] == [] and got["unsure"] == ["Pike", "Twin Pike"]
    lines = ["Mega Bot", "-2 HP", "+1 DEF"]                      # a heading, then its lines
    carry, out = None, []
    for line in lines:
        got = F.parse_line(line, names(), TYPES, "changelog", carry)
        carry = got["heading"] or (carry if got.get("carried") or got["why"] == "no card" else None)
        out += [(f["card"], f["field"], f["by"]) for f in got["facts"]]
    assert out == [("Mega Bot", "hp", -2), ("Mega Bot", "def", 1)]


def test_build_writes_facts_without_text(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    (root / "messages").mkdir(parents=True)
    (root / "derived" / "catalogue").mkdir(parents=True)
    (root / "derived" / "catalogue" / "channels.json").write_text(json.dumps({"channels": []}))
    (root / "channels.json").write_text(json.dumps([{"id": "301", "name": "update-01"}, {"id": "302", "name": F.GITHUB}]))
    monkeypatch.setattr(F, "AUTHORS", str(tmp_path / "authors.json"))
    (tmp_path / "authors.json").write_text(json.dumps({"someone": "Somebody"}))
    update = "301"
    rows = [{"id": "1", "timestamp": "2021-03-04T05:06:07+00:00", "author": {"bot": False},
             "content": "Iron Golem: zany wording tweak to its ability\nIron Golem: ATK 4 -> 5\nIron Golem: reword"},
            {"id": "4", "timestamp": "2021-04-04T05:06:07+00:00", "author": {"bot": False},
             "content": "Iron Golem: ATK 4 -> 5\nBolt: 3 stars"},  # a repost; a 'to' repeat may be a 2nd change
            {"id": "5", "timestamp": "2021-02-04T05:06:07+00:00", "author": {"bot": False},
             "content": "Bolt: 3 stars\nBolt: +2 HP"},          # points or bars: review
            {"id": "2", "timestamp": "2024-01-01T00:00:00+00:00", "author": {"bot": False},
             "content": "Iron Golem: ATK 5 -> 6"}]                # after the card repo: left out
    (root / "messages" / f"{update}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    notice = {"id": "3", "timestamp": "2022-09-01T00:00:00+00:00", "author": {"bot": True}, "webhook_id": "9",
              "embeds": [{"title": "[Cards:main] 2 new commits", "description":
                          "[`abcdef1`](u) Quirky Sun Cookie fix - someone\n[`abcdef2`](u) readme - someone"}]}
    listed = {**notice, "id": "6", "embeds": [{"title": "[Cards:main] 1 new commit", "description":
                                              "[`abcdef3`](u) Sun Cookie and Krabs update - someone"}]}
    (root / "messages" / "302.jsonl").write_text(json.dumps(notice) + "\n" + json.dumps(listed) + "\n")
    F.build(str(root), names(), TYPES, log=lambda *a: None)
    out = root / "derived" / "facts"
    got = [json.loads(line) for line in (out / "facts.jsonl").read_text().splitlines()]
    assert [(f["card"], f.get("kind") or f["field"], f["line"], f.get("also"), f.get("repeat_of"))
            for f in got if f["source"] == "changelog"] == [
        ("Bolt", "stars", "5:0", None, None), ("Bolt", "hp", "5:1", None, None),
        ("Iron Golem", "ability changed", "1:0", ["1:2"], None),
        ("Iron Golem", "atk", "1:1", ["4:0"], None), ("Bolt", "stars", "4:1", None, got[0]["id"])]
    assert [(f["cards"], f["author"]) for f in got if f["source"] == "push"] == [(["Sun Cookie"], "Somebody")] * 2
    hp = next(f for f in got if f.get("field") == "hp")
    assert (hp["unit"], hp["confidence"]) == ("unknown", "unsure")
    assert [r["why"] for r in map(json.loads, (out / "review.jsonl").read_text().splitlines())
            if r.get("fact") == hp["id"]] == [F.DOUBT["unit"]]
    for name in ("facts.jsonl", "review.jsonl"):
        assert "zany" not in (out / name).read_text() and "Quirky" not in (out / name).read_text()
    row = json.loads((out / "firstrepo.jsonl").read_text())       # one notice: the listing one is evidence only
    assert (row["card"], row["author"], row["commit"]) == ("Sun Cookie", "Somebody", "abcdef1")   # a Created hint
    assert not (out / "switches.json").exists()                   # owner Q5 (11-14): no text-only rows


def test_hp_def_units():
    day = lambda d: F.stamp(f"2020-{d}T00:00:00+00:00")
    fact = lambda card, field, d, **vals: {"id": card + field + d, "card": card, "field": field,
                                           "date": f"2020-{d}T00:00:00+00:00", "source": "changelog",
                                           "confidence": "sure", **vals}
    reads = {"Iron Golem": [(day("08-10"), {"hp": 8, "def": 2})], "Mega Bot": [(day("06-01"), {"def": 5}),
                                                                               (day("08-10"), {"def": 4})],
             "Gloom": [(day("07-10"), {"hp": 3})], "Sun Cookie": [(day("07-01"), {"def": 1})],
             "Twin Pike": [(F.stamp("2021-06-01T00:00:00+00:00"), {"def": 4})]}
    facts = [fact("Iron Golem", "hp", "08-01", to=16, said="bars"),   # the picture after shows 8: points
             fact("Iron Golem", "def", "08-01", to=2),                # = the picture
             fact("Mega Bot", "def", "08-01", by=-1),                 # 5 -> 4
             fact("Bolt", "hp", "07-01", **{"from": 14, "to": 12}),   # more than a row of bars: points
             fact("Bolt", "hp", "07-02", by=-1),                      # half a bar in points: bars
             fact("Sun Cookie", "def", "07-05", by=-1),               # from 1: -2 bars can't be
             fact("Pike", "def", "07-01", to=3, said="list"),         # a 4-number stat list
             fact("Gloom", "hp", "07-01", to=8),                      # the read is past the next change
             fact("Gloom", "hp", "07-05", by=-2),
             fact("Twin Pike", "def", "07-01", to=2),              # the only read is 11 months later
             fact("Shard", "atk", "07-01", by=-2)]
    F.stat_units(facts, reads, set(), Counter())
    assert [(f["card"], f["unit"], f.get("unit_how"), f.get("from"), f.get("to"), f.get("by"), f["confidence"])
            for f in facts] == [
        ("Iron Golem", "points", "read", None, 8, None, "sure"), ("Iron Golem", "bars", "read", None, 2, None, "sure"),
        ("Mega Bot", "bars", "read", None, None, -1, "sure"), ("Bolt", "points", "range", 7, 6, None, "sure"),
        ("Bolt", "bars", "range", None, None, -1, "sure"), ("Sun Cookie", "bars", "range", None, None, -1, "sure"),
        ("Pike", "bars", "list", None, 3, None, "sure"), ("Gloom", "unknown", None, None, 8, None, "unsure"),
        ("Gloom", "unknown", None, None, None, -2, "unsure"), ("Twin Pike", "unknown", None, None, 2, None, "unsure"),
        ("Shard", "bars", None, None, None, -2, "sure")]
    assert facts[0]["stated"] == {"to": 16} and facts[7]["_doubt"] == F.DOUBT["unit"]
    assert not any("said" in f for f in facts)


def test_reads_join_the_matcher_and_old_rows(tmp_path):
    stats = {"A-B": {}, "A b": {}, "Dup": {"duplicate_of": "A b"}}
    ids = F.card_ids(stats)
    assert ids == {"a-b": "A b", "a-b-2": "A-B", "dup": "Dup"}
    for parts, rows in (
            (F.VERDICTS, [{"file": "x.png", "verdict": "version_sure", "card": "a-b-2",
                           "first": {"date": "2020-01-02T00:00:00+00:00"}},
                          {"file": "y.png", "verdict": "version_sure", "card": "dup", "joke": True,
                           "first": {"date": "2020-01-02T00:00:00+00:00"}}]),
            (F.BAR_READS, [{"src": "discord", "file": "x.png", "stats": {"hp": {"v": 4, "sure": True},
                                                                        "def": {"v": 9, "sure": False}}},
                           {"src": "discord", "file": "y.png", "stats": {"hp": {"v": 2, "sure": True}}}]),
            (F.OLD_ROWS, [{"card": "A-B", "date": "2023-04-21T01:00:00-07:00",
                           "fields": {"stats": {"value": {"hp": 3, "def": 2, "atk": 1}, "confidence": "sure"}}},
                          {"card": "Dup", "date": "2023-04-21T01:00:00-07:00",
                           "fields": {"stats": {"value": {"hp": 1}, "confidence": "review"}}}])):
        path = tmp_path.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    reads = F.load_reads(str(tmp_path), ids)
    assert list(reads) == ["A-B"] and [r for _, r in reads["A-B"]] == [{"hp": 4}, {"hp": 3, "def": 2}]


def test_internal_old_names_and_their_time_limit():
    base = names()
    inner = F.Names({"Iron Golem": ["Iron Golem"], "Gloom": ["Gloom"], "Stand Still": ["Stand Still"]}, {}, [],
                    with_stats={"Iron Golem"}, internal=[("Rust Golem", "Iron Golem", None), ("Gloom", "Iron Golem",
                                                         None), ("Gloom", "Stand Still", "2020-08-02T01:49:19+00:00")])
    assert inner.by["rustgolem"] == {("Iron Golem", "internal")}
    assert inner.by["gloom"] == {("Gloom", "name")}                  # a name a card has today stays that card's
    assert inner.at("2020-08-01T09:50:40+00:00").by["gloom"] == {("Stand Still", "internal")}
    assert inner.at("2020-08-02T02:12:57+00:00") is inner            # after the limit: today's card
    got = F.parse_line("Rust Golem: +1 ATK", inner, TYPES, "changelog")["facts"]
    assert [(f["card"], f["field"], f["by"]) for f in got] == [("Iron Golem", "atk", 1)]
    assert base.by.get("rustgolem") is None
    assert facts("Mega Bot: Changed name to Mega Bot") == [{"card": "Mega Bot", "field": "name", "to": "Mega Bot",
                                                            "sure": False}]   # the prefix is no old name


def test_changelog_picture_entries():
    line = lambda text, y, h=0.02: [text, 1.0, 0.05, y, 0.8, h]
    lines = [line("Some Changes", 0.0, 0.07), line("Iron Golem (Rework) ATK -1 DEF -1 No longer", 0.08),
             line("explodes on death", 0.105), line("Sun Cookie (Buff) Heals more", 0.15), line("Items", 0.22),
             line("Old Relic (Nerf/Rework) Is now a 2 star item Cannot target", 0.26)]
    assert F.picture_entries(lines) == [("Iron Golem", "Rework", "ATK -1 DEF -1 No longer explodes on death"),
                                        ("Sun Cookie", "Buff", "Heals more"),
                                        ("Old Relic", "Nerf/Rework", "Is now a 2 star item Cannot target")]
    n = names()
    got = F.picture_line("Iron Golem", "ATK -1 DEF -1 No longer explodes on death", n, n, TYPES)
    assert [(f["card"], f.get("field") or f.get("kind"), f.get("by"), f["sure"]) for f in got["facts"]] == [
        ("Iron Golem", "atk", -1, True), ("Iron Golem", "def", -1, True), ("Iron Golem", "ability changed", None, True)]
    got = F.picture_line("Sun Cookie", "Heals more", n, n, TYPES)    # the tag says it changed: the ability
    assert [(f["card"], f["kind"], f["sure"]) for f in got["facts"]] == [("Sun Cookie", "ability changed", True)]
    got = F.picture_line("Sun Cookie", "Back to its old effect, Is now a 3 star item", n, n, TYPES)
    assert [f.get("field") or f["kind"] for f in got["facts"]] == ["stars", "ability changed"]   # more than stated
    got = F.picture_line("Iron Golem", "Is now a 2 star creature, ATK -1", n, n, TYPES)
    assert [f.get("field") or f["kind"] for f in got["facts"]] == ["stars", "atk"]
    got = F.picture_line("Golems", "Damage 2 -> 3", n, n, TYPES)   # no card has this name: review
    assert got["facts"] == [] and "Golems" in got["head"]


def test_eye_told_short_forms_and_the_february_document(tmp_path):
    root = tmp_path
    (root / "messages").mkdir()
    (root / "derived" / "facts").mkdir(parents=True)
    msg = {"id": "900", "timestamp": "2020-08-01T10:00:00+00:00", "channel_name": "update-02",
           "content": "golem: +1 ATK\ngolem: +1 DEF", "author": {"id": "1"}}
    channel = "303"
    (root / "channels.json").write_text(json.dumps([{"id": channel, "name": "update-02"}]))
    (root / "messages" / f"{channel}.jsonl").write_text(json.dumps(msg) + "\n")
    (root / "derived" / "facts" / "eye_lines.json").write_text(json.dumps(
        {"lines": {"900:0": {"card": "Iron Golem", "form": "golem"}}}))
    got = [(g["facts"], g.get("eye")) for *_, g in F.changelog_lines(str(root), names(), TYPES, chat=False)]
    assert [(f["card"], f["field"], f["by"]) for f in got[0][0]] == [("Iron Golem", "atk", 1)] and got[0][1]
    assert got[1][0] == [] and not got[1][1]                          # only the told line, never a loose match
    import zipfile                                                   # a docx changelog: its heads, no text kept
    doc = root / "attachments" / "6655b62e4972ff.docx"
    doc.parent.mkdir()
    with zipfile.ZipFile(doc, "w") as z:
        z.writestr("word/document.xml", "<w:p><w:r><w:t>Some changes</w:t></w:r></w:p>"
                                        "<w:p><w:r><w:t>Iron Golem (Nerf) ATK 5 -&gt; 4</w:t></w:r></w:p>")
    (root / "attachments.jsonl").write_text(json.dumps({"sha256": "6655b62e4972ff", "file": "attachments/" + doc.name,
                                                        "message_id": "901", "container_id": channel}) + "\n")
    with (root / "messages" / f"{channel}.jsonl").open("a") as f:
        f.write(json.dumps({**msg, "id": "901", "content": ""}) + "\n")
    lines = list(F.document_lines(str(root), names(), TYPES, Counter()))
    assert [(n, [(f["card"], f["field"], f["to"]) for f in g["facts"]]) for *_, n, _t, g in lines] == [
        ("doc0", [("Iron Golem", "atk", 4)])]
    assert F.line_order("901:doc3") == 3


def test_a_line_given_its_card(tmp_path):
    """Owner D9: a line naming no sure card takes the card given (owner.json lines over the review page's answers;
    a Morgue entry's 'morgue:<id>', never the live card it resembles; 'none' leaves it out)."""
    n = names()
    got = F.picture_line("Golems", "Damage 2 -> 3", n, n, TYPES, "Iron Golem")
    assert got["facts"] and {f["card"] for f in got["facts"]} == {"Iron Golem"} and not got["unsure"]
    got = F.picture_line("Mega Bot 2.0", "Is now a 4 star creature", n, n, TYPES, "morgue:LC-1")
    assert [(f["card"], f.get("field") or f["kind"]) for f in got["facts"]] == [("morgue:LC-1", "stars")]
    assert {f["card"] for f in F.parse_line("Pike: ATK +1", n, TYPES, "changelog", told="Twin Pike")["facts"]} == {
        "Twin Pike"}                                                   # a name two cards had: the one given
    (tmp_path / "derived" / "review").mkdir(parents=True)
    n.ids = {"iron-golem": "Iron Golem", "pike": "Pike"}
    (tmp_path / "derived" / "review" / "rulings.json").write_text(json.dumps({"rulings": {
        "R1": {"kind": "ambiguous card", "choice": "card", "value": "pike", "item": {"line": "9:img1"}},
        "R2": {"kind": "ambiguous card", "choice": "none", "item": {"line": "9:img2"}},
        "R3": {"kind": "ambiguous card", "choice": "card", "value": "pike", "item": {"line": "9:img3"}}}}))
    (tmp_path / "derived" / "review" / "owner.json").write_text(json.dumps({"lines": {
        "R3": {"card": "iron-golem", "line": "9:img3"}, "R4": {"morgue": "LC-1", "line": "9:doc0"}}}))
    assert F.told_lines(str(tmp_path), n) == {"9:img1": "Pike", "9:img2": None, "9:img3": "Iron Golem",
                                              "9:doc0": "morgue:LC-1"}
    first = F.picture_line("Golems", "ATK +1", n, n, TYPES)
    again = lambda card: F.picture_line("Golems", "ATK +1", n, n, TYPES, card)
    given = F.as_given(first, "9:img1", {"9:img1": "Pike"}, again)
    assert {f["card"] for f in given["facts"]} == {"Pike"} and given["given"]["card"] == "Pike"
    assert F.as_given(first, "9:img2", {"9:img2": None}, again)["facts"] == []
    assert F.as_given(first, "9:img5", {}, again) is first                 # not given: as read
