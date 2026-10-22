"""changelog: per-version field changes and the buff/nerf verdict, every event kind, seq numbering, the Discord message
within Discord's limits, and the posted cursor that moves only after a post went through."""
import json
from collections import defaultdict
from types import SimpleNamespace

import pytest

import commands.changelog as cl
import commands.card_models as cm

T = 1_792_700_000.0                                                       # 2026-10-22, Pacific


def _card(name="A", **kw):
    base = dict(name=name, card_type="creature", path=f"Creatures/X/3 Stars/{name}.psd", stars=3, ability="(1CD) Hit.",
                stats=cm.CardStats(5, 2, 3, 4), blob_sha="b1", author="Bob")
    base.update(kw)
    return cm.CardInfo(**base)


# ========================
# Field changes and the verdict
# ========================
def test_tags_and_fields():
    assert cl.tags_of("(1GCD)(2U)(OYT) Heal. (1.5 points) (E2T) (1/2CD) (FTE2T) (2PGT) (X)") == \
        ["(1GCD)", "(2U)", "(OYT)", "(E2T)", "(1/2CD)", "(FTE2T)", "(2PGT)"]
    fields = cl.fields_of(_card(display_name="Shown", stats=cm.CardStats(5, -1, 3, 4), creatures=[cm.CardStats(1, 2, 3, -1)]))
    assert fields["name"] == "Shown" and (fields["hp"], fields["def"]) == (5, None) and fields["creatures"] == [[1, 2, 3, None]]
    assert fields["tags"] == ["(1CD)"] and fields["hidden"] is False


@pytest.mark.parametrize("old, new, verdict, buffs, nerfs", [
    (dict(stats=cm.CardStats(5, 2, 3, 4)), dict(stats=cm.CardStats(6, 2, 3, 4)), "buff", ["HP 5 -> 6"], []),
    (dict(stats=cm.CardStats(5, 2, 3, 4)), dict(stats=cm.CardStats(5, 1, 3, 4)), "nerf", [], ["DEF 2 -> 1"]),
    (dict(stats=cm.CardStats(5, 2, 3, 4)), dict(stats=cm.CardStats(6, 2, 2, 4)), "mixed", ["HP 5 -> 6"], ["ATK 3 -> 2"]),
    (dict(ability="(2CD) Hit."), dict(ability="(1CD) Hit."), "buff", ["(2CD) -> (1CD)"], []),
    (dict(ability="(1GCD)(3U) Heal."), dict(ability="(2GCD)(2U) Heal."), "nerf", [], ["(1GCD) -> (2GCD)", "(3U) -> (2U)"]),
    (dict(ability="(1PT) Hit."), dict(ability="(1CD) Hit."), None, [], []),          # a renamed tag is not judged
    (dict(ability="(1CD) Hit."), dict(ability="(1CD)(2U) Hit."), None, [], []),     # an added tag is not judged
    (dict(stars=3), dict(stars=4), None, [], []),                                  # stars are listed, not judged
    (dict(stats=cm.CardStats(-1, 2, 3, 4)), dict(stats=cm.CardStats(5, 2, 3, 4)), None, [], []),   # a stat found
    (dict(creatures=[cm.CardStats(1, 1, 1, 1), cm.CardStats(4, 4, 4, 4)]),
     dict(creatures=[cm.CardStats(1, 1, 1, 1), cm.CardStats(4, 10, 4, 4)]), "buff", ["creature 2 DEF 4 -> 10"], []),
])
def test_classify(old, new, verdict, buffs, nerfs):
    changes = cl.field_changes(_card(**old), _card(**new))
    assert cl.classify(changes) == {"verdict": verdict, "buffs": buffs, "nerfs": nerfs}


def test_field_changes_list_the_data_fields():
    changes = cl.field_changes(_card(ability="(2CD) Hit.", types=["fire"], display_name=None),
                               _card(ability="(1CD) Smash.", types=["ice"], display_name="Big A", stars=4))
    assert list(changes) == ["name", "stars", "types", "ability", "tags"]
    assert changes["name"] == ["A", "Big A"] and changes["tags"] == [["(2CD)"], ["(1CD)"]]
    assert cl.field_changes(_card(), _card(path="Elsewhere/A.psd", blob_sha="b2", timestamp=9.0)) == {}


def test_version_changes():
    v0 = _card(stats=cm.CardStats(5, 2, 3, 4))
    v1 = _card(stats=cm.CardStats(6, 2, 3, 4))
    v1.version = {"from": 100.0, "cause": "psd"}
    current = _card(stats=cm.CardStats(6, 2, 3, 4), ability="(2CD) Hit.")
    current.version = {"from": 200.0, "cause": "edit"}
    steps = cl.version_changes([v0, v1], current)
    assert [(s["from"], s["cause"], list(s["changes"]), s["verdict"]) for s in steps] == \
        [(100.0, "psd", ["hp"], "buff"), (200.0, "edit", ["ability", "tags"], "nerf")]


# ========================
# Events
# ========================
def _index():
    commit = SimpleNamespace(sha="c9", author="maker", at=T)
    return SimpleNamespace(blob_intro=lambda blob: commit if blob else None, newest={"Items/Moved2.psd": commit},
                           by_sha={"cd": SimpleNamespace(sha="cd", author="remover", at=T)})


def test_every_kind():
    before = {"Changed": _card("Changed"), "Art": _card("Art"), "Reparse": _card("Reparse"), "Moved": _card("Moved"),
              "Moved2": _card("Moved2", path="Items/1 Stars/Moved2.psd", stars=1), "Edited": _card("Edited"),
              "Authors": _card("Authors"), "Gone": _card("Gone"), "Old": _card("Old"), "Same": _card("Same"),
              "Dup2": _card("Dup2", duplicate_of="Same")}
    after = {"Changed": _card("Changed", blob_sha="b2", stats=cm.CardStats(5, 2, 3, 9)), "Art": _card("Art", blob_sha="b2"),
             "Reparse": _card("Reparse", ability="(1CD) Hit hard."), "Moved": _card("Moved", path="Items/Moved.psd"),
             "Moved2": _card("Moved2", path="Items/Moved2.psd", stars=2), "Edited": _card("Edited", note="checked"),
             "Authors": _card("Authors", author="Alice"), "New": _card("New"), "Back": _card("Back"), "Same": _card("Same"),
             "Dup2": _card("Dup2", duplicate_of="Same", display_name="Other")}
    archived_before = {"Back": _card("Back")}
    archived_after = {"Gone": _card("Gone", archived={"commit": "cd"}),
                      "Old": _card("Old", archived={"commit": "cd", "renamed_to": "New"})}
    events = cl.events(before, after, archived_before, archived_after, {"Changed": 1, "Art": 1, "Reparse": 1, "Moved2": 1},
                       {"Edited": "web-user"}, _index())
    kinds = {e["card"]: e["kind"] for e in events}
    assert kinds == {"Changed": "changed", "Art": "art", "Reparse": "reparse", "Moved": "moved", "Moved2": "moved",
                     "Edited": "edited", "Authors": "authors", "Gone": "archived", "Old": "renamed", "New": "new",
                     "Back": "restored"}                                          # Same and its duplicate: nothing
    by = {e["card"]: e for e in events}
    assert by["Changed"]["fields"] == {"spd": [4, 9]} and by["Changed"]["verdict"] == "buff" and by["Changed"]["commit"] == "c9"
    assert by["Changed"]["blob"] == ["b1", "b2"] and by["Changed"]["by"] == "maker"
    assert by["Edited"]["edit_by"] == "web-user" and "edit_by" not in by["Authors"]
    assert by["Moved2"]["fields"] == {"stars": [1, 2]} and by["Moved2"]["from_path"] == "Items/1 Stars/Moved2.psd"
    assert by["Gone"]["commit"] == "cd" and by["Gone"]["by"] == "remover" and by["Old"]["renamed_to"] == "New"
    assert by["New"]["by"] == "Bob" and set(cl.KINDS) >= set(kinds.values())


def test_seq_never_reused():
    first = cl.number([{"kind": "new", "card": "A"}, {"kind": "new", "card": "B"}], last_seq=0, posted_seq=0, at=T, build="1")
    assert [e["seq"] for e in first] == [1, 2] and first[0]["build"] == "1" and first[0]["at"] == T
    after_restore = cl.number([{"kind": "restore", "card": ""}], last_seq=1, posted_seq=2, at=T)   # a restore dropped seq 2
    assert after_restore[0]["seq"] == 3


def test_month_files(tmp_path):
    db = str(tmp_path)
    events = cl.number([{"kind": "new", "card": "A"}], 0, at=T) + cl.number([{"kind": "new", "card": "B"}], 1, at=T + 20 * 86400)   # Nov 11
    cl.append_events(db, events[:1])
    cl.append_events(db, events[1:])
    assert sorted(p.name for p in (tmp_path / "changelog").iterdir()) == ["2026-10.jsonl", "2026-11.jsonl"]
    assert [e["card"] for e in cl.read_events(db)] == ["A", "B"]


# ========================
# The daily post
# ========================
def test_payload_words_links_and_order():
    events = cl.number([{"kind": "new", "card": "Kirby", "title": "Kirby", "card_type": "creature", "stars": 3, "by": "Bob"},
                        {"kind": "changed", "card": "Waddle Dee", "fields": {"hp": [5, 6], "ability": ["a", "b"], "tags": [[], []]},
                         "verdict": "buff"},
                        {"kind": "archived", "card": "Gone"}, {"kind": "renamed", "card": "ZOEY!!!!!", "renamed_to": "Z!!!!!"},
                        {"kind": "edited", "card": "Meta", "fields": {"subseries": [None, "Kirby"]}, "edit_by": "discord:mike"},
                        {"kind": "reparse", "card": "R"}, {"kind": "reparse", "card": "S"}], 0, at=T)
    body = cl.payload(events, "https://site", {"Kirby": "kirby"})
    assert body["allowed_mentions"] == {"parse": []} and "content" not in body
    titles = [e.get("title") for e in body["embeds"]]
    assert titles == ["Card changes, Oct 22: New cards", "Changed", "Moved and renamed", "Archived", "Database edits", None]
    text = "\n".join(e["description"] for e in body["embeds"])
    assert "[Kirby](https://site/c/kirby/) (Creature, 3★) by Bob" in text
    assert "[Waddle Dee](https://site/#/cards?q=Waddle%20Dee): HP 5 → 6; ability changed (buff)" in text
    assert "ZOEY!!!!! is now [Z!!!!!](" in text and "Gone (PSD deleted)" in text
    assert "[Meta](https://site/#/cards?q=Meta): subseries → Kirby (by mike, via Discord)" in text
    assert body["embeds"][len(body["embeds"]) - 1]["description"] == "The parser re-read 2 cards."
    assert cl.payload([], "https://site") is None


def test_payload_stays_within_discord_limits():
    events = cl.number([{"kind": k, "card": f"{k} card {i} " + "x" * 60, "card_type": "item", "fields": {"ability": ["a", "b"]}}
                        for k in ("new", "changed", "art", "moved", "archived", "restored", "edited") for i in range(150)], 0, at=T)
    body = cl.payload(events, "https://michaeljsr.github.io/TTSCardMaker")
    assert len(body["embeds"]) <= cl.EMBEDS_MAX and cl.size(body) <= cl.MESSAGE_MAX
    assert all(len(e["description"]) <= cl.DESCRIPTION_MAX for e in body["embeds"])
    last = body["embeds"][len(body["embeds"]) - 1]["description"]
    shown = sum(e["description"].count("\n") + 1 for e in body["embeds"][:len(body["embeds"]) - 1])
    assert last == f"{len(events) - shown} more on the website: https://michaeljsr.github.io/TTSCardMaker/#/changes/changelog"


def test_titles_across_days_and_months():
    day = 86400
    assert cl.title_of([{"at": T}]) == "Card changes, Oct 22"
    assert cl.title_of([{"at": T - day}, {"at": T}]) == "Card changes, Oct 21-22"
    assert cl.title_of([{"at": T - 22 * day}, {"at": T}]) == "Card changes, Sep 30-Oct 22"


def test_cursor_moves_only_after_the_post(tmp_path):
    db = str(tmp_path / "db")
    cl.append_events(db, cl.number([{"kind": "new", "card": "A"}, {"kind": "new", "card": "B"}], 0, at=T))
    out = tmp_path / "discord.json"
    assert cl.main(["post", "--db", db, "--out", str(out)]) == 0
    sent = json.loads(out.read_text(encoding="utf-8"))
    assert sent["cursor"]["seq"] == 2 and sent["message"]["embeds"] and cl.read_posted(db) == {"seq": 0, "date": ""}
    assert cl.main(["post", "--db", db, "--out", str(out)]) == 0                      # not advanced: the same post again
    assert json.loads(out.read_text(encoding="utf-8"))["cursor"]["seq"] == 2
    assert cl.main(["advance", "--db", db, "--seq", "2", "--date", "2026-10-22"]) == 0
    assert cl.read_posted(db) == {"seq": 2, "date": "2026-10-22"}
    assert cl.main(["post", "--db", db, "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["message"] is None              # nothing new
    cl.advance(db, 1, "2026-10-21")
    assert cl.read_posted(db) == {"seq": 2, "date": "2026-10-22"}                      # never backwards


def test_history_cli(tmp_path, capsys):
    import pickle
    v0 = _card(stats=cm.CardStats(5, 2, 3, 4))
    current = _card(stats=cm.CardStats(5, 2, 3, 2))
    current.version = {"from": T, "cause": "psd"}
    (tmp_path / "stats.pkl").write_bytes(pickle.dumps({"A": current}))
    (tmp_path / "old_stats.pkl").write_bytes(pickle.dumps(defaultdict(list, {"A": [v0]})))
    assert cl.main(["history", "--pickles", str(tmp_path), "A"]) == 0
    assert capsys.readouterr().out.strip() == "2026-10-22 (psd): spd [nerf: SPD 4 -> 2]"
    assert cl.main(["history", "--pickles", str(tmp_path), "Nobody"]) == 1
