"""changelog: per-version field changes and the buff/nerf verdict, every event kind, seq numbering, the Discord message
grouped by push (the activity list and the fallback), one message per push within Discord's limits and at most 20 a
run, and the posted cursor that moves only after each accepted message."""
import json
from types import SimpleNamespace


import commands.changelog as cl
import commands.card_models as cm

T = 1_792_700_000.0                                                       # 2026-10-22, Pacific


def _card(name="A", **kw):
    base = dict(name=name, card_type="creature", path=f"Creatures/X/3 Stars/{name}.psd", stars=3, ability="(1CD) Hit.",
                stats=cm.CardStats(5, 2, 3, 4), blob_sha="b1", author="Bob", parser_version=2)
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


CLASSIFY_CASES = [
    (dict(stats=cm.CardStats(5, 2, 3, 4)), dict(stats=cm.CardStats(6, 2, 3, 4)), "buff", ["HP 5 -> 6"], []),
    (dict(stats=cm.CardStats(5, 2, 3, 4)), dict(stats=cm.CardStats(5, 1, 3, 4)), "nerf", [], ["DEF 2 -> 1"]),
    (dict(stats=cm.CardStats(5, 2, 3, 4)), dict(stats=cm.CardStats(6, 2, 2, 4)), "mixed", ["HP 5 -> 6"], ["ATK 3 -> 2"]),
    (dict(ability="(1GCD)(3U) Heal."), dict(ability="(2GCD)(2U) Heal."), "nerf", [], ["(1GCD) -> (2GCD)", "(3U) -> (2U)"]),
    (dict(ability="(1PT) Hit."), dict(ability="(1CD) Hit."), None, [], []),
]


def test_classify():
    for old, new, verdict, buffs, nerfs in CLASSIFY_CASES:
        changes = cl.field_changes(_card(**old), _card(**new))
        assert cl.classify(changes) == {"verdict": verdict, "buffs": buffs, "nerfs": nerfs}


def test_field_changes_list_the_data_fields():
    changes = cl.field_changes(_card(ability="(2CD) Hit.", types=["fire"], display_name=None),
                               _card(ability="(1CD) Smash.", types=["ice"], display_name="Big A", stars=4))
    assert list(changes) == ["name", "stars", "types", "ability", "tags"]
    assert changes["name"] == ["A", "Big A"] and changes["tags"] == [["(2CD)"], ["(1CD)"]]
    assert cl.field_changes(_card(), _card(path="Elsewhere/A.psd", blob_sha="b2", timestamp=9.0)) == {}


def test_field_changes_skip_what_a_reading_could_not_know():
    """No phantom steps: a name before parser 2 read titles (Mol D Truffy), a series the 2026-10-20 split moved under
    a parent (Bob-ulk), stars a parse could not find (Bubble), an empty ability's two forms, a retired icon's name."""
    early = _card(parser_version=None, display_name=None)
    assert cl.field_changes(early, _card(display_name="Mol D. A")) == {}
    assert list(cl.field_changes(early, _card(display_name="Mol D. A", stars=4))) == ["stars"]
    mario = _card(series="Mario", subseries="Paper Mario")
    assert cl.field_changes(_card(series="Paper Mario"), mario) == {}
    assert list(cl.field_changes(_card(series="Zelda"), mario)) == ["series", "subseries"]
    assert cl.field_changes(_card(stars=0, problems=['NO STARS FOUND (put the star layers in a group named "Stars")']),
                            _card(stars=1)) == {}
    assert cl.field_changes(_card(ability=None), _card(ability="")) == {}
    assert cl.field_changes(_card(types=["offense"]), _card(types=["negation"])) == {}
    assert cl.field_changes(_card(display_name="A"), _card()) == {}                 # a title equal to the name


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
             "Authors": _card("Authors", author="Alice"), "New": _card("New"), "Fresh": _card("Fresh"), "Back": _card("Back"),
             "Same": _card("Same"),
             "Dup2": _card("Dup2", duplicate_of="Same", display_name="Other")}
    archived_before = {"Back": _card("Back")}
    archived_after = {"Gone": _card("Gone", archived={"commit": "cd"}),
                      "Old": _card("Old", archived={"commit": "cd", "renamed_to": "New"})}
    events = cl.events(before, after, archived_before, archived_after, {"Changed": 1, "Art": 1, "Reparse": 1, "Moved2": 1},
                       {"Edited": "web-user"}, _index())
    kinds = {e["card"]: e["kind"] for e in events}
    assert kinds == {"Changed": "changed", "Art": "art", "Reparse": "reparse", "Moved": "moved", "Moved2": "moved",
                     "Edited": "edited", "Authors": "authors", "Gone": "archived", "Old": "renamed", "Fresh": "new",
                     "Back": "restored"}                     # Same and its duplicate: nothing; New: the rename's line only
    by = {e["card"]: e for e in events}
    assert by["Changed"]["fields"] == {"spd": [4, 9]} and by["Changed"]["verdict"] == "buff" and by["Changed"]["commit"] == "c9"
    assert by["Changed"]["blob"] == ["b1", "b2"] and by["Changed"]["by"] == "maker"
    assert by["Edited"]["edit_by"] == "web-user" and "edit_by" not in by["Authors"]
    assert by["Moved2"]["fields"] == {"stars": [1, 2]} and by["Moved2"]["from_path"] == "Items/1 Stars/Moved2.psd"
    assert by["Gone"]["commit"] == "cd" and by["Gone"]["by"] == "remover" and by["Old"]["renamed_to"] == "New"
    assert by["Fresh"]["by"] == "Bob" and set(cl.KINDS) >= set(kinds.values())


def test_a_reread_cards_edits_are_their_own_event():
    """A re-read card's fields go to the PSD's event only when its parse changed (parsed_values put back) and no edit
    sets them now; an edit (new, changed or removed) and the author are the edit's or the authors file's."""
    before = {"Art": _card("Art"), "Both": _card("Both"), "Unedit": _card("Unedit", ability="Edited.",
                                                                           parsed_values={"ability": "(1CD) Hit."}),
              "Back": _card("Back", ability="Edited.", parsed_values={"ability": "(1CD) Hit."})}
    after = {"Art": _card("Art", blob_sha="b2", display_name="Shown", parsed_values={"name": None, "problems": []}),
             "Both": _card("Both", blob_sha="b2", stats=cm.CardStats(6, 2, 3, 4), display_name="Shown", author="Al",
                           parsed_values={"name": None, "problems": []}),
             "Unedit": _card("Unedit", blob_sha="b2", ability="New."),           # edit gone, the PSD changed too
             "Back": _card("Back", blob_sha="b2")}                              # edit gone, the PSD as before
    found = sorted((e["card"], e["kind"], sorted(e.get("fields") or {})) for e in
                   cl.events(before, after, {}, {}, dict.fromkeys(after, 1), {n: "web" for n in after}))
    assert found == [("Art", "art", []), ("Art", "edited", ["name"]), ("Back", "art", []), ("Back", "edited", ["ability", "tags"]),
                     ("Both", "changed", ["hp"]), ("Both", "edited", ["author", "name"]), ("Unedit", "changed", ["ability"])]


def test_bot_edits_never_read_as_website_edits():
    """A _by AutoCroissant entry, and fields the migration listed in an editor's _bot, are the bot's: "(by AutoCroissant)",
    never "on the website" (changes.js byWhom)."""
    before = {n: _card(n) for n in ("Bot", "Moved", "Mixed", "Web")}
    after = {"Bot": _card("Bot", note="x"), "Moved": _card("Moved", series="Kirby"),
             "Mixed": _card("Mixed", series="Kirby", note="y"), "Web": _card("Web", card_type="item")}
    meta = {"Bot": {"_by": "autocroissant"}, "Moved": {"_by": "web", "_bot": ["series"]},
            "Mixed": {"_by": "web", "_bot": ["series"]}, "Web": {"_by": "web", "_bot": ["series"]}}
    by = {e["card"]: e["edit_by"] for e in cl.events(before, after, {}, {}, edits_by=meta)}
    assert by == {"Bot": "autocroissant", "Moved": "AutoCroissant", "Mixed": "web", "Web": "web"}
    lines = [cl._line({"kind": "edited", "card": "A", "fields": {"note": [None, "x"]}, "edit_by": b}, "https://site", None)
             for b in ("AutoCroissant", "web")]
    assert lines[0].endswith(": note \u2192 x (by AutoCroissant)") and lines[1].endswith("(by web, on the website)")


def test_seq_never_reused():
    first = cl.number([{"kind": "new", "card": "A"}, {"kind": "new", "card": "B"}], last_seq=0, posted_seq=0, at=T, build="1")
    assert [e["seq"] for e in first] == [1, 2] and first[0]["build"] == "1" and first[0]["at"] == T
    after_restore = cl.number([{"kind": "restore", "card": ""}], last_seq=1, posted_seq=2, at=T)   # a restore dropped seq 2
    assert after_restore[0]["seq"] == 3


# ========================
# Pushes and the posts (one per push)
# ========================
PUSH = {"after": "a2" * 20, "before": "a1" * 20, "by": "Zomoroko", "at": T + 3600}


def test_pushes_from_the_activity_list_and_the_fallback():
    """The activity list's push holding the event's commit; commits it doesn't list: consecutive ones by one author
    are one push (before = the commit before them); a re-read has none. Events come back in push order."""
    events = [{"kind": "changed", "card": "A", "commit": "c3"}, {"kind": "new", "card": "B", "commit": "c1"},
              {"kind": "edited", "card": "C", "commit": "c5"}, {"kind": "new", "card": "D", "commit": "c4"},
              {"kind": "reparse", "card": "E", "commit": "c1"}, {"kind": "art", "card": "F", "commit": "old"}]
    listed = [{"after": "c2", "before": "c0", "by": "zomo-gh", "at": 50.0, "commits": ["c2", "c1"]}]
    commits = [("c1", 10.0, "Zomoroko"), ("c2", 20.0, "Zomoroko"), ("c3", 30.0, "Bob"), ("c4", 40.0, "Bob"),
               ("c5", 60.0, "Carol")]
    out = cl.assign_pushes(events, listed, commits, base="c0")
    assert [e["card"] for e in out] == ["B", "A", "D", "C", "F", "E"]
    by = {e["card"]: e.get("push") for e in out}
    assert by["B"] == {"after": "c2", "before": "c0", "by": "zomo-gh", "at": 50.0}
    assert by["A"] == by["D"] == {"after": "c4", "before": "c2", "by": "Bob", "at": 40.0}
    assert by["C"] == {"after": "c5", "before": "c4", "by": "Carol", "at": 60.0}
    assert by["E"] is None and by["F"] is None                                    # a re-read; a commit of no push
    fallback = cl.assign_pushes([dict(e) for e in events], [], commits, base="c0")
    assert {e["card"]: (e.get("push") or {}).get("after") for e in fallback} == \
        {"B": "c2", "A": "c4", "D": "c4", "C": "c5", "F": None, "E": None}


def test_the_bake_commit_is_a_push_of_its_own():
    """Without the activity list, the run's bake commit B never joins the push before it, even by the same author."""
    commits = [("c1", 10.0, "AutoCroissant"), ("B", 20.0, "AutoCroissant")]
    events = [{"kind": "edited", "card": "A", "commit": "c1"}, {"kind": "changed", "card": "C", "commit": "B"}]
    by = {e["card"]: e["push"] for e in cl.assign_pushes([dict(e) for e in events], [], commits, "c0", {"B"})}
    assert by["A"] == {"after": "c1", "before": "c0", "by": "AutoCroissant", "at": 10.0}
    assert by["C"] == {"after": "B", "before": "c1", "by": cl.CARD_SYNC, "at": 20.0}
    merged = {e["card"]: e["push"]["after"] for e in cl.assign_pushes([dict(e) for e in events], [], commits, "c0")}
    assert merged == {"A": "B", "C": "B"}                                         # what --baked prevents


def test_payload_words_links_and_order():
    events = [dict(e, push=PUSH) for e in cl.number([
        {"kind": "new", "card": "Kirby", "title": "Kirby", "card_type": "creature", "stars": 3, "by": "Bob"},
        {"kind": "new", "card": "Big Pal", "card_type": "creatures", "stars": 4},
        {"kind": "changed", "card": "Waddle Dee", "fields": {"hp": [5, 6], "ability": ["a", "b"], "tags": [[], []]},
         "verdict": "buff"},
        {"kind": "changed", "card": "Waddle Doo", "fields": {"atk": [1, 2]}},
        {"kind": "archived", "card": "Gone"}, {"kind": "renamed", "card": "ZOEY!!!!!", "renamed_to": "Z!!!!!"},
        {"kind": "edited", "card": "Meta", "fields": {"subseries": [None, "Kirby"]}, "edit_by": "mike"},
        {"kind": "reparse", "card": "R"}, {"kind": "reparse", "card": "S"}], 0, at=T)]
    body = cl.payload(events, "https://site", {"Kirby": "kirby"}, "MichaelJSr/TTSCardMaker")
    assert body["allowed_mentions"] == {"parse": []} and "content" not in body
    first = body["embeds"][0]
    assert first["title"] == "Zomoroko, Oct 22 14:13: 2 new, 2 changed, 1 renamed, 1 archived, 1 edited, 2 re-read"
    assert first["url"] == "https://github.com/MichaelJSr/TTSCardMaker/compare/a1a1a1a1a1a1...a2a2a2a2a2a2"
    assert [e["description"].split("\n")[0] for e in body["embeds"]] == \
        ["**New cards**", "**Changed**", "**Moved and renamed**", "**Archived**", "**Database edits**", "The parser re-read 2 cards."]
    text = "\n".join(e["description"] for e in body["embeds"])
    assert "[Kirby](https://site/c/kirby/) (Creature, 3★) by Bob" in text
    assert "(Auxiliary creature, 4★)" in text                                    # app.js's word, never "Other"
    assert "[Waddle Dee](https://site/#/cards?q=Waddle%20Dee): HP 5 → 6; ability changed (buff)" in text
    assert "ZOEY!!!!! is now [Z!!!!!](" in text and "Gone (PSD deleted)" in text
    assert "[Meta](https://site/#/cards?q=Meta): subseries → Kirby (by mike, on the website)" in text
    assert cl.payload([], "https://site") is None


def test_links_by_the_sites_key_and_rulebook_pages_to_the_rulebook(tmp_path):
    """cards.json's `name` is the shown title: ids go by `key` (the database's name), else name; a rulebook page (not
    in cards.json) links the rulebook searched for it, never a card search."""
    site = {"cards": [{"id": "metal-general", "name": "MEtal General", "key": "Metal General"},
                      {"id": "kirby", "name": "Kirby"}]}
    assert cl.site_ids(site) == {"Metal General": "metal-general", "Kirby": "kirby"}
    db = str(tmp_path / "db")
    cl.append_events(db, [dict(e, push=PUSH) for e in cl.number([
        {"kind": "changed", "card": "Metal General", "title": "MEtal General", "card_type": "creature", "fields": {"hp": [1, 2]}},
        {"kind": "changed", "card": "11 Abilities2", "card_type": "rulebook", "path": "Auxiliary/Rulebook/11_Abilities2.psd",
         "fields": {"ability": ["a", "b"]}},
        {"kind": "new", "card": "Page", "card_type": "rulebook", "stars": None}], 0, at=T)])
    (tmp_path / "cards.json").write_text(json.dumps(site), encoding="utf-8")
    out = tmp_path / "discord.json"
    assert cl.main(["post", "--db", db, "--out", str(out), "--ids", str(tmp_path / "cards.json"), "--site", "https://site"]) == 0
    text = "\n".join(e.get("description") or "" for e in json.loads(out.read_text(encoding="utf-8"))["messages"][0]["body"]["embeds"])
    assert "[MEtal General](https://site/c/metal-general/): HP 1 → 2" in text
    assert "[11 Abilities2](https://site/#/rules?q=11%20Abilities2): ability changed" in text
    assert "[Page](https://site/#/rules?q=Page) (Rulebook page)" in text and "#/cards?q=" not in text


def test_every_kind_posts_alone():
    """A group of any one kind (a rollback's lone restore event included) makes a titled message with a count; the
    guards' kinds make none."""
    for kind in (k for k in cl.KINDS if k not in cl.SILENT):
        event = {"kind": kind, "card": "A", "renamed_to": "B", "restored_from": "f00dfeed" * 5, "fields": {"hp": [1, 2]}}
        group = cl.number([event], 0, at=T, build="https://x/runs/1")
        out = cl.messages(group, "https://site", repository="o/r")
        title = out[0]["body"]["embeds"][0]["title"]
        assert len(out) == 1 and title.startswith("Card sync, ") and not title.endswith(": "), kind
    body = cl.payload(cl.number([{"kind": "restore", "card": "", "restored_from": "f00dfeed" * 5}], 0, at=T))
    assert body["embeds"][0]["title"].endswith(": 1 data rollback")
    assert body["embeds"][0]["description"] == "**Rollback**\nCard data restored from f00dfeed"


def test_bake_and_guard_events():
    """The bake's events ride their push's post; the guards' (and every event of a guards commit's push) never post:
    a group of them alone is a message without a body, so the cursor still moves past it (owner, 2026-10-25)."""
    commits = [("m", 10.0, "AutoCroissant"), ("g", 20.0, "AutoCroissant")]
    pushed = cl.assign_pushes([{"kind": "baked", "card": "A", "commit": "m"}, {"kind": "art", "card": "B", "commit": "g"}],
                              [], commits, "c0", {"m"}, {"g"})
    assert [(e["push"]["after"], e["push"]["by"]) for e in pushed] == [("m", cl.CARD_SYNC), ("g", cl.GUARDS)]
    sync = dict(PUSH, by=cl.CARD_SYNC)
    guards = dict(PUSH, after="b2" * 20, by=cl.GUARDS)
    events = cl.number([dict(e, push=sync) for e in (
        {"kind": "baked", "card": "Kirby", "reason": "hp 5 -> 8", "pngs": ["a.png"]},
        {"kind": "bake-refused", "card": "Meta", "reason": "font: no Face"},
        {"kind": "guard-fix", "card": "Kirby", "line": "straight apostrophes"})]
        + [dict(e, push=guards) for e in ({"kind": "art", "card": "Kirby"}, {"kind": "guard-png", "card": "Kirby", "line": "re-exported"})], 0, at=T)
    first, second = cl.messages(events, "https://site", repository="o/r")
    assert first["body"]["embeds"][0]["title"] == "Website edits baked into the PSDs, Oct 22 14:13: 1 baked, 1 bake refused"
    text = "\n".join(e["description"] for e in first["body"]["embeds"])
    assert text == ("**Baked into the PSDs**\n[Kirby](https://site/#/cards?q=Kirby): hp 5 -> 8 (PNG re-made)\n"
                    "**Bake refused**\n[Meta](https://site/#/cards?q=Meta): font: no Face")
    assert second == {"body": None, "cursor": {"seq": 5, "push": "b2" * 20}}
    assert cl.messages(events[2:3] + events[3:], "https://site") == [{"body": None, "cursor": {"seq": 3, "push": PUSH["after"]}},
                                                                     {"body": None, "cursor": {"seq": 5, "push": "b2" * 20}}]


def test_payload_stays_within_discord_limits():
    events = [dict(e, push=PUSH) for e in cl.number(
        [{"kind": k, "card": f"{k} card {i} " + "x" * 60, "card_type": "item", "fields": {"ability": ["a", "b"]}}
         for k in ("new", "changed", "art", "moved", "archived", "restored", "edited") for i in range(150)], 0, at=T)]
    body = cl.payload(events, "https://michaeljsr.github.io/TTSCardMaker")
    assert len(body["embeds"]) <= cl.EMBEDS_MAX and cl.size(body) <= cl.MESSAGE_MAX
    assert all(len(e["description"]) <= cl.DESCRIPTION_MAX for e in body["embeds"])
    last = body["embeds"][len(body["embeds"]) - 1]["description"]
    shown = sum(e["description"].count("\n") for e in body["embeds"][:len(body["embeds"]) - 1])   # less the headings
    assert last == f"{len(events) - shown} more on the website: https://michaeljsr.github.io/TTSCardMaker/#/changes/changelog"


def test_one_message_per_push_at_most_twenty(tmp_path):
    """Pushes post oldest first, one message each (a re-read on its own); 25 pushes: 20 now, 5 on the next run."""
    def push(i):
        return {"after": f"{i:040x}", "before": f"{i + 100:040x}", "by": f"maker{i}", "at": T + i}
    events = [{"kind": "new", "card": f"Card {i}", "push": push(i), "build": "1"} for i in range(25)]
    events.insert(1, {"kind": "changed", "card": "Card 0b", "push": push(0), "build": "1"})
    events.append({"kind": "reparse", "card": "R", "build": "1"})
    numbered = cl.number(events, 0, at=T)
    out = cl.messages(numbered, "https://site", repository="o/r")
    assert len(out) == 20 and out[0]["cursor"] == {"seq": 2, "push": push(0)["after"]}
    assert out[0]["body"]["embeds"][0]["title"].startswith("maker0, ") and out[1]["cursor"]["seq"] == 3
    rest = cl.messages([e for e in numbered if e["seq"] > out[19]["cursor"]["seq"]], "https://site", repository="o/r")
    assert len(rest) == 6 and rest[5]["cursor"] == {"seq": 27, "push": None}
    assert rest[5]["body"]["embeds"][0]["description"] == "The parser re-read 1 card."


def test_cursor_moves_only_after_each_message(tmp_path):
    db = str(tmp_path / "db")
    p1 = dict(PUSH)
    p2 = dict(PUSH, after="b2" * 20, before=PUSH["after"], by="Bob")
    cl.append_events(db, [dict(e, push=p) for e, p in zip(
        cl.number([{"kind": "new", "card": "A"}, {"kind": "new", "card": "B"}], 0, at=T), (p1, p2))])
    out = tmp_path / "discord.json"
    assert cl.main(["post", "--db", db, "--out", str(out), "--repo", "o/r"]) == 0
    sent = json.loads(out.read_text(encoding="utf-8"))["messages"]
    assert [m["cursor"] for m in sent] == [{"seq": 1, "push": p1["after"]}, {"seq": 2, "push": p2["after"]}]
    assert cl.read_posted(db) == {"seq": 0, "push": None}
    assert cl.main(["advance", "--db", db, "--seq", "1", "--push", p1["after"]]) == 0      # the first went through
    assert cl.main(["post", "--db", db, "--out", str(out)]) == 0
    left = json.loads(out.read_text(encoding="utf-8"))["messages"]
    assert [m["cursor"]["seq"] for m in left] == [2] and left[0]["body"]["embeds"][0]["title"].startswith("Bob, ")
    cl.advance(db, 2, p2["after"])
    assert cl.main(["post", "--db", db, "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["messages"] == []
    cl.advance(db, 1, p1["after"])
    assert cl.read_posted(db) == {"seq": 2, "push": p2["after"]}                     # never backwards

