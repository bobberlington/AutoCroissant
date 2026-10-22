"""discord_archive against a fake Discord (GETs only): paging, resume, rate limits, categories, threads, downloads, redaction."""
import json
import os
import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from requests import ConnectionError as RequestsConnectionError

import commands.discord_archive as da
from conftest import REPO, Reply

G, BOT, ROLE = "100000000000000001", "200000000000000002", "77"     # made-up ids
TOKEN = "FAKE.tok3n-for_tests.Xy9"
BASE = (1600000000000 - da.EPOCH_MS) << 22
READ = da.VIEW_CHANNEL | da.READ_HISTORY


def sf(n: int) -> str:
    """A snowflake n ms after BASE."""
    return str(BASE + (n << 22))


class FakeDiscord:
    """A guild behind Discord's REST paging rules (newest first, after/before cursors); it has only get()."""
    def __init__(self, everyone: int = READ, role: int = 0):
        self.guild = {"id": G, "owner_id": "1", "roles": [{"id": G, "permissions": str(everyone)},
                                                           {"id": ROLE, "permissions": str(role)}]}
        self.member = {"user": {"id": BOT}, "roles": [ROLE]}
        self.channels, self.active, self.messages, self.files = [], [], {}, {}
        self.archived = {"public": {}, "private": {}}
        self.forbidden, self.queued, self.calls, self.joined = set(), {}, [], set()

    def channel(self, cid, name, kind=0, parent=None, position=0, overwrites=None, n=0):
        self.channels.append({"id": cid, "name": name, "type": kind, "parent_id": parent, "position": position,
                              "permission_overwrites": overwrites or [], "guild_id": G})
        self.add(cid, range(1, n + 1))

    def thread(self, tid, name, parent, kind=11, where="active", stamp="2021-01-01T00:00:00+00:00", n=1):
        thread = {"id": tid, "name": name, "type": kind, "parent_id": parent, "guild_id": G,
                  "thread_metadata": {"archived": where != "active", "archive_timestamp": stamp}}
        if where == "active":
            self.active.append(thread)
        else:
            self.archived[where].setdefault(parent, []).append(thread)
        self.add(tid, range(1, n + 1))

    def add(self, cid, numbers, **extra):
        self.messages.setdefault(cid, []).extend(
            {"id": sf(k), "timestamp": f"2020-09-13T12:26:{k % 60:02d}+00:00", "edited_timestamp": None,
             "content": f"message {k}", "author": {"id": "5", "username": "maker", "global_name": "Maker"},
             "attachments": [], "embeds": [], **extra} for k in numbers)

    def get(self, url, params=None, headers=None, timeout=None, stream=False, allow_redirects=True):
        parts, params = urlsplit(url), dict(params or {})
        path = parts.path.removeprefix("/api/v10")
        self.calls.append({"host": parts.hostname, "path": path, "params": params, "headers": dict(headers or {})})
        assert headers["User-Agent"].startswith("DiscordBot")
        if da.is_cdn(parts.hostname):
            assert "Authorization" not in headers                           # the CDN never gets the token
            data = self.files.get(url)
            return Reply(200, data) if data is not None else Reply(404, b"<Error>expired</Error>")
        assert headers["Authorization"] == f"Bot {TOKEN}"
        if self.queued.get(path):
            return self.queued[path].pop(0)
        return self.route(path, params)

    def route(self, path, params):
        fixed = {"/users/@me": {"id": BOT}, f"/guilds/{G}": self.guild, f"/guilds/{G}/members/{BOT}": self.member,
                 f"/guilds/{G}/channels": self.channels, f"/guilds/{G}/threads/active": {"threads": self.active}}
        if path in fixed:
            return Reply(200, body=fixed[path])
        part = path.strip("/").split("/")
        cid = part[1]
        if cid in self.forbidden:
            return Reply(403, body={"message": "Missing Access", "code": 50001})
        limit = int(params.get("limit", 50))
        if part[2:3] == ["threads"]:                                       # before: a time, strict
            stamp = lambda t: datetime.fromisoformat(t["thread_metadata"]["archive_timestamp"])   # noqa: E731
            threads = sorted(self.archived[part[4]].get(cid, []), key=stamp, reverse=True)
            threads = [t for t in threads if "before" not in params
                       or stamp(t) < datetime.fromisoformat(params["before"])]
            return Reply(200, body={"threads": threads[:limit], "members": [], "has_more": len(threads) > limit})
        if part[2:4] == ["users", "@me"]:                                  # joined private: before an id
            threads = sorted((t for t in self.archived["private"].get(cid, []) if t["id"] in self.joined),
                             key=lambda t: int(t["id"]), reverse=True)
            threads = [t for t in threads if "before" not in params or int(t["id"]) < int(params["before"])]
            return Reply(200, body={"threads": threads[:limit], "members": [], "has_more": len(threads) > limit})
        ordered = sorted(self.messages.get(cid, []), key=lambda m: int(m["id"]))
        if len(part) == 4:
            found = [m for m in ordered if m["id"] == part[3]]
            return Reply(200, body=found[0]) if found else Reply(404, body={"message": "Unknown Message", "code": 10008})
        if "after" in params:
            chosen = [m for m in ordered if int(m["id"]) > int(params["after"])][:limit]
        elif "before" in params:
            chosen = [m for m in ordered if int(m["id"]) < int(params["before"])][-limit:]
        else:
            chosen = ordered[-limit:]
        return Reply(200, body=list(reversed(chosen)))                    # Discord: newest first


OPEN: list = []                                                          # stores of earlier "runs"


def make(tmp_path, fake, **options):
    while OPEN:
        OPEN.pop().close()                                                  # the run before has ended
    clock, sleeps, logs = [0.0], [], []

    def sleeper(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    rest = da.Rest(TOKEN, fake, sleeper, lambda: clock[0])
    OPEN.append(da.Store(str(tmp_path / "out")))
    return da.Miner(rest, OPEN[-1], G, log=logs.append, **options), sleeps


def stored(tmp_path, cid):
    return [json.loads(line) for line in (tmp_path / "out" / "messages" / f"{cid}.jsonl").read_text().splitlines()]


def cursor(tmp_path, cid):
    return json.loads((tmp_path / "out" / "cursors" / f"{cid}.json").read_text())


def message_calls(fake, cid):
    return [c["params"] for c in fake.calls if c["path"] == f"/channels/{cid}/messages"]


# ========================
# Paging and resume
# ========================
def test_pages_oldest_first(tmp_path):
    fake = FakeDiscord()
    fake.channel("101", "card-archive", n=250)
    miner, _ = make(tmp_path, fake)
    summary = miner.run(["101"])
    got = stored(tmp_path, "101")
    assert [m["id"] for m in got] == [sf(k) for k in range(1, 251)]
    assert message_calls(fake, "101") == [{"limit": 100, "after": "0"}, {"limit": 100, "after": sf(100)},
                                          {"limit": 100, "after": sf(200)}]
    assert cursor(tmp_path, "101")["start_reached"] and cursor(tmp_path, "101")["count"] == 250
    first = got[0]
    assert first["channel_id"] == "101" and first["channel_name"] == "card-archive" and first["thread"] is None
    assert first["author"] == {"id": "5", "username": "maker", "global_name": "Maker", "bot": False}
    assert first["jump_url"] == f"https://discord.com/channels/{G}/101/{sf(1)}"
    assert summary["totals"]["new"] == 250 and summary["channels"][0]["first"].startswith("2020-09-13")


def test_newest_first_then_resume_fills_both_ends(tmp_path):
    fake = FakeDiscord()
    fake.channel("101", "cards", n=250)
    miner, _ = make(tmp_path, fake, order="newest", limit=150)
    miner.run(["101"])
    got = stored(tmp_path, "101")
    assert len(got) == 150 and got[0]["id"] == sf(250)
    assert message_calls(fake, "101") == [{"limit": 100}, {"limit": 50, "before": sf(151)}]
    assert not cursor(tmp_path, "101")["start_reached"] and cursor(tmp_path, "101")["caught_up"]

    fake.add("101", range(251, 271))
    fake.calls.clear()
    make(tmp_path, fake, order="newest")[0].run(["101"])
    ids = [m["id"] for m in stored(tmp_path, "101")]
    assert len(ids) == len(set(ids)) == 270 and set(ids) == {sf(k) for k in range(1, 271)}
    assert message_calls(fake, "101") == [{"limit": 100, "after": sf(250)}, {"limit": 100, "before": sf(101)},
                                          {"limit": 100, "before": sf(1)}]
    assert cursor(tmp_path, "101")["start_reached"]


def test_rerun_appends_only_new_messages(tmp_path):
    fake = FakeDiscord()
    fake.channel("101", "cards", n=150)
    make(tmp_path, fake)[0].run(["101"])
    fake.add("101", range(151, 181))
    fake.calls.clear()
    summary = make(tmp_path, fake)[0].run(["101"])
    assert message_calls(fake, "101") == [{"limit": 100, "after": sf(150)}]
    assert [m["id"] for m in stored(tmp_path, "101")] == [sf(k) for k in range(1, 181)]
    assert summary["totals"]["new"] == 30 and summary["channels"][0]["messages"] == 180
    fake.calls.clear()
    assert make(tmp_path, fake)[0].run(["101"])["totals"]["new"] == 0
    assert len(message_calls(fake, "101")) == 1 and len(stored(tmp_path, "101")) == 180


def test_a_page_repeating_the_cursor_adds_no_duplicate(tmp_path):
    class Repeats(FakeDiscord):
        def route(self, path, params):
            reply = super().route(path, params)
            if path.endswith("/messages") and params.get("after", "0") != "0":
                reply._body = reply._body + [m for m in self.messages["101"] if m["id"] == params["after"]]
            return reply
    fake = Repeats()
    fake.channel("101", "cards", n=100)
    make(tmp_path, fake)[0].run(["101"])
    fake.add("101", [101])
    make(tmp_path, fake)[0].run(["101"])
    assert [m["id"] for m in stored(tmp_path, "101")] == [sf(k) for k in range(1, 102)]


def test_torn_last_line_is_cut_and_refetched(tmp_path):
    fake = FakeDiscord()
    fake.channel("101", "cards", n=120)
    make(tmp_path, fake, limit=100)[0].run(["101"])
    path = tmp_path / "out" / "messages" / "101.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"id": "' + sf(101)[:6])                                   # a crash mid-write
    make(tmp_path, fake)[0].run(["101"])
    assert [m["id"] for m in stored(tmp_path, "101")] == [sf(k) for k in range(1, 121)]


# ========================
# Rate limits and errors
# ========================
def test_429_waits_retry_after_and_retries(tmp_path):
    fake = FakeDiscord()
    fake.channel("101", "cards", n=10)
    fake.queued["/channels/101/messages"] = [Reply(429, body={"message": "You are being rate limited.",
                                                              "retry_after": 1.5, "global": False})]
    miner, sleeps = make(tmp_path, fake)
    summary = miner.run(["101"])
    assert len(stored(tmp_path, "101")) == 10 and summary["rate_limited"] == 1
    assert 1.5 <= max(sleeps) < 1.6


def test_cloudflare_429_uses_retry_after_header_and_global():
    fake = FakeDiscord()
    fake.queued["/users/@me"] = [Reply(429, b"<html>slow down</html>", headers={"Retry-After": "3"})]
    fake.queued[f"/guilds/{G}"] = [Reply(429, body={"retry_after": 2, "global": True})]
    clock, sleeps = [0.0], []
    rest = da.Rest(TOKEN, fake, lambda s: (sleeps.append(s), clock.__setitem__(0, clock[0] + s)), lambda: clock[0])
    assert rest.get("/users/@me") == {"id": BOT}
    assert rest.get(f"/guilds/{G}")["id"] == G
    assert [round(s, 2) for s in sleeps if s > 1] == [3.05, 2.05] and rest.limited == 2


def test_empty_bucket_waits_reset_after_on_that_route_only():
    fake = FakeDiscord()
    fake.queued["/users/@me"] = [Reply(200, body={"id": BOT}, headers={"x-ratelimit-remaining": "0",
                                                                       "x-ratelimit-reset-after": "4"})]
    clock, sleeps = [0.0], []
    rest = da.Rest(TOKEN, fake, lambda s: (sleeps.append(s), clock.__setitem__(0, clock[0] + s)), lambda: clock[0])
    rest.get("/users/@me")
    rest.get(f"/guilds/{G}")                                                # another bucket: only the pacing gap
    assert max(sleeps) < 0.1
    rest.get("/users/@me")
    assert round(max(sleeps), 2) == round(4.05 - da.MIN_GAP, 2)


def test_message_routes_share_a_bucket():
    assert da.route_key(f"/channels/101/messages/{sf(1)}") == da.route_key(f"/channels/101/messages/{sf(2)}")
    assert da.route_key("/channels/101/messages") != da.route_key("/channels/102/messages")
    assert da.route_key(f"/guilds/{G}/members/{BOT}") == f"/guilds/{G}/members/:id"
    fake = FakeDiscord()
    fake.channel("101", "cards", n=2)
    fake.queued[f"/channels/101/messages/{sf(1)}"] = [Reply(200, body={"id": sf(1)}, headers={
        "X-RateLimit-Remaining": "0", "X-RateLimit-Reset-After": "3"})]
    clock, sleeps = [0.0], []
    rest = da.Rest(TOKEN, fake, lambda s: (sleeps.append(s), clock.__setitem__(0, clock[0] + s)), lambda: clock[0])
    rest.get(f"/channels/101/messages/{sf(1)}")
    rest.get(f"/channels/101/messages/{sf(2)}")                             # waits before, no 429
    assert round(max(sleeps), 2) == 3.05 and rest.limited == 0


def test_repeated_429s_have_a_floor_and_stop_the_run(tmp_path, monkeypatch):
    fake = FakeDiscord()
    fake.queued["/users/@me"] = [Reply(429, b"", headers={"Retry-After": "0"})]
    clock, sleeps = [0.0], []
    rest = da.Rest(TOKEN, fake, lambda s: (sleeps.append(s), clock.__setitem__(0, clock[0] + s)), lambda: clock[0])
    rest.get("/users/@me")
    assert max(sleeps) >= da.MIN_LIMIT_WAIT

    monkeypatch.setattr(da, "INVALID_CAP", 5)
    fake = FakeDiscord()
    for cid in ("101", "102", "103"):
        fake.channel(cid, "cards", n=1)
        fake.queued[f"/channels/{cid}/messages"] = [Reply(429, body={"retry_after": 0}) for _ in range(3)]
    miner, _ = make(tmp_path, fake)
    with pytest.raises(da.ApiError) as stopped:
        miner.run(["101", "102", "103"])
    assert stopped.value.fatal and "rejected requests" in stopped.value.reason()
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["status"].startswith("stopped")
    assert not message_calls(fake, "103")


def test_server_errors_back_off_then_succeed():
    fake = FakeDiscord()
    fake.queued["/users/@me"] = [Reply(502), Reply(503)]
    clock, sleeps = [0.0], []
    rest = da.Rest(TOKEN, fake, lambda s: (sleeps.append(s), clock.__setitem__(0, clock[0] + s)), lambda: clock[0])
    assert rest.get("/users/@me") == {"id": BOT}
    assert [round(s, 2) for s in sleeps if s > 0.5] == [1.0, 2.0]


def test_not_a_member_stops_with_a_hint(tmp_path):
    fake = FakeDiscord()
    fake.queued[f"/guilds/{G}"] = [Reply(404, body={"message": "Unknown Guild", "code": 10004})]
    miner, _ = make(tmp_path, fake)
    with pytest.raises(SystemExit, match="isn't a member"):
        miner.run([], include_all=True)
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["status"].startswith("stopped")


# ========================
# Channels, categories, threads
# ========================
def test_category_expands_to_its_channels_in_order(tmp_path):
    fake = FakeDiscord()
    fake.channel("200", "Archive", kind=da.CATEGORY, position=5)
    fake.channel("202", "b-archive", parent="200", position=2, n=3)
    fake.channel("201", "a-archive", parent="200", position=1, n=2)
    fake.channel("300", "general", n=4)
    summary = make(tmp_path, fake)[0].run(["200"])
    assert [r["id"] for r in summary["channels"]] == ["201", "202"]
    assert not message_calls(fake, "300")
    inventory = {e["id"]: e for e in json.loads((tmp_path / "out" / "channels.json").read_text())}
    assert inventory["200"]["status"] == "category" and inventory["300"]["status"] == "not selected"
    assert inventory["201"]["status"] == "mined" and inventory["201"]["messages"] == 2


def test_all_mode_puts_given_channels_first_and_records_skips(tmp_path):
    deny_view = [{"id": G, "type": 0, "allow": "0", "deny": str(da.VIEW_CHANNEL)}]
    fake = FakeDiscord()
    fake.channel("300", "general", n=2)
    fake.channel("200", "Archive", kind=da.CATEGORY, position=1)
    fake.channel("201", "old-cards", parent="200", n=2)
    fake.channel("400", "hidden", overwrites=deny_view, n=2)
    fake.channel("401", "no-history", overwrites=[{"id": ROLE, "type": 0, "allow": "0", "deny": str(da.READ_HISTORY)}],
                 n=2)
    fake.channel("402", "bot-allowed", overwrites=deny_view + [{"id": ROLE, "type": 0, "allow": str(READ), "deny": "0"}],
                 n=1)
    fake.channel("500", "voice", kind=2, n=1)
    fake.channel("600", "locked", n=1)
    fake.forbidden.add("600")
    fake.channel("700", "directory", kind=14)
    summary = make(tmp_path, fake)[0].run(["201"], include_all=True)
    rows = {r["id"]: (r["status"], r.get("reason")) for r in summary["channels"]}
    assert summary["channels"][0]["id"] == "201"
    assert rows["400"] == ("skipped", "no View Channel") and rows["401"] == ("skipped", "no Read Message History")
    assert rows["600"] == ("skipped", "HTTP 403 Missing Access (50001)")
    assert rows["700"][0] == "skipped" and "no messages" in rows["700"][1]
    assert rows["402"][0] == rows["500"][0] == rows["300"][0] == "mined"
    assert not message_calls(fake, "400") and not message_calls(fake, "401")
    assert len(stored(tmp_path, "500")) == 1


def test_threads_active_archived_and_private_need_manage_threads(tmp_path, monkeypatch):
    monkeypatch.setattr(da, "PAGE", 2)
    fake = FakeDiscord()
    fake.channel("101", "cards", n=1)
    fake.channel("150", "card-forum", kind=15)
    fake.thread("111", "live", "101", n=3)
    for i in range(3):
        fake.thread(str(120 + i), f"old {i}", "101", where="public", stamp=f"2021-01-0{i + 1}T00:00:00+00:00")
    fake.thread("130", "secret", "101", kind=da.PRIVATE_THREAD, where="private")
    fake.thread("131", "joined secret", "101", kind=da.PRIVATE_THREAD, where="private")
    fake.joined.add("131")
    fake.thread("151", "a post", "150", where="public")
    summary = make(tmp_path, fake)[0].run(["101", "150"])
    mined = [r["id"] for r in summary["channels"] if r["status"] == "mined"]
    assert mined == ["101", "111", "120", "121", "122", "131", "151"]
    assert not any("/threads/archived/private" in c["path"] and "/users/" not in c["path"] for c in fake.calls)
    assert not message_calls(fake, "150") and not message_calls(fake, "130")
    public = [c["params"] for c in fake.calls if c["path"] == "/channels/101/threads/archived/public"]
    assert public == [{"limit": 2}, {"limit": 2, "before": "2021-01-02T00:00:00.001000+00:00"}]   # 1 ms past
    record = stored(tmp_path, "111")[0]
    assert record["channel_id"] == "101" and record["thread"] == {"id": "111", "name": "live", "type": 11}
    assert record["jump_url"] == f"https://discord.com/channels/{G}/111/{sf(1)}"
    inventory = {e["id"]: e for e in json.loads((tmp_path / "out" / "channels.json").read_text())}
    assert inventory["101"]["threads"] == 5 and "no Manage Threads" in inventory["101"]["thread_notes"][0]
    assert inventory["151"]["thread_of"] == "150" and inventory["150"]["status"] == "threads only"

    fake.calls.clear()                                                      # a thread given by id, then its channel
    given = make(tmp_path / "given", fake)[0].run(["111", "101"])
    assert [r["id"] for r in given["channels"]] == ["111", "101", "120", "121", "122", "131"]
    assert [c["after"] for c in message_calls(fake, "111")] == ["0", sf(2)]   # mined once
    assert stored(tmp_path / "given", "111")[0]["thread"]["id"] == "111"

    fake.calls.clear()                                                      # its channel, then the thread by id
    after = make(tmp_path / "after", fake)[0].run(["101", "111"])
    assert [r["id"] for r in after["channels"]].count("111") == 1
    assert [c["after"] for c in message_calls(fake, "111")] == ["0", sf(2)]

    fake.calls.clear()                                                      # a rerun skips stored archived threads
    rerun = make(tmp_path, fake)[0].run(["101"])
    assert {r["id"]: r["status"] for r in rerun["channels"]}["120"] == "unchanged"
    assert not message_calls(fake, "120") and len(message_calls(fake, "111")) == 1

    managed = FakeDiscord(role=da.MANAGE_THREADS)
    managed.channel("101", "cards", n=1)
    managed.thread("130", "secret", "101", kind=da.PRIVATE_THREAD, where="private")
    rows = make(tmp_path / "managed", managed)[0].run(["101"])["channels"]
    assert [r["id"] for r in rows] == ["101", "130"]
    assert not any("/users/@me/threads" in c["path"] for c in managed.calls)


def test_archived_threads_tied_at_a_page_edge_are_all_found(tmp_path, monkeypatch):
    monkeypatch.setattr(da, "PAGE", 2)
    fake = FakeDiscord()
    fake.channel("101", "cards", n=1)
    fake.thread("120", "oldest", "101", where="public", stamp="2021-01-01T00:00:00.000000+00:00")
    for tid in ("121", "122"):                                              # a bulk archive: one stamp
        fake.thread(tid, "tied", "101", where="public", stamp="2021-01-02T00:00:00.000000+00:00")
    fake.thread("123", "newest", "101", where="public", stamp="2021-01-03T00:00:00.000000+00:00")
    rows = make(tmp_path, fake)[0].run(["101"])["channels"]
    assert [r["id"] for r in rows] == ["101", "120", "121", "122", "123"]
    assert len([c for c in fake.calls if c["path"].endswith("/archived/public")]) == 3

    fake = FakeDiscord()                                                    # a whole page tied: steps past it
    fake.channel("101", "cards", n=1)
    for tid in ("121", "122"):
        fake.thread(tid, "tied", "101", where="public", stamp="2021-01-02T00:00:00+00:00")
    fake.thread("120", "older", "101", where="public", stamp="2021-01-01T00:00:00+00:00")
    rows = make(tmp_path / "full", fake)[0].run(["101"])["channels"]
    assert [r["id"] for r in rows] == ["101", "120", "121", "122"]


# ========================
# Attachments
# ========================
def test_images_are_stored_by_sha_with_failures_recorded(tmp_path):
    png = b"\x89PNG not really"
    cdn = "https://cdn.discordapp.com/attachments/101/{}/{}?ex=1&is=2&hm=3"

    def attachment(aid, name, kind="image/png", size=len(png), url=None):
        return {"id": aid, "filename": name, "size": size, "width": 360, "height": 540, "content_type": kind,
                "url": url or cdn.format(aid, name)}
    fake = FakeDiscord()
    fake.channel("101", "cards")
    fake.add("101", [1], attachments=[attachment("9001", "Mini Doomer.png"), attachment("9002", "copy.PNG"),
                                      attachment("9003", "huge.psd", "image/vnd.adobe.photoshop", 30 << 20),
                                      attachment("9004", "notes.txt", "text/plain")])
    fake.add("101", [2], attachments=[attachment("9005", "gone.png"),
                                      attachment("9006", "x.png", url="https://evil.example/x.png")],
             message_snapshots=[{"message": {"content": "fwd", "attachments": [attachment("9007", "fwd.webp")]}}])
    fake.files.update({cdn.format("9001", "Mini Doomer.png"): png, cdn.format("9002", "copy.PNG"): png,
                       cdn.format("9007", "fwd.webp"): b"RIFF webp"})
    summary = make(tmp_path, fake)[0].run(["101"])
    digest = sha256(png).hexdigest()
    one, two = stored(tmp_path, "101")
    got = {a["id"]: a for a in one["attachments"] + two["attachments"]}
    assert got["9001"]["sha256"] == got["9002"]["sha256"] == digest and got["9001"]["file"] == f"attachments/{digest}.png"
    assert got["9001"]["width"] == 360 and got["9001"]["url"].startswith("https://cdn.discordapp.com/")
    assert got["9003"]["download"] == "too_big" and "download" not in got["9004"] and "sha256" not in got["9004"]
    assert got["9005"]["download"] == "failed: HTTP 404 CDN" and "not a Discord CDN host" in got["9006"]["download"]
    assert two["forwarded"][0]["attachments"][0]["sha256"] == sha256(b"RIFF webp").hexdigest()
    files = sorted(p.name for p in (tmp_path / "out" / "attachments").iterdir())
    assert files == sorted([f"{digest}.png", f"{sha256(b'RIFF webp').hexdigest()}.webp"])
    hosts = [c["host"] for c in fake.calls]
    assert "evil.example" not in hosts and not any("huge.psd" in c["path"] for c in fake.calls)
    assert summary["totals"]["images"] == 3 and summary["totals"]["too_big"] == 1
    assert summary["totals"]["download_failures"] == 2

    fresh = cdn.format("9005", "gone.png") + "&fresh=1"                     # a re-read message has a new signed URL
    fake.messages["101"][1]["attachments"][0]["url"] = fresh
    fake.files[fresh] = b"fresh bytes"
    index_path = tmp_path / "out" / "attachments.jsonl"
    index_path.write_text(index_path.read_text() + '{"id": "90')                # a torn line, cut on load
    (tmp_path / "out" / "attachments" / ".part-9005").write_bytes(b"half")    # an interrupted download
    counts = make(tmp_path, fake)[0].retry_downloads()
    assert not (tmp_path / "out" / "attachments" / ".part-9005").exists()
    assert counts["images"] == 1
    index = {}
    for line in (tmp_path / "out" / "attachments.jsonl").read_text().splitlines():
        entry = json.loads(line)
        index[entry["id"]] = entry
    assert index["9005"]["sha256"] == sha256(b"fresh bytes").hexdigest()
    patched = stored(tmp_path, "101")[1]["attachments"][0]                 # the message line is patched too
    assert patched["sha256"] == index["9005"]["sha256"] and "download" not in patched
    assert stored(tmp_path, "101")[1]["attachments"][1]["download"].startswith("failed")


def test_same_bytes_share_one_file_and_a_cut_file_is_replaced(tmp_path):
    jpg = b"\xff\xd8 jpeg bytes"
    cdn = "https://cdn.discordapp.com/attachments/101/{}/{}"
    digest = sha256(jpg).hexdigest()
    folder = tmp_path / "out" / "attachments"
    folder.mkdir(parents=True)
    (folder / f"{digest}.jpg").write_bytes(jpg[:3])                         # cut short by a power loss
    fake = FakeDiscord()
    fake.channel("101", "cards")
    fake.add("101", [1], attachments=[{"id": i, "filename": name, "size": len(jpg), "content_type": "image/jpeg",
                                       "url": cdn.format(i, name)} for i, name in (("9001", "a.jpg"), ("9002", "b.jpeg"),
                                                                                    ("9003", "c.JPE"))])
    fake.files.update({cdn.format(i, name): jpg for i, name in (("9001", "a.jpg"), ("9002", "b.jpeg"),
                                                                 ("9003", "c.JPE"))})
    make(tmp_path, fake)[0].run(["101"])
    assert sorted(p.name for p in folder.iterdir()) == [f"{digest}.jpg"] and (folder / f"{digest}.jpg").read_bytes() == jpg
    assert {a["file"] for a in stored(tmp_path, "101")[0]["attachments"]} == {f"attachments/{digest}.jpg"}


def test_cdn_redirects_stay_on_the_cdn(tmp_path):
    class Redirects(FakeDiscord):
        def get(self, url, params=None, headers=None, timeout=None, stream=False, allow_redirects=True):
            assert not allow_redirects
            if url.endswith("/hop"):
                self.calls.append({"host": urlsplit(url).hostname, "path": urlsplit(url).path})
                return Reply(302, headers={"Location": self.hop})
            return super().get(url, params, headers, timeout, stream)
    fake = Redirects()
    fake.files["https://media.discordapp.net/real.png"] = b"real"
    rest = da.Rest(TOKEN, fake, lambda s: None, lambda: 0.0)
    fake.hop = "https://media.discordapp.net/real.png"
    assert rest.fetch_file("https://cdn.discordapp.com/hop", str(tmp_path / "a")) == (sha256(b"real").hexdigest(), 4)
    fake.hop = "https://evil.example/x.png"
    with pytest.raises(da.ApiError, match="not a Discord CDN host"):
        rest.fetch_file("https://cdn.discordapp.com/hop", str(tmp_path / "b"))
    assert "evil.example" not in [c["host"] for c in fake.calls]


# ========================
# Safety: the token, the out dir, GETs only
# ========================
def test_token_is_redacted_everywhere(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(da, "DEFAULT_OUT", str(tmp_path))
    class Boom(FakeDiscord):
        def get(self, url, params=None, headers=None, **kwargs):
            raise RequestsConnectionError(f"refused {headers.get('Authorization')} for {url}")
    monkeypatch.setattr(da, "MAX_TRIES", 2)
    monkeypatch.setattr(da, "sleep", lambda s: None)
    rest = da.Rest(TOKEN, Boom())
    with pytest.raises(da.ApiError) as caught:
        rest.get("/users/@me")
    assert TOKEN not in str(caught.value) and da.REDACTED in str(caught.value) and TOKEN not in repr(rest)

    config = tmp_path / "config.py"
    config.write_text(f"TOKEN = {TOKEN!r}\nADMINS = []\n")
    out = tmp_path / "archive"
    with pytest.raises(SystemExit) as stopped:
        da.main(["--guild", G, "--all", "--out", str(out), "--config", str(config)], session=Boom())
    assert TOKEN not in str(stopped.value.code) and da.REDACTED in str(stopped.value.code)

    edge = FakeDiscord()                                                    # the token across the 200-char cut
    edge.queued["/users/@me"] = [Reply(400, ("x" * 195 + TOKEN).encode())]
    with pytest.raises(da.ApiError) as cut:
        da.Rest(TOKEN, edge, lambda s: None, lambda: 0.0).get("/users/@me")
    assert TOKEN[:5] not in cut.value.text

    echo = FakeDiscord()
    echo.queued["/users/@me"] = [Reply(401, body={"message": f"401: Unauthorized {TOKEN}", "code": 0})]
    with pytest.raises(SystemExit) as refused:
        da.main(["--guild", G, "--all", "--out", str(out), "--config", str(config)], session=echo)
    assert "HTTP 401" in str(refused.value.code) and TOKEN not in str(refused.value.code)

    broken = tmp_path / "broken.py"
    broken.write_text(f'TOKEN = "{TOKEN}\n')                                 # a syntax error quoting the token line
    with pytest.raises(SystemExit) as unreadable:
        da.load_token(str(broken))
    assert TOKEN not in str(unreadable.value.code)
    printed = capsys.readouterr()
    assert TOKEN not in printed.out + printed.err
    assert out.exists() and not any(TOKEN.encode() in p.read_bytes() for p in out.rglob("*") if p.is_file())


def test_load_token_reads_without_running_config(tmp_path):
    config = tmp_path / "config.py"
    config.write_text(f"import os\nTOKEN = 'old'\nos.environ['RAN'] = '1'\nif os:\n    TOKEN = 'nested'\n"
                      f"TOKEN: str = ' {TOKEN} '\n")
    assert da.load_token(str(config)) == TOKEN                              # the last top-level one, as imported
    assert "RAN" not in __import__("os").environ


def test_out_dir_inside_a_repo_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(da, "DEFAULT_OUT", str(tmp_path / "archive"))
    with pytest.raises(SystemExit) as refused:                              # only the archive folder
        da.main(["--guild", G, "--all", "--out", str(tmp_path / "elsewhere"), "--config", str(tmp_path / "no.py")])
    assert refused.value.code == 2 and "must be" in capsys.readouterr().err and not (tmp_path / "elsewhere").exists()
    monkeypatch.setattr(da, "DEFAULT_OUT", str(tmp_path))
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    with pytest.raises(SystemExit) as refused:
        da.main(["--guild", G, "--all", "--out", str(tmp_path / "repo" / "data"), "--config", str(tmp_path / "no.py")])
    assert refused.value.code == 2 and not (tmp_path / "repo" / "data").exists()
    assert da.inside_git(str(REPO / "commands")) is not None and da.inside_git(str(tmp_path / "elsewhere")) is None
    os.symlink(tmp_path / "repo", tmp_path / "link")                         # a symlink into a repo
    assert da.inside_git(str(tmp_path / "link" / "data")) == os.path.realpath(tmp_path / "repo")


def test_bad_ids_are_refused_before_any_request(tmp_path):
    with pytest.raises(SystemExit) as refused:
        da.main(["--guild", G, "--channels", "../etc", "--out", str(tmp_path / "o"), "--config", str(tmp_path / "x")])
    assert refused.value.code == 2


def test_get_only():
    source = (REPO / "commands" / "discord_archive.py").read_text()
    assert not re.search(r"\.(post|put|patch|delete|request|head|options)\(", source)
    assert not re.search(r"^\s*(import discord|from discord)", source, re.M)
    assert re.findall(r"self\._session\.(\w+)\(", source) == ["get"]


def test_permissions_follow_discords_order():
    guild = {"id": G, "owner_id": "1", "roles": [{"id": G, "permissions": str(da.VIEW_CHANNEL)},
                                                  {"id": ROLE, "permissions": str(da.READ_HISTORY)}]}
    member = {"user": {"id": BOT}, "roles": [ROLE]}
    base = da.base_permissions(guild, member)
    assert base == READ
    channel = {"permission_overwrites": [{"id": G, "allow": "0", "deny": str(READ)},
                                         {"id": ROLE, "allow": str(da.VIEW_CHANNEL), "deny": "0"},
                                         {"id": BOT, "allow": "0", "deny": str(da.VIEW_CHANNEL)}]}
    assert da.channel_permissions(base, channel, G, member) == 0
    channel["permission_overwrites"].pop()
    assert da.channel_permissions(base, channel, G, member) == da.VIEW_CHANNEL
    admin = {**guild, "roles": guild["roles"] + [{"id": "9", "permissions": str(da.ADMINISTRATOR)}]}
    assert da.base_permissions(admin, {**member, "roles": [ROLE, "9"]}) == da.ALL_PERMISSIONS
    assert Path(da.CONFIG_PATH).name == "config.py"


# ========================
# Review fixes
# ========================
def test_odd_retry_after_values_wait_a_second():
    for odd in ([1, 2], "soon", float("nan"), float("inf")):
        reply = Reply(429, body={"retry_after": odd})
        assert da.Rest(TOKEN, FakeDiscord())._retry_after(reply)[0] == 1.0 + da.RATE_MARGIN


def test_a_full_page_repeating_the_cursor_backwards_leaves_no_gap(tmp_path):
    class Inclusive(FakeDiscord):                                           # before: the cursor comes back too
        def route(self, path, params):
            if path.endswith("/messages") and "before" in params:
                ordered = sorted(self.messages["101"], key=lambda m: int(m["id"]))
                chosen = [m for m in ordered if int(m["id"]) <= int(params["before"])][-int(params["limit"]):]
                return Reply(200, body=list(reversed(chosen)))
            return super().route(path, params)
    fake = Inclusive()
    fake.channel("101", "cards", n=250)
    make(tmp_path, fake, order="newest", limit=150)[0].run(["101"])
    make(tmp_path, fake, order="newest")[0].run(["101"])
    assert [m["id"] for m in stored(tmp_path, "101")] == [sf(k) for k in range(250, 100, -1)] + \
        [sf(k) for k in range(100, 0, -1)]
    assert cursor(tmp_path, "101")["start_reached"]


def test_one_run_at_a_time_and_no_symlinks(tmp_path):
    first = da.Store(str(tmp_path / "out"))
    (tmp_path / "out" / "attachments" / ".part-9001").write_bytes(b"mid download")
    with pytest.raises(SystemExit, match="another run"):
        da.Store(str(tmp_path / "out"))
    assert (tmp_path / "out" / "attachments" / ".part-9001").exists()      # the other run's file is kept
    first.close()
    target = tmp_path / "elsewhere.jsonl"
    target.write_text("")
    os.symlink(target, tmp_path / "out" / "messages" / "101.jsonl")
    store = da.Store(str(tmp_path / "out"))
    with pytest.raises(OSError):
        store.append("101", [{"id": "1"}])
    store.close()
    assert target.read_text() == ""
    os.symlink(tmp_path / "away", tmp_path / "linked")
    (tmp_path / "linked-out").mkdir()
    os.symlink(tmp_path / "away", tmp_path / "linked-out" / "messages")
    with pytest.raises(SystemExit, match="symlink"):
        da.Store(str(tmp_path / "linked-out"))
    assert not (tmp_path / "away").exists()


def test_unknown_permissions_fall_back_to_joined_private_threads(tmp_path):
    fake = FakeDiscord()
    fake.channel("101", "cards", n=1)
    fake.thread("131", "joined secret", "101", kind=da.PRIVATE_THREAD, where="private")
    fake.joined.add("131")
    fake.queued[f"/guilds/{G}/members/{BOT}"] = [Reply(403, body={"message": "Missing Access", "code": 50001})]
    fake.queued["/channels/101/threads/archived/private"] = [Reply(403, body={"message": "Missing Access",
                                                                              "code": 50001})]
    rows = make(tmp_path, fake)[0].run(["101"])["channels"]
    assert [r["id"] for r in rows] == ["101", "131"]
    notes = {e["id"]: e for e in json.loads((tmp_path / "out" / "channels.json").read_text())}["101"]["thread_notes"]
    assert "HTTP 403" in notes[0]


def test_all_mode_mines_orphans_and_lists_unplanned_threads(tmp_path):
    fake = FakeDiscord()
    fake.channel("300", "general", n=1)
    fake.channel("201", "orphan", parent="999", n=2)                        # its category isn't listed
    fake.channel("202", "kept", parent="998", n=1)
    fake.thread("211", "live elsewhere", "202", n=1)
    rows = make(tmp_path, fake)[0].run([], include_all=True)["channels"]
    assert {r["id"] for r in rows if r["status"] == "mined"} == {"300", "201", "202", "211"}
    listed = {e["id"]: e for e in json.loads((tmp_path / "out" / "channels.json").read_text())}
    make(tmp_path / "some", fake)[0].run(["300"])
    some = {e["id"]: e for e in json.loads((tmp_path / "some" / "out" / "channels.json").read_text())}
    assert listed["201"]["status"] == "mined"
    assert some["211"]["status"] == "not selected" and some["211"]["thread_of"] == "202"


def test_embed_images_are_stored_and_lost_index_failures_retried(tmp_path):
    cdn = "https://cdn.discordapp.com/attachments/1/2/card.png"
    proxy = "https://images-ext-1.discordapp.net/external/abc/https/example.com/art.webp"
    fake = FakeDiscord()
    fake.channel("101", "cards")
    fake.add("101", [1], embeds=[{"type": "image", "url": cdn, "thumbnail": {"url": cdn, "proxy_url": cdn}},
                                 {"type": "rich", "image": {"url": "https://example.com/art.webp", "proxy_url": proxy}},
                                 {"type": "rich", "image": {"url": "https://example.com/x.png"}},
                                 {"type": "link", "thumbnail": {"url": "https://[cdn.example/x.png"}}])
    fake.add("101", [2], content="cut \ud83d")                              # a lone surrogate, valid JSON
    fake.files[cdn] = b"card bytes"
    make(tmp_path, fake)[0].run(["101"])
    first, second, third, broken = stored(tmp_path, "101")[0]["embeds"]
    assert first["media"][0]["id"] == f"{sf(1)}-e0-thumbnail"
    assert first["media"][0]["file"] == f"attachments/{sha256(b'card bytes').hexdigest()}.png"
    assert second["media"][0]["download"] == "failed: HTTP 404 CDN" and "media" not in third and "media" not in broken
    assert stored(tmp_path, "101")[1]["content"] == "cut \ud83d"
    assert not any(c["host"] == "example.com" for c in fake.calls)

    (tmp_path / "out" / "attachments.jsonl").write_text("")                # a power loss took the index
    fake.files[proxy] = b"art bytes"
    counts = make(tmp_path, fake)[0].retry_downloads()
    assert counts["images"] == 1
    assert stored(tmp_path, "101")[0]["embeds"][1]["media"][0]["sha256"] == sha256(b"art bytes").hexdigest()


def test_mentions_are_kept_for_credits(tmp_path):
    fake = FakeDiscord()
    fake.channel("101", "art", n=0)
    fake.add("101", [1], mentions=[{"id": "7", "username": "artist", "global_name": "Artist", "avatar": "x"}])
    miner, _ = make(tmp_path, fake)
    miner.run(["101"])
    assert stored(tmp_path, "101")[0]["mentions"] == [{"id": "7", "username": "artist", "global_name": "Artist"}]
