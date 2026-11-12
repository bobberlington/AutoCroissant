"""forum_votes (owner 2026-11-09): the proposal forum's steps against a fake forum, its posts and a clock."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from itertools import count
from os.path import exists
from types import SimpleNamespace

import pytest
from discord import ChannelType, Forbidden, HTTPException, NotFound
from discord.ui import Button

import commands.analytics as rem
import commands.forum_votes as fv
import commands.reminder_panel as panel
import commands.utils as utils
from global_config import FORUM_JSON

T0 = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
BOT, PERSON, OTHER_BOT = 99, 7, 98
FORUM, ROLE, RESULTS, OTHER = 1569539619471499324, 546427268751294475, 1087579710428565526, 1087579710428565999
DAY6, DAY7 = T0 + timedelta(days=6, minutes=1), T0 + timedelta(days=7, minutes=1)
_ids = count(1000)


def _error(kind, status, code, text):
    return kind(SimpleNamespace(status=status, reason=text), {"code": code, "message": text})


# ========================
# Fakes
# ========================
class Tag:
    def __init__(self, id, name):
        self.id, self.name = id, name


class Emoji:
    def __init__(self, id, name):
        self.id, self.name = id, name

    def __str__(self):
        return f"<:{self.name}:{self.id}>"


class Reaction:
    def __init__(self, emoji, count, me=False):
        self.emoji, self.count, self.me = emoji, count, me


class Message:
    def __init__(self, channel, content, author=BOT, **extra):
        self.id, self.channel, self.content = next(_ids), channel, content
        self.author = SimpleNamespace(id=author, bot=author != PERSON)
        self.__dict__.update(extra)


class Starter:
    """A post's first message: its reactions, adding one."""
    def __init__(self, thread, content):
        self.thread, self.id, self.content, self.reactions = thread, thread.id, content, []

    async def add_reaction(self, emoji):
        if self.thread.archived:
            raise _error(HTTPException, 400, 50083, "Thread is archived")
        found = next((r for r in self.reactions if str(r.emoji) == str(emoji)), None)
        if found is None:
            self.reactions.append(Reaction(emoji, 1, True))
        elif not found.me:
            found.count, found.me = found.count + 1, True

    def vote(self, emoji, n):
        found = next((r for r in self.reactions if str(r.emoji) == str(emoji)), None)
        if found is None:
            self.reactions.append(Reaction(emoji, n))
        else:
            found.count += n


class Thread:
    def __init__(self, forum, created, tags=(), name="Bigger hands", archived=False, locked=False, owner=PERSON):
        self.id, self.parent_id, self.forum, self.guild, self.owner_id = next(_ids), forum.id, forum, forum.guild, owner
        self.created_at, self.name, self.archived, self.locked = created, name, archived, locked
        self.applied_tags = [forum.tag(n) for n in tags]
        self.jump_url = f"https://discord.com/channels/1/{self.id}"
        self.starter, self.messages, self.edits = Starter(self, "Please change this."), [], []
        self.deleted, self.refuse = False, set()
        (forum.archived if archived else forum.threads).append(self)

    @property
    def tags(self):
        return [t.name for t in self.applied_tags]

    async def edit(self, **kw):
        if self.deleted:
            raise _error(NotFound, 404, 10003, "Unknown Channel")
        if "edit" in self.refuse or ("locked" in kw and "lock" in self.refuse):
            raise _error(Forbidden, 403, 50013, "Missing Permissions")
        if self.archived and kw.get("archived") is not False:
            raise _error(HTTPException, 400, 50083, "Thread is archived")
        if "applied_tags" in kw:
            assert len(kw["applied_tags"]) <= 5
            self.applied_tags = [self.forum.get_tag(t.id) for t in kw["applied_tags"]]
        self.locked = kw.get("locked", self.locked)
        self._archive(kw.get("archived", self.archived))
        self.edits.append(kw)
        return self

    def _archive(self, archived):
        if archived != self.archived:
            (self.forum.archived if self.archived else self.forum.threads).remove(self)
            (self.forum.archived if archived else self.forum.threads).append(self)
            self.archived = archived

    async def send(self, content, allowed_mentions=None):
        if self.deleted:
            raise _error(NotFound, 404, 10003, "Unknown Channel")
        if "send" in self.refuse or self.locked:
            raise _error(Forbidden, 403, 50013, "Missing Permissions")
        self._archive(False)                            # sending opens an archived post
        self.messages.append(Message(self, content, allowed_mentions=allowed_mentions))
        return self.messages[len(self.messages) - 1]

    async def history(self, limit=100):
        for message in list(reversed(self.messages))[:limit]:
            yield message

    def get_partial_message(self, message_id):
        assert message_id == self.id
        return self.starter

    async def fetch_message(self, message_id):
        return self.get_partial_message(message_id)


class Forum:
    type = ChannelType.forum

    def __init__(self, names=("Open", "Closing Soon", "Closed", "Idea", "Cards", "Rules"), emojis=(Emoji(77, "abstain"),)):
        self.id, self.available_tags = FORUM, [Tag(i + 1, n) for i, n in enumerate(names)]
        self.guild = SimpleNamespace(id=1, emojis=list(emojis), fetch_emojis=self._emojis)
        self.threads, self.archived = [], []
        self.flags = SimpleNamespace(require_tag=False)

    async def _emojis(self):
        return list(self.guild.emojis)

    def tag(self, name):
        return next(t for t in self.available_tags if t.name == name)

    def get_tag(self, tag_id):
        return next((t for t in self.available_tags if t.id == tag_id), None)

    async def archived_threads(self, limit=None):
        for thread in list(self.archived):
            yield thread

    async def create_thread(self, name, content, applied_tags=(), allowed_mentions=None):
        thread = Thread(self, T0, name=name, owner=BOT)
        thread.applied_tags, thread.starter.content = list(applied_tags), content
        return SimpleNamespace(thread=thread, message=thread.starter)


class Channel:
    type = ChannelType.text

    def __init__(self, channel_id=RESULTS):
        self.id, self.sent = channel_id, []

    async def send(self, content=None, allowed_mentions=None, **extra):
        self.sent.append(Message(self, content, allowed_mentions=allowed_mentions, **extra))
        return self.sent[len(self.sent) - 1]

    async def history(self, limit=100):
        for message in list(reversed(self.sent))[:limit]:
            yield message


class Client:
    def __init__(self, forum, app_emojis=()):
        self.user, self.forum, self.results, self.app_emojis = SimpleNamespace(id=BOT), forum, Channel(), list(app_emojis)
        self.guilds = [SimpleNamespace(get_role=lambda role_id: object() if role_id == ROLE else None)]
        self.fetched = []

    def get_channel(self, channel_id):              # the gateway's cache: active posts only
        known = {self.forum.id: self.forum, RESULTS: self.results, OTHER: Channel(OTHER)}
        known.update({t.id: t for t in self.forum.threads if not t.deleted})
        return known.get(channel_id)

    async def fetch_channel(self, channel_id):
        self.fetched.append(channel_id)
        found = self.get_channel(channel_id) or next((t for t in self.forum.archived if t.id == channel_id), None)
        if found is None or getattr(found, "deleted", False):
            raise _error(NotFound, 404, 10003, "Unknown Channel")
        return found

    async def fetch_application_emojis(self):
        return list(self.app_emojis)


# ========================
# Fixtures and helpers
# ========================
@pytest.fixture(autouse=True)
def clock(monkeypatch):
    """A fresh module (no file, nothing logged) and a clock the test moves."""
    restart(monkeypatch)
    for name in ("_settings", "_posts", "_logged", "_abstain"):
        monkeypatch.setattr(fv, name, type(getattr(fv, name))())
    monkeypatch.setattr(fv, "STARTER_WAIT", 0)
    now = [T0]
    monkeypatch.setattr(fv, "_now", lambda: now[0])
    return now


def restart(monkeypatch):
    """What a bot restart forgets (the file stays)."""
    for name, value in (("_loaded", False), ("_load_error", None), ("_scanned", None), ("_checked", None),
                        ("_settings", {}), ("_posts", {}), ("_logged", set()), ("_gone", set()), ("_known", {})):
        monkeypatch.setattr(fv, name, value)
    for name in ("_lock", "_load_lock", "_save_lock"):
        monkeypatch.setattr(fv, name, asyncio.Lock())


def saved() -> dict:
    with open(FORUM_JSON, encoding="utf-8") as f:
        return json.load(f)


def contents(thread) -> list:
    return [m.content for m in thread.messages]


def passes(client, *times, now=None):
    """One pass at each time."""
    async def go():
        for when in times:
            now[0] = when
            await fv.run_pass(client)
    asyncio.run(go())


# ========================
# Tests
# ========================
def test_new_post_opened_other_forum_and_off_ignored(clock):
    forum = Forum()
    client = Client(forum)
    post, elsewhere = Thread(forum, T0), Thread(forum, T0)
    elsewhere.parent_id = 5

    async def go():
        await fv.post_created(client, post)
        await fv.post_created(client, elsewhere)
        fv._settings["enabled"] = False
        late = Thread(forum, T0)
        await fv.post_created(client, late)
        await fv.run_pass(client)
        return late
    late = asyncio.run(go())
    assert post.tags == ["Open"] and [str(r.emoji) for r in post.starter.reactions] == ["👍", "👎", "<:abstain:77>"]
    assert all(r.me and r.count == 1 for r in post.starter.reactions)
    assert elsewhere.tags == late.tags == [] and not elsewhere.starter.reactions and not late.starter.reactions
    assert saved()["posts"][str(post.id)]["done"] == ["open tag", "reactions"] and set(saved()["posts"]) == {str(post.id)}


@pytest.mark.parametrize("votes, text, announce", [
    ([("👍", 6, True), ("👎", 2, True), (Emoji(77, "abstain"), 9, True)], "Most votes: 👍 (5).", True),
    ([("👍", 1, True), ("👎", 1, True), (Emoji(77, "Abstain"), 4, True)], "No votes.", False),
    ([("👍", 3, True), ("👎", 5, True)], "Most votes: 👎 (4).", False),
    ([("👍", 3, True), ("👎", 3, True), ("🎉", 1, False)], "Most votes: 👍 and 👎, tied (2 each).", False),
    ([("👍", 3, True), (Emoji(5, "yes"), 2, False), ("👎", 1, True)],
     "Most votes: 👍 and <:yes:5>, tied (2 each).", True),
    ([("👍", 2, True), ("👎🏽", 4, False)], "Most votes: 👎🏽 (4).", False),
    ([("👍", 1, True), (Emoji(5, "yes"), 3, False)], "Most votes: <:yes:5> (3).", True),
])
def test_tally_any_emoji_minus_the_bots_abstain_ignored(votes, text, announce):
    assert fv.outcome(*fv.tally([Reaction(e, n, me) for e, n, me in votes])) == {"text": text, "announce": announce}


def test_winner_text_quotes_the_proposal_within_discords_limit():
    link = "https://discord.com/channels/1/2"
    assert fv.winner_text("Most votes: 👍 (2).", link, "") == f"Most votes: 👍 (2). {link}"
    assert fv.winner_text("Most votes: 👍 (2).", link, "Bigger hands\n\nfor all\n") == \
        f"Most votes: 👍 (2). {link}\n> Bigger hands\n> \n> for all"
    long = fv.winner_text("Most votes: 👍 (2).", link, "word " * 1000)
    assert len(long) <= fv.LIMIT and long.endswith("…") and long.startswith(f"Most votes: 👍 (2). {link}\n> word")


def test_day6_pings_once_and_tags_across_restarts(clock, monkeypatch):
    forum = Forum()
    client = Client(forum)
    post, other = Thread(forum, T0), Thread(forum, T0)
    other.messages.append(Message(other, f"<@&{ROLE}> please vote on this change.", author=OTHER_BOT))
    passes(client, T0, DAY6, DAY6 + timedelta(minutes=5), now=clock)
    assert len(other.messages) == 1 and other.tags == ["Open", "Closing Soon"]   # another bot's ping: none from us
    ping = post.messages[0]
    assert contents(post) == [f"<@&{ROLE}> please vote on this change. Voting closes in 24 hours."]
    mentions = ping.allowed_mentions
    assert [r.id for r in mentions.roles] == [ROLE] and mentions.users is False and mentions.everyone is False
    assert post.tags == ["Open", "Closing Soon"]
    restart(monkeypatch)                                    # the record says pinged
    passes(client, DAY6 + timedelta(minutes=10), now=clock)
    restart(monkeypatch)
    entry = saved()
    del entry["posts"][str(post.id)]                        # the record lost: the post itself says pinged
    with open(FORUM_JSON, "w", encoding="utf-8") as f:
        json.dump(entry, f)
    passes(client, DAY6 + timedelta(minutes=15), now=clock)
    assert len(post.messages) == 1 and post.tags == ["Open", "Closing Soon"]


@pytest.mark.parametrize("votes, note, announced", [
    ([("👍", 4), (Emoji(77, "abstain"), 6)], "Voting is closed. Most votes: 👍 (4).", True),
    ([("👎", 4), ("👍", 3)], "Voting is closed. Most votes: 👎 (4).", False),
])
def test_day7_notes_announces_a_winner_swaps_tags_locks(clock, votes, note, announced):
    forum = Forum()
    client = Client(forum)
    post = Thread(forum, T0, tags=("Idea",))
    passes(client, T0, DAY6, now=clock)
    for emoji, n in votes:
        post.starter.vote(emoji, n)
    passes(client, DAY7, DAY7 + timedelta(minutes=5), DAY7 + timedelta(days=3), now=clock)
    assert contents(post) == [contents(post)[0], note]
    assert post.tags == ["Idea", "Closed"] and post.locked and post.archived
    results = client.results.sent
    if announced:
        top = note.split("Most votes: ")[1]
        assert [m.content for m in results] == [f"Most votes: {top} {post.jump_url}\n> Please change this."]
        assert results[0].allowed_mentions.roles is False
    else:
        assert results == []
    assert saved()["posts"][str(post.id)]["end"] == "closed"


def test_existing_posts_on_enable_overdue_steps_in_order(clock):
    forum = Forum()
    client = Client(forum)
    old = Thread(forum, T0 - timedelta(days=8))
    old.starter.vote("👍", 2)
    sleeping = Thread(forum, T0 - timedelta(days=1), tags=("Cards", "Open"), archived=True)   # reactions: unarchived
    locked = Thread(forum, T0 - timedelta(days=2), archived=True, locked=True)
    closed = Thread(forum, T0 - timedelta(days=2), tags=("Closed",), archived=True)
    held = Thread(forum, T0 - timedelta(days=3), locked=True)
    reacted = Thread(forum, T0 - timedelta(days=1), tags=("Open",), archived=True)   # nothing to add: left archived
    reacted.starter.reactions = [Reaction(e, 1, True) for e in ("👍", "👎", Emoji(77, "abstain"))]
    inter = SimpleNamespace(channel_id=5)

    async def go():
        fv._settings["enabled"] = False
        await fv.run_pass(client)                        # off: nothing
        assert not old.messages and old.tags == []
        await fv.forum_votes(inter, client, enabled=True)
        (args, kwargs), func = utils.command_queue.popleft()
        await func(*args, **kwargs)
    asyncio.run(go())
    assert contents(old) == [f"<@&{ROLE}> please vote on this change. Voting closes now.",
                             "Voting is closed. Most votes: 👍 (2)."]
    assert old.locked and old.tags == ["Closed"] and len(client.results.sent) == 1
    assert sleeping.tags == ["Cards", "Open"] and not sleeping.archived and len(sleeping.starter.reactions) == 3
    assert not any(t.edits or t.starter.reactions or t.messages for t in (locked, closed, held))
    assert reacted.archived and not reacted.edits and len(reacted.starter.reactions) == 3
    assert set(saved()["posts"]) == {str(old.id), str(sleeping.id), str(held.id), str(reacted.id)}   # held waits
    assert saved()["settings"] == {"enabled": True}


def test_missing_tags_and_the_five_tag_limit(clock, capsys):
    forum = Forum(names=("Open", "Closed", "A", "B", "C", "D"), emojis=())
    client = Client(forum)                              # no :abstain: anywhere: 👍 👎 only
    full = Thread(forum, T0, tags=("Open", "A", "B", "C", "D"))
    other = Thread(forum, T0)
    passes(client, T0, DAY6, DAY6 + timedelta(minutes=5), now=clock)
    assert full.tags == ["Open", "A", "B", "C", "D"] and other.tags == ["Open"] and len(other.messages) == 1
    assert [str(r.emoji) for r in other.starter.reactions] == ["👍", "👎"]
    passes(client, DAY7, now=clock)
    assert full.tags == ["A", "B", "C", "D", "Closed"] and other.tags == ["Closed"] and full.locked
    out = capsys.readouterr().out
    assert out.count('has no tag "Closing Soon"') == 1 and out.count("no :abstain: emoji") == 1


def test_abstain_from_the_bots_own_emojis(clock):
    forum = Forum(emojis=(Emoji(3, "party"),))
    client = Client(forum, app_emojis=(Emoji(88, "abstain"),))
    post = Thread(forum, T0)
    passes(client, T0, now=clock)
    assert [str(r.emoji) for r in post.starter.reactions] == ["👍", "👎", "<:abstain:88>"]


def test_five_tags_skip_closing_soon_then_make_room_for_closed(clock, capsys):
    forum = Forum(names=("Open", "Closing Soon", "Closed", "Idea", "Cards", "Rules", "Other"))
    client = Client(forum)
    post = Thread(forum, T0, tags=("Open", "Idea", "Cards", "Rules", "Other"))
    passes(client, T0, DAY6, DAY6 + timedelta(minutes=5), now=clock)
    assert post.tags == ["Open", "Idea", "Cards", "Rules", "Other"] and len(post.messages) == 1
    assert capsys.readouterr().out.count('no room for the tag "Closing Soon"') == 1
    passes(client, DAY7, now=clock)
    assert post.tags == ["Idea", "Cards", "Rules", "Other", "Closed"] and post.locked


def test_deleted_or_closed_by_hand_dropped_a_lock_waits(clock, capsys):
    forum = Forum()
    client = Client(forum)
    gone, shut, closed, paused = Thread(forum, T0), Thread(forum, T0), Thread(forum, T0), Thread(forum, T0)
    passes(client, T0, now=clock)
    gone.deleted = True
    forum.threads.remove(gone)
    shut.locked = paused.locked = True
    shut._archive(True)
    closed.applied_tags = [forum.tag("Closed")]
    passes(client, DAY6, DAY7, now=clock)
    assert not shut.messages and not closed.messages and closed.tags == ["Closed"] and not closed.locked
    assert not paused.messages and paused.tags == ["Open"]                  # a person's lock: its steps wait
    posts = saved()["posts"]
    assert str(gone.id) not in posts and posts[str(shut.id)]["end"] == "closed and locked by hand"
    assert posts[str(closed.id)]["end"] == "closed by hand" and not posts[str(paused.id)]["end"]
    paused.locked = False
    passes(client, DAY7 + timedelta(minutes=5), now=clock)                 # unlocked: overdue steps in order
    assert contents(paused) == [f"<@&{ROLE}> please vote on this change. Voting closes now.",
                                "Voting is closed. No votes."] and paused.locked and paused.tags == ["Closed"]


def test_a_deleted_post_leaves_at_once_never_on_another_error(clock, monkeypatch, capsys):
    forum = Forum()
    client = Client(forum)
    told, away, flaky, later = Thread(forum, T0), Thread(forum, T0), Thread(forum, T0), Thread(forum, T0)
    passes(client, T0, now=clock)
    told.deleted = True                                 # the delete event (the cache still lists it)
    asyncio.run(fv.post_deleted(told.id))
    passes(client, T0 + timedelta(minutes=5), now=clock)
    assert str(told.id) not in saved()["posts"]
    restart(monkeypatch)                                # deleted while the bot was away: the start's check
    forum.threads.remove(told)
    away.deleted = True
    forum.threads.remove(away)
    forum.threads.remove(flaky)                         # unseen by the gateway, and Discord fails to answer
    fetch = client.fetch_channel

    async def fails(channel_id):
        if channel_id == flaky.id:
            raise _error(HTTPException, 503, 0, "Service Unavailable")
        return await fetch(channel_id)
    client.fetch_channel = fails
    passes(client, T0 + timedelta(hours=1), now=clock)
    assert set(saved()["posts"]) == {str(flaky.id), str(later.id)}
    later.deleted = True                                # no event: the next hourly check
    forum.threads.remove(later)
    passes(client, T0 + timedelta(hours=1, minutes=55), now=clock)
    assert str(later.id) in saved()["posts"]
    passes(client, T0 + timedelta(hours=2), now=clock)
    assert set(saved()["posts"]) == {str(flaky.id)}
    out = capsys.readouterr().out
    assert all(out.count(f"post {t.id} was deleted: dropped") == 1 for t in (told, away, later))
    assert out.count(f"post {flaky.id} can't be checked (kept)") == 1


def test_reminders_list_open_posts_read_only(clock, monkeypatch):
    forum = Forum()
    client = Client(forum)
    pinged, fresh, old, shut = (Thread(forum, T0 - timedelta(days=n)) for n in (6.5, 2, 3, 8))   # shut: closed
    passes(client, T0, now=clock)
    fresh.name = "Smaller hands"                        # Discord's name first, then the record's, then the link
    old._archive(True)
    del fv._posts[str(old.id)]["title"]
    del fv._known[str(old.id)]                          # an archived post not seen this session
    when = datetime(2026, 11, 4, 18, tzinfo=timezone.utc).astimezone(rem.TIMEZONE)
    reminder = {"id": "a1b2c3d4", "channel_id": 5, "msg": "Game night", "when": when, "frequency": None,
                "command": None, "enabled": True, "creator_id": PERSON, "problem": None}
    monkeypatch.setattr(rem, "snapshot", lambda guild_id: [dict(reminder)])
    monkeypatch.setattr(rem, "is_admin", lambda interaction: True)
    monkeypatch.setattr(panel, "_now", lambda: T0.astimezone(rem.TIMEZONE))
    asked = SimpleNamespace(guild_id=1, channel_id=fresh.id, user=SimpleNamespace(id=PERSON), client=client)

    async def look():
        snap = await panel._listing(1, client)
        view = panel.ReminderPanel(asked)
        view.start(snap, asked.channel_id)
        shown = view.render(snap)
        options = {o.value: (o.label, o.description) for o in view.children[3].options}
        buttons = {b.custom_id: b.disabled for b in view.children if isinstance(b, Button)}
        return snap, shown["embed"], options, buttons
    snap, embed, options, buttons = asyncio.run(look())
    assert [r["id"] for r in snap] == [f"forum-{pinged.id}", f"forum-{old.id}", "a1b2c3d4", f"forum-{fresh.id}"]
    assert options[f"forum-{pinged.id}"] == ("Proposal forum · Bigger hands", "closes Nov 01 16:00")
    assert options[f"forum-{old.id}"] == (f"Proposal forum · post {old.id}",
                                          "vote ping Nov 04 04:00 · closes Nov 05 04:00")
    assert options[f"forum-{fresh.id}"] == ("Proposal forum · Smaller hands",
                                            "vote ping Nov 05 04:00 · closes Nov 06 04:00")
    assert embed.title == "Proposal forum · Smaller hands"
    assert embed.url == f"https://discord.com/channels/1/{fresh.id}"
    assert [(f.name, f.value.split(" (")[0]) for f in embed.fields] == [
        ("Post", f"<#{fresh.id}>"), ("Vote ping", "2026-11-05 04:00 PST"), ("Closes", "2026-11-06 04:00 PST")]
    assert embed.description == panel.FORUM_NOTE and "6 days" in embed.description
    assert buttons == {"rem:prev": False, "rem:refresh": False, "rem:next": False, "rem:resched": True,
                       "rem:off": True, "rem:delete": True, "rem:new": False}
    fresh.deleted = True
    forum.threads.remove(fresh)
    asyncio.run(fv.post_deleted(fresh.id))
    steps = fv.upcoming()                               # no client (a dry call): the records' titles
    assert [(s["post"], s["title"], s["ping"]) for s in steps] == [
        (pinged.id, "Bigger hands", None), (old.id, None, datetime(2026, 11, 4, 12, tzinfo=timezone.utc))]
    assert asyncio.run(panel._listing(1, client))[1]["id"] == f"forum-{old.id}"
    assert fv.upcoming(client, guild_id=2) == []        # another server
    fv._settings["enabled"] = False
    assert [r["id"] for r in asyncio.run(panel._listing(1, client))] == ["a1b2c3d4"]


def test_listings_leave_out_posts_closed_by_hand_and_the_test_post(clock, monkeypatch, capsys):
    forum = Forum()
    client = Client(forum)
    tagged, shut, paused, kept, away = (Thread(forum, T0) for _ in range(5))
    Thread(forum, T0, name=fv.TEST_TITLE, owner=BOT)    # the CLI's test post: never a proposal
    passes(client, T0, now=clock)
    tagged.applied_tags = [forum.tag("Closed")]         # day 1: closed by people, the steps learn it on day 6
    shut.locked = True
    shut._archive(True)
    paused.locked = True                                # only locked: its steps wait, still listed
    passes(client, T0 + timedelta(days=1), now=clock)
    assert {s["post"]: s["locked"] for s in fv.upcoming(client, 1)} == {paused.id: True, kept.id: False,
                                                                         away.id: False}
    restart(monkeypatch)                                # closed while the bot was away: the start's check reads it
    away.locked = True
    away._archive(True)
    passes(client, T0 + timedelta(days=1, hours=1), now=clock)
    assert {s["post"] for s in fv.upcoming(client, 1)} == {paused.id, kept.id}
    assert "Open posts: 2" in fv.status_text(client)
    kept.locked = True                                  # closed after the start: as last seen, no pass needed
    kept._archive(True)
    assert {s["post"] for s in fv.upcoming(client, 1)} == {paused.id}
    assert not any(entry["end"] for entry in saved()["posts"].values())   # the steps unchanged
    step = next(s for s in fv.upcoming(client, 1) if s["post"] == paused.id)
    card = panel._forum_card(step, DAY6 + timedelta(hours=1))
    assert [(name, value.split(" (")[0]) for name, value in card["fields"]] == [
        ("Post", f"<#{paused.id}>"), ("Vote ping", "Due now"), ("Closes", "2026-11-08 04:00 PST"),
        ("Status", "Locked: waits until unlocked")]
    assert panel._forum_option(step)[1].endswith(" · locked")
    monkeypatch.setattr(rem, "snapshot", lambda guild_id: [])
    monkeypatch.setattr(rem, "is_admin", lambda interaction: True)
    view = panel.ReminderPanel(SimpleNamespace(guild_id=1, user=SimpleNamespace(id=PERSON), client=client))
    view.confirming = True                              # a Delete confirm never lands on a forum step
    view.render(asyncio.run(panel._listing(1, client)))
    assert not view.confirming and "rem:confirm" not in {getattr(b, "custom_id", "") for b in view.children}

    def broken(*args):
        raise KeyError("forum")
    monkeypatch.setattr(fv, "upcoming", broken)
    monkeypatch.setattr(panel, "_forum_failed", False)
    assert asyncio.run(panel._listing(1, client)) == asyncio.run(panel._listing(1, client)) == []
    assert capsys.readouterr().out.count("the proposal forum's steps left out") == 1   # reminders still show


def test_missing_permissions_logged_once_no_retry_loop(clock, capsys):
    forum = Forum()
    client = Client(forum)
    post = Thread(forum, T0)
    post.refuse = {"edit"}
    passes(client, T0, T0 + timedelta(minutes=5), DAY6, DAY7, DAY7 + timedelta(minutes=5), now=clock)
    out = capsys.readouterr().out
    assert post.tags == [] and len(post.starter.reactions) == 3 and len(post.messages) == 2 and not post.locked
    for part in ("open tag", "closing tag", "closed tags", "lock"):
        assert out.count(f"post {post.id}: {part} failed: Forbidden") == 1, part
    assert saved()["posts"][str(post.id)]["end"] == "closed"


def test_failed_winner_message_logged_once_and_the_close_goes_on(clock, capsys):
    forum = Forum()
    client = Client(forum)
    post = Thread(forum, T0)
    passes(client, T0, DAY6, now=clock)
    post.starter.vote("👍", 2)

    async def fail(**kwargs):
        raise _error(HTTPException, 500, 0, "Server Error")
    client.results.send = fail
    passes(client, DAY7, DAY7 + timedelta(minutes=5), now=clock)
    out = capsys.readouterr().out
    assert out.count(f"post {post.id}: winner failed") == 1 and "forward" not in out
    assert post.tags == ["Closed"] and post.locked and post.archived and len(post.messages) == 2


def test_a_failed_listing_or_a_lasting_4xx_never_stalls_a_post(clock, capsys):
    forum = Forum()
    client = Client(forum)
    asleep, full = Thread(forum, T0 - timedelta(hours=1), archived=True), Thread(forum, T0)
    listing, fail = forum.archived_threads, [True]

    async def flaky(limit=None):                        # the archived posts can't be listed at the start
        if fail:
            fail.clear()
            raise _error(HTTPException, 503, 0, "Service Unavailable")
        async for thread in listing(limit=limit):
            yield thread

    async def limit_reached(emoji):                     # 20 reactions already: refused for good
        raise _error(HTTPException, 400, 30010, "Maximum number of reactions (20)")
    forum.archived_threads, full.starter.add_reaction = flaky, limit_reached
    passes(client, T0, T0 + timedelta(minutes=5), DAY6, DAY7, now=clock)
    assert asleep.tags == ["Closed"] and len(asleep.starter.reactions) == 3 and len(asleep.messages) == 2
    assert contents(full) == [f"<@&{ROLE}> please vote on this change. Voting closes in 24 hours.",
                              "Voting is closed. No votes."] and full.tags == ["Closed"] and full.locked
    out = capsys.readouterr().out
    assert out.count("archived posts can't be read") == 1 and out.count(f"post {full.id}: reactions failed") == 1


def test_a_restart_mid_close_never_sends_twice(clock, monkeypatch):
    forum = Forum()
    client = Client(forum)
    post = Thread(forum, T0)
    passes(client, T0, DAY6, now=clock)
    post.starter.vote("👍", 3)
    send, sends = client.results.send, count(1)

    async def restarts_after(*args, **kwargs):          # Discord took it, then a restart before the record
        await send(*args, **kwargs)
        if next(sends) <= 1:
            raise asyncio.CancelledError
    client.results.send = restarts_after
    with pytest.raises(asyncio.CancelledError):
        passes(client, DAY7, now=clock)
    restart(monkeypatch)
    passes(client, DAY7 + timedelta(minutes=10), now=clock)
    assert [m.content for m in client.results.sent] == [f"Most votes: 👍 (3). {post.jump_url}\n> Please change this."]
    assert contents(post)[1:] == ["Voting is closed. Most votes: 👍 (3)."] and post.locked and post.tags == ["Closed"]


def test_settings_persisted_and_checked(clock, monkeypatch):
    forum = Forum()
    client = Client(forum)
    inter = SimpleNamespace(channel_id=5)

    async def go(**options):
        utils.dispatch_queue.clear()
        await fv.forum_votes(inter, client, **options)
        return [kw for _, kw in utils.dispatch_queue]
    sent = asyncio.run(go(enabled=False, role=f"<@&{ROLE}>", results=f"<#{OTHER}>"))
    assert sent[0]["allowed_mentions"].roles is False and "**off**" in sent[0]["content"]
    assert f"Results channel: <#{OTHER}>" in sent[0]["content"]
    restart(monkeypatch)
    assert asyncio.run(fv._ready()) and fv.setting("results") == OTHER and fv.setting("role") == ROLE
    assert fv.setting("enabled") is False and fv.setting("forum") == FORUM
    refused = asyncio.run(go(forum=str(OTHER), role="123456789012345678", results="nope"))
    assert len(refused) == 1 and refused[0]["content"].count("✗") == 3 and fv.setting("results") == OTHER
    assert saved()["settings"] == {"enabled": False, "role": str(ROLE), "results": str(OTHER)}
    with open(FORUM_JSON, "w", encoding="utf-8") as f:
        f.write("{broken")
    restart(monkeypatch)
    assert asyncio.run(go())[0]["content"].startswith("✗ Forum votes are off")
    asyncio.run(fv.run_pass(client))
    assert open(FORUM_JSON, encoding="utf-8").read() == "{broken"   # never overwritten


def test_live_test_cli_flow(clock, monkeypatch, capsys):
    forum = Forum()
    client = Client(forum)

    async def wait(seconds):                             # people vote while the test waits
        assert seconds == 120
        thread = forum.threads[0]
        thread.starter.vote("👍", 3)
        thread.starter.vote(Emoji(77, "abstain"), 5)
    monkeypatch.setattr(fv, "sleep", wait)
    post = asyncio.run(fv.live_test(client, 2))
    thread = post.thread
    assert thread.name == fv.TEST_TITLE and thread.starter.content.endswith("closes in 2 minutes.")
    assert contents(thread) == ["Voting is closed. Most votes: 👍 (3)."] and thread.tags == ["Closed"] and thread.locked
    winner = f"Most votes: 👍 (3). {thread.jump_url}\n> This is a test and closes in 2 minutes."
    assert [m.content for m in client.results.sent] == [winner] and "proposals bot" not in winner
    assert not any("<@&" in (m.content or "") for m in thread.messages + client.results.sent)
    out = capsys.readouterr().out
    assert f"Test post {thread.id}" in out and "Outcome: Most votes: 👍 (3). Winner message: yes" in out
    assert not exists(FORUM_JSON)


def test_test_post_stopped_or_left_open_never_pinged(clock, monkeypatch):
    forum = Forum()
    client = Client(forum)

    async def hang_up(seconds):                          # SIGHUP during the wait
        raise asyncio.CancelledError
    monkeypatch.setattr(fv, "sleep", hang_up)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(fv.live_test(client, 10))
    stopped = forum.archived[0]
    assert contents(stopped) == ["The test stopped early."] and stopped.tags == ["Closed"] and stopped.locked
    left = Thread(forum, T0, name=fv.TEST_TITLE, owner=BOT)  # a killed test: its post left open
    left.starter.vote("👍", 2)

    async def bot_sees():
        await fv.post_created(client, stopped)
        await fv.post_created(client, left)
    asyncio.run(bot_sees())
    passes(client, DAY6, DAY7, now=clock)
    assert contents(stopped) == ["The test stopped early."]
    assert contents(left) == ["Voting is closed. Most votes: 👍 (2)."] and left.locked and left.tags == ["Closed"]
    assert client.results.sent == []
