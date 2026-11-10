"""
The proposal forum's votes (owner 2026-11-09). In the watched forum every post gets the tag Open and the
reactions 👍 👎 :abstain: when it is made (or found open); 6 days later the bot asks the role to vote and adds
Closing Soon; at 7 days it posts the outcome (the reaction with the most votes, :abstain: aside), forwards a won post
to the results channel with a winner message (never when 👎 is among the top, or nobody voted), swaps the tags for
Closed, then locks and archives the post.

An async worker like type_emojis: Discord is awaited on the event loop, the JSON written in a thread. main runs
run_pass every PASS_MINUTES and post_created on on_thread_create; /forum_votes shows or sets the settings. FORUM_JSON
(gitignored, per machine) holds the settings and the parts done per post, so a restart never repeats one; the ping and
the note also look for a bot's message in the post first, the forward and the winner message in the results channel (a
lost file, a restart mid-close). A post a person locks waits until unlocked.

    python -m commands.forum_votes test --minutes N    a test post, closed N minutes later (REST only, no ping)

The bot never pings its own test post and closes one left open without a forward.
"""
from argparse import ArgumentParser
from asyncio import CancelledError, Lock, current_task, get_running_loop, run, sleep, to_thread
from datetime import datetime, timedelta, timezone
from json import dumps, loads
from os import fsync, remove, replace
from os.path import exists
from re import compile as re_compile
from signal import SIGHUP, SIGTERM
from traceback import format_exc
from typing import Optional

from aiohttp import ClientError
from discord import AllowedMentions, ChannelType, Forbidden, HTTPException, Interaction, NotFound, Object
from discord.utils import escape_markdown, snowflake_time

from global_config import FORUM_ID, FORUM_JSON, FORUM_RESULTS_ID, FORUM_ROLE_ID, FORUM_VOTES_ON
from commands.utils import queue_command, queue_message

# ========================
# Configuration
# ========================
PASS_MINUTES = 5                            # main's loop: how often due steps run
PING_AFTER = timedelta(days=6)              # the role is asked to vote
CLOSE_AFTER = timedelta(days=7)             # voting closes
OPEN, SOON, CLOSED = "Open", "Closing Soon", "Closed"   # forum tags, matched by name in any case
UP, DOWN, ABSTAIN = "👍", "👎", "abstain"    # the reactions (:abstain: is a custom emoji, found by name)
MAX_TAGS = 5                                # Discord's limit of tags on a post
HISTORY = 50                                # recent messages searched for a bot's ping, note, forward or winner
STARTER_WAIT = 2                            # s: a new post's first message can lag its thread
UNKNOWN_CHANNEL = 10003                     # Discord's error code: the post was deleted
ARCHIVED = 50083                            # Discord's error code: the post is archived (the next pass opens it)
PING_WORDS = "please vote on this change"
NOTE_WORDS = "Voting is closed."
TEST_TITLE = "Test: proposals bot"
TEST_STOPPED = "The test stopped early."
SCHEMA = "autocroissant.forum-votes"
SKIN_TONES = dict.fromkeys([0xFE0F, *range(0x1F3FB, 0x1F400)])   # 👎🏽 counts as 👎 for the forward rule
MENTION = re_compile(r"<?[#@&!]*(\d{15,22})>?")                 # an ID, <#channel> or <@&role>
NO_MENTIONS = AllowedMentions.none()
DEFAULTS = {"enabled": FORUM_VOTES_ON, "forum": FORUM_ID, "role": FORUM_ROLE_ID, "results": FORUM_RESULTS_ID}
STEPS = {"open": ("open tag", "reactions"),                     # a step's parts, in order (recorded when done)
         "ping": ("ping", "closing tag"),
         "close": ("count", "note", "forward", "winner", "closed tags", "lock")}

# ========================
# Global State
# ========================
_settings: dict = {}            # what /forum_votes set (else DEFAULTS)
_posts: dict = {}               # thread id (str) -> {"forum", "created", "done": [parts], "outcome", "end"}
_loaded = False
_load_error = None              # FORUM_JSON can't be read: votes off this session, the file left untouched
_lock = Lock()                  # one pass at a time (the loop, a new post, a settings change)
_load_lock = Lock()
_save_lock = Lock()             # saves land in order
_scanned = None                 # the forum whose open posts were included since the start (or the last change)
_logged: set = set()            # what was logged once
_abstain: dict = {}             # guild id -> its :abstain: emoji


class Gone(Exception):
    """The post was deleted."""


class Retry(Exception):
    """A part failed in a way that may pass: the step goes on next pass."""


class Post:
    """A post being worked on: its thread and forum, its record (the parts done) and how to save it."""

    def __init__(self, client, forum, thread, entry: dict, save):
        self.client, self.forum, self.thread, self.entry, self.save = client, forum, thread, entry, save
        self.done: list = entry.setdefault("done", [])
        self.sent: dict = {}            # part -> the message it sent (the test prints them)
        self.did: list = []             # what was done, in words


# ========================
# Helper Functions
# ========================
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _log_once(key, text: str) -> None:
    """Print a problem once per key (a post and a part, a forum and a tag): no flood every pass."""
    if key not in _logged:
        _logged.add(key)
        print(f"Forum votes: {text}")


def setting(name: str):
    """A setting: what /forum_votes saved, else global_config's default."""
    value = _settings.get(name, DEFAULTS[name])
    return bool(value) if name == "enabled" else int(value)


def _created(thread) -> datetime:
    return getattr(thread, "created_at", None) or snowflake_time(thread.id)


def _record(thread, forum) -> dict:
    return {"forum": str(forum.id), "created": _created(thread).isoformat(), "done": [], "outcome": None, "end": None}


def _tag_ids(thread) -> list:
    """The post's tag ids (raw: a REST-only client has no parent forum to resolve tags with)."""
    raw = getattr(thread, "_applied_tags", None)
    return [int(i) for i in raw] if raw is not None else [tag.id for tag in thread.applied_tags]


def _tag(forum, name: str):
    """The forum's tag of that name (any case); a missing one is logged once, never made."""
    tag = next((t for t in forum.available_tags if t.name.casefold() == name.casefold()), None)
    if tag is None:
        _log_once((forum.id, name), f"the forum {forum.id} has no tag \"{name}\": skipped")
    return tag


def _has(forum, thread, name: str) -> bool:
    ids = set(_tag_ids(thread))
    return any(t.id in ids and t.name.casefold() == name.casefold() for t in forum.available_tags)


def _passing(e: Exception) -> bool:
    """A failure that may pass (a 5xx, 401, 408, 429, an archived post, the network); any other is for good."""
    if isinstance(e, HTTPException):
        return e.status >= 500 or e.status in (401, 408, 429) or e.code == ARCHIVED
    return isinstance(e, (OSError, ClientError))


def _custom_name(emoji) -> str:
    """A custom emoji's name, lowercased ("" for a plain one)."""
    return (getattr(emoji, "name", "") or "").casefold() if getattr(emoji, "id", None) else ""


def _is_down(emoji: str) -> bool:
    return emoji.translate(SKIN_TONES) == DOWN


def tally(reactions) -> tuple:
    """(the top reactions as text, their votes): a reaction's count minus the bot's own; :abstain: never counts."""
    votes = [(str(r.emoji), r.count - (1 if r.me else 0)) for r in reactions if _custom_name(r.emoji) != ABSTAIN]
    best = max((n for _, n in votes), default=0)
    return ([emoji for emoji, n in votes if n == best] if best > 0 else []), best


def outcome(top: list, votes: int) -> dict:
    """The close's words, and whether the post goes to the results channel (a winner, 👎 not among the top)."""
    if not top:
        return {"text": "No votes.", "forward": False}
    if len(top) == 1:
        text = f"Most votes: {top[0]} ({votes})."
    else:
        text = f"Most votes: {', '.join(top[:len(top) - 1])} and {top[len(top) - 1]}, tied ({votes} each)."
    return {"text": text, "forward": not any(_is_down(emoji) for emoji in top)}


def _outcome(post: Post) -> dict:
    return post.entry.get("outcome") or {"text": "The votes could not be read.", "forward": False}


def _is_test(post: Post) -> bool:
    """The live test's post: the bot's own, by its title."""
    return post.thread.name == TEST_TITLE and getattr(post.thread, "owner_id", None) == post.client.user.id


def _due(entry: dict, now: datetime) -> list:
    """The steps due for a post by its age, in order."""
    created = datetime.fromisoformat(entry["created"])
    return ["open"] + (["ping"] if now >= created + PING_AFTER else []) \
        + (["close"] if now >= created + CLOSE_AFTER else [])


def _finished(entry: dict, step: str) -> bool:
    return STEPS[step][len(STEPS[step]) - 1] in entry["done"]


def _id(text: str) -> Optional[int]:
    found = MENTION.fullmatch(text.strip())
    return int(found.group(1)) if found else None


async def _channel(client, channel_id: int):
    return client.get_channel(channel_id) or await client.fetch_channel(channel_id)


async def _forum(client):
    """The watched forum; None (logged once) when the bot can't see it or it is not a forum."""
    forum_id = setting("forum")
    try:
        forum = await _channel(client, forum_id)
    except HTTPException as e:
        _log_once(("forum", forum_id), f"the forum {forum_id} can't be read: {e}")
        return None
    if getattr(forum, "type", None) != ChannelType.forum:
        _log_once(("forum", forum_id), f"{forum_id} is not a forum: nothing done")
        return None
    return forum


# ========================
# Storage
# ========================
def _check(data) -> tuple:
    """(settings, posts) from FORUM_JSON's data; ValueError when it is not this module's file."""
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("not a forum votes file")
    settings, posts = data.get("settings"), data.get("posts")
    if not isinstance(settings, dict) or not isinstance(posts, dict):
        raise ValueError("no settings or posts")
    for name, value in settings.items():
        if name not in DEFAULTS or (isinstance(value, bool) != (name == "enabled")) \
                or (name != "enabled" and not str(value).isdigit()):
            raise ValueError(f"bad setting {name}")
    for key, entry in posts.items():
        if not key.isdigit() or not isinstance(entry, dict) or not isinstance(entry.get("done"), list):
            raise ValueError(f"bad post {key}")
        datetime.fromisoformat(entry["created"])
    return settings, posts


def _read() -> None:
    """Load FORUM_JSON once; an unreadable one turns the votes off this session and is left untouched."""
    global _loaded, _load_error, _settings, _posts
    if _loaded:
        return
    _loaded = True
    if not exists(FORUM_JSON):
        return
    try:
        with open(FORUM_JSON, encoding="utf-8") as f:
            _settings, _posts = _check(loads(f.read()))
    except Exception as e:
        _load_error = f"{FORUM_JSON} can't be read ({type(e).__name__}: {e}); fix it or move it aside and restart"
        print(f"Forum votes: OFF: {_load_error}")


async def _ready() -> bool:
    """Loaded (in a thread, once); False when the file can't be read."""
    async with _load_lock:
        if not _loaded:
            await to_thread(_read)
    return _load_error is None


def _write(data: bytes) -> None:
    temp = f"{FORUM_JSON}.tmp"
    try:
        with open(temp, "wb") as f:
            f.write(data)
            f.flush()
            fsync(f.fileno())
        replace(temp, FORUM_JSON)
    finally:
        if exists(temp):
            remove(temp)


async def _save() -> None:
    """FORUM_JSON, written in a thread, in order; a failure is logged once (the schedule runs on from memory)."""
    if _load_error:
        return
    async with _save_lock:
        data = dumps({"schema": SCHEMA, "settings": _settings, "posts": _posts}, indent=1, sort_keys=True,
                     ensure_ascii=False) + "\n"
        try:
            await to_thread(_write, data.encode("utf-8"))
        except OSError as e:
            _log_once(("save", str(e)), f"{FORUM_JSON} not saved: {e}")


async def _no_save() -> None:
    """The test's posts are never recorded (the bot's file is the running bot's)."""


# ========================
# Steps
# ========================
async def _part(post: Post, name: str, action, final: bool = False) -> None:
    """
    Run one part of a step unless done. A failure that may pass: logged once and Retry (the next pass goes on), unless
    final (the forward and the winner message: never repeated). Any other (refused, missing, a 4xx such as the reaction
    limit): logged once and recorded as done, so the later parts go on.
    """
    if name in post.done:
        return
    try:
        await action()
    except Exception as e:
        if isinstance(e, NotFound) and e.code == UNKNOWN_CHANNEL and not final:
            raise Gone() from e
        _log_once((post.thread.id, name), f"post {post.thread.id}: {name} failed: {type(e).__name__}: {e}")
        if not final and _passing(e):
            raise Retry() from e
    post.done.append(name)
    await post.save()


async def _retag(post: Post, add: str = "", drop: tuple = ()) -> None:
    """Drop, then add, tags by name; with MAX_TAGS already on, the add is logged and skipped. One edit at most."""
    forum, thread = post.forum, post.thread
    ids = _tag_ids(thread)
    dropped = {t.id for t in forum.available_tags if t.name.casefold() in {n.casefold() for n in drop}}
    new = [i for i in ids if i not in dropped]
    tag = _tag(forum, add) if add else None
    if tag is not None and tag.id not in new:
        if len(new) < MAX_TAGS:
            new.append(tag.id)
        else:
            _log_once((thread.id, add), f"post {thread.id}: no room for the tag \"{add}\" ({MAX_TAGS} tags): skipped")
    if new == ids:
        return
    unarchive = {"archived": False} if thread.archived else {}   # an archived post can't be edited
    post.thread = await thread.edit(applied_tags=[forum.get_tag(i) or Object(id=i) for i in new], **unarchive) or thread
    post.did.append(f"tags: {', '.join(t.name for t in forum.available_tags if t.id in new) or 'none'}")


async def _find_abstain(client, guild):
    """:abstain: from the server's emojis, else the bot's own; missing: logged once (None)."""
    if guild.id in _abstain:
        return _abstain[guild.id]
    emojis = list(getattr(guild, "emojis", ()))
    sources = ([] if emojis else [guild.fetch_emojis]) + [client.fetch_application_emojis]   # REST-only: no cache
    found = next((e for e in emojis if e.name.casefold() == ABSTAIN), None)
    for source in sources:
        if found is not None:
            break
        try:
            found = next((e for e in await source() if e.name.casefold() == ABSTAIN), None)
        except HTTPException as e:
            _log_once(("emojis", source.__name__), f"{source.__name__} failed: {e}")
    if found is None:
        _log_once(("abstain", guild.id), "no :abstain: emoji in the server or the bot's emojis: 👍 and 👎 only")
    else:
        _abstain[guild.id] = found
    return found


async def _react(post: Post) -> None:
    """👍 👎 :abstain: on the post's first message (one the bot already added is a no-op)."""
    abstain = await _find_abstain(post.client, post.thread.guild)
    emojis = [UP, DOWN] + ([abstain] if abstain else [])
    message = post.thread.get_partial_message(post.thread.id)   # a forum post's first message has the post's id
    if post.thread.archived:                    # no reactions in an archived post: opened only if some are missing
        mine = {str(r.emoji) for r in (await post.thread.fetch_message(post.thread.id)).reactions if r.me}
        emojis = [e for e in emojis if str(e) not in mine]
        if not emojis:
            post.did.append("reactions: already there")
            return
        post.thread = await post.thread.edit(archived=False) or post.thread
    try:
        await message.add_reaction(emojis[0])
    except NotFound as e:
        if e.code == UNKNOWN_CHANNEL:
            raise
        await sleep(STARTER_WAIT)
        await message.add_reaction(emojis[0])
    for emoji in emojis[1:]:
        await message.add_reaction(emoji)
    post.did.append(f"reactions: {' '.join(str(e) for e in emojis)}")


async def _sent(post: Post, channel, match, unread=Forbidden) -> bool:
    """
    A bot (this one, or another identity running the votes) already sent a message there that matches: never twice.
    unread: the errors reading the history that leave it to the record.
    """
    me = post.client.user.id
    try:
        async for message in channel.history(limit=HISTORY):
            if (message.author.id == me or getattr(message.author, "bot", False)) and match(message):
                return True
    except unread:
        pass
    return False


async def _said(post: Post, words: str) -> bool:
    return await _sent(post, post.thread, lambda message: words in (message.content or ""))


def _check_mention(post: Post, role_id: int) -> None:
    """Discord drops a ping it may not make without an error: log it once."""
    guild = post.thread.guild
    role = guild.get_role(role_id) if hasattr(guild, "get_role") else None
    me = getattr(guild, "me", None)
    if role is not None and me is not None and not role.mentionable \
            and not post.forum.permissions_for(me).mention_everyone:
        _log_once((post.thread.id, "mention"), f"post {post.thread.id}: the role {role_id} can't be pinged "
                                               "(not mentionable, no Mention All Roles)")


async def _ping(post: Post, role_id: int, close_at: datetime) -> None:
    if _is_test(post):                          # a test post left open: never a ping
        post.did.append("ping: skipped (test post)")
        return
    if await _said(post, PING_WORDS):
        post.did.append("ping: already there")
        return
    _check_mention(post, role_id)
    hours = round((close_at - _now()).total_seconds() / 3600)
    when = f"in {hours} hour{'' if hours == 1 else 's'}" if hours >= 1 else "now"   # plain words: they stay true
    post.sent["ping"] = await post.thread.send(
        f"<@&{role_id}> {PING_WORDS}. Voting closes {when}.",
        allowed_mentions=AllowedMentions(everyone=False, users=False, roles=[Object(id=role_id)]))
    post.did.append("ping")


async def _count(post: Post) -> None:
    """The outcome, kept in the record so a resumed close says the same (unread for good: "could not be read")."""
    reactions = (await post.thread.fetch_message(post.thread.id)).reactions
    post.entry["outcome"] = outcome(*tally(reactions))
    post.did.append(f"outcome: {post.entry['outcome']['text']}")


async def _note(post: Post) -> None:
    if await _said(post, NOTE_WORDS):
        post.did.append("note: already there")
        return
    post.sent["note"] = await post.thread.send(f"{NOTE_WORDS} {_outcome(post)['text']}", allowed_mentions=NO_MENTIONS)
    post.did.append("note")


async def _forward(post: Post, results_id: int) -> None:
    channel = await _channel(post.client, results_id)
    if await _sent(post, channel, lambda message: getattr(message.reference, "message_id", None) == post.thread.id,
                   HTTPException):            # unread: forwarded (a double is rarer than a lost forward)
        post.did.append("forward: already there")
        return
    post.sent["forward"] = await post.thread.get_partial_message(post.thread.id).forward(channel)
    post.did.append(f"forwarded to {results_id}")


async def _announce(post: Post, results_id: int) -> None:
    channel = await _channel(post.client, results_id)
    if await _sent(post, channel, lambda message: post.thread.jump_url in (message.content or "").split(),
                   HTTPException):
        post.did.append("winner message: already there")
        return
    post.sent["winner"] = await channel.send(
        f"**{escape_markdown(post.thread.name)}**: {_outcome(post)['text']} {post.thread.jump_url}",
        allowed_mentions=NO_MENTIONS)
    post.did.append("winner message")


async def _shut(post: Post) -> None:
    thread = post.thread
    if thread.archived:                         # an archived post can't be edited: lock it open, then archive
        thread = await thread.edit(archived=False, locked=True) or thread
    post.thread = await thread.edit(locked=True, archived=True) or thread
    post.did.append("locked and archived")


async def open_post(post: Post) -> None:
    """A new post, or one found open: the tag Open, then the three reactions."""
    await _part(post, "open tag", lambda: _retag(post, add=OPEN))
    await _part(post, "reactions", lambda: _react(post))


async def ping_post(post: Post, role_id: int, close_at: datetime) -> None:
    """Day 6: ask the role to vote, add Closing Soon."""
    await _part(post, "ping", lambda: _ping(post, role_id, close_at))
    await _part(post, "closing tag", lambda: _retag(post, add=SOON))


async def close_post(post: Post, results_id: int, forward: bool = True) -> None:
    """
    Day 7: count, note, forward a won post with its winner message, swap the tags for Closed, lock and archive.
    forward=False: no forward (the schedule closing the bot's own test post).
    """
    await _part(post, "count", lambda: _count(post))
    await _part(post, "note", lambda: _note(post))
    if forward and _outcome(post)["forward"]:
        await _part(post, "forward", lambda: _forward(post, results_id), final=True)
        await _part(post, "winner", lambda: _announce(post, results_id), final=True)
    await _part(post, "closed tags", lambda: _retag(post, add=CLOSED, drop=(OPEN, SOON)))
    await _part(post, "lock", lambda: _shut(post))


# ========================
# Scheduler
# ========================
def _include(forum, thread) -> bool:
    """Record a post not seen yet if it is open (no Closed; archived and locked is closed). True if recorded."""
    key = str(thread.id)
    if key in _posts or getattr(thread, "parent_id", None) != forum.id or (thread.locked and thread.archived) \
            or _has(forum, thread, CLOSED):
        return False
    _posts[key] = _record(thread, forum)
    return True


def _held(forum, thread, entry: dict) -> str:
    """Before the bot's note: "wait" (locked by a person, still open) or why a person closed it ("": go on)."""
    if "note" in entry["done"]:                 # the bot's own close under way
        return ""
    closed = _has(forum, thread, CLOSED)
    if thread.locked and (thread.archived or closed):
        return "closed and locked by hand"
    if closed:
        return "closed by hand"
    return "wait" if thread.locked else ""


async def _advance(client, forum, key: str, entry: dict, thread=None) -> None:
    """Run a post's due steps in order (open, the day-6 ping, the day-7 close); the thread is read only when one is."""
    due = [step for step in _due(entry, _now()) if not _finished(entry, step)]
    if not due:
        return
    try:
        if thread is None:
            thread = await _channel(client, int(key))
        held = _held(forum, thread, entry)
        if held == "wait":
            _log_once((thread.id, "held"), f"post {thread.id} is locked: its steps wait until it is unlocked")
            return
        if held:
            entry["end"] = held
            print(f"Forum votes: post {thread.id} {held}: dropped")
            await _save()
            return
        post = Post(client, forum, thread, entry, _save)
        await open_post(post)
        if "ping" in due:
            await ping_post(post, setting("role"), datetime.fromisoformat(entry["created"]) + CLOSE_AFTER)
        if "close" in due:
            await close_post(post, setting("results"), forward=not _is_test(post))
            entry["end"] = "closed"
            await _save()
    except (Gone, NotFound):
        _posts.pop(key, None)
        print(f"Forum votes: post {key} was deleted: dropped")
        await _save()
    except Retry:
        pass
    except HTTPException as e:
        _log_once((int(key), "read"), f"post {key} can't be read: {e}")


async def _open_posts(forum) -> tuple:
    """
    (the forum's posts not locked, all read): the active ones (the gateway's) and the archived ones (asked of Discord;
    a failure is logged once and the listing asked again next pass).
    """
    found, whole = list(forum.threads), True
    try:
        found += [t async for t in forum.archived_threads(limit=None) if not t.locked]
    except (HTTPException, OSError, ClientError) as e:
        _log_once(("archived", forum.id), f"the forum {forum.id}'s archived posts can't be read (again next pass): {e}")
        whole = False
    return list({t.id: t for t in found}.values()), whole


async def _pass(client) -> None:
    global _scanned
    if not await _ready() or not setting("enabled"):
        return
    forum = await _forum(client)
    if forum is None:
        return
    if _scanned != forum.id:                    # the start, turned on, or another forum: every open post
        found, whole = await _open_posts(forum)
        _scanned = forum.id if whole else None
    else:
        found = list(forum.threads)             # a post made while the bot was away
    threads = {str(t.id): t for t in sorted(found, key=_created) if _include(forum, t)}
    if threads:
        await _save()
    for key, entry in sorted(_posts.items(), key=lambda item: item[1]["created"]):
        if entry["forum"] == str(forum.id) and not entry.get("end"):
            await _advance(client, forum, key, entry, threads.get(key))


# ========================
# Public API Functions
# ========================
async def run_pass(client) -> None:
    """main's loop (every PASS_MINUTES): include posts not seen yet, run every due step. Never raises."""
    async with _lock:
        try:
            await _pass(client)
        except Exception as e:
            _log_once(("pass", repr(e)), f"a pass failed:\n{format_exc()}")


async def post_created(client, thread) -> None:
    """on_thread_create: a new post in the watched forum is opened at once."""
    try:
        if getattr(thread, "parent_id", None) is None or not await _ready() or not setting("enabled") \
                or thread.parent_id != setting("forum"):
            return
        async with _lock:
            forum = await _forum(client)
            if forum is None or not _include(forum, thread):
                return
            await _save()
            await _advance(client, forum, str(thread.id), _posts[str(thread.id)], thread)
    except Exception as e:
        _log_once(("new post", repr(e)), f"a new post failed:\n{format_exc()}")


def status_text() -> str:
    """The settings and the open posts (/forum_votes)."""
    forum = setting("forum")
    open_posts = [e for e in _posts.values() if e["forum"] == str(forum) and not e.get("end")]
    lines = [f"Proposal votes: **{'on' if setting('enabled') else 'off'}**",
             f"Forum: <#{forum}>",
             f"Role asked to vote on day 6: <@&{setting('role')}>",
             f"Results channel: <#{setting('results')}>",
             f"Open posts: {len(open_posts)}"]
    steps = [datetime.fromisoformat(e["created"]) + (CLOSE_AFTER if _finished(e, "ping") else PING_AFTER)
             for e in open_posts]
    if steps:
        lines.append(f"Next step: <t:{int(min(steps).timestamp())}:R>")
    return "\n".join(lines)


async def forum_votes(interaction: Interaction, client, enabled: Optional[bool] = None, forum: str = "",
                      role: str = "", results: str = "") -> None:
    """
    /forum_votes (admin: main checks): no options show the settings; options change them. Turning the votes on, or
    another forum, includes its open posts at once.

    Args:
        enabled: on or off (None: unchanged)
        forum, role, results: an ID or a mention ("": unchanged)
    """
    global _scanned
    if not await _ready():
        queue_message(interaction, f"✗ Forum votes are off: {_load_error}.")
        return
    changes, problems = {}, []
    if forum:
        channel = client.get_channel(_id(forum) or 0)
        if getattr(channel, "type", None) != ChannelType.forum:
            problems.append(f"✗ `{forum}` is not a forum the bot can see.")
        else:
            changes["forum"] = str(channel.id)
    if results:
        channel = client.get_channel(_id(results) or 0)
        if channel is None or getattr(channel, "type", None) == ChannelType.forum or not hasattr(channel, "send"):
            problems.append(f"✗ `{results}` is not a channel the bot can post in.")
        else:
            changes["results"] = str(channel.id)
    if role:
        role_id = _id(role)
        if role_id is None or not any(guild.get_role(role_id) for guild in client.guilds):
            problems.append(f"✗ `{role}` is not a role in the bot's servers.")
        else:
            changes["role"] = str(role_id)
    if problems:
        queue_message(interaction, "\n".join(problems), allowed_mentions=NO_MENTIONS)
        return
    if enabled is not None:
        changes["enabled"] = bool(enabled)
    if changes:
        moved = int(changes.get("forum", setting("forum"))) != setting("forum")
        rescan = moved or (changes.get("enabled") and not setting("enabled"))
        _settings.update(changes)
        await _save()
        if rescan and setting("enabled"):
            _scanned = None
            queue_command(run_pass, client)
    queue_message(interaction, status_text(), allowed_mentions=NO_MENTIONS)


async def _stopped(post: Post) -> None:
    """The test stopped before its close: its post gets a short note, Closed, a lock and archived (each tried)."""
    if "lock" in post.done:
        return

    async def fresh():
        post.thread = await post.client.fetch_channel(post.thread.id)
    failed = []
    for name, action in (("read", fresh), ("note", lambda: post.thread.send(TEST_STOPPED, allowed_mentions=NO_MENTIONS)),
                         ("closed tags", lambda: _retag(post, add=CLOSED, drop=(OPEN, SOON))), ("lock", lambda: _shut(post))):
        try:
            await action()
        except Exception as e:
            failed.append(f"{name}: {type(e).__name__}: {e}")
    try:
        print(f"Test post {post.thread.id} stopped early: " + ("; ".join(failed) or "closed, locked and archived"))
    except OSError:
        pass                                    # the terminal is gone (a dropped ssh)


async def live_test(client, minutes: float) -> Post:
    """
    The live test: a post in the watched forum, opened, then closed `minutes` later with the bot's own steps (no
    ping, nothing recorded). The running bot opens it too (no harm: the steps are idempotent; locked, it is dropped).
    Stopped early (an error, Ctrl-C, SIGHUP or SIGTERM), the post is closed at once.
    """
    await to_thread(_read)
    forum = await _forum(client)
    if forum is None:
        raise SystemExit(f"Forum votes: the forum {setting('forum')} can't be used (see above).")
    tags = [t for t in [_tag(forum, OPEN)] if t] if getattr(getattr(forum, "flags", None), "require_tag", False) else []
    span = f"{minutes:g} minute{'' if minutes == 1 else 's'}"
    made = await forum.create_thread(name=TEST_TITLE, applied_tags=tags, allowed_mentions=NO_MENTIONS,
                                     content=f"This is a test of the proposals bot and closes in {span}.")
    post = Post(client, forum, made.thread, _record(made.thread, forum), _no_save)
    try:
        print(f"Test post {made.thread.id} (first message {made.message.id}): {made.thread.jump_url}")
        await open_post(post)
        print(f"Opened: {'; '.join(post.did) or 'nothing to do'}")
        print(f"Closing in {span}...")
        await sleep(minutes * 60)
        post.did = []
        post.thread = await client.fetch_channel(made.thread.id)
        await close_post(post, setting("results"))
    except BaseException:
        await _stopped(post)
        raise
    print(f"Closed: {'; '.join(post.did)}")
    print(f"Outcome: {_outcome(post)['text']} Forwarded: {'yes' if _outcome(post)['forward'] else 'no'}")
    for part, message in post.sent.items():
        print(f"  {part}: message {message.id} in {message.channel.id}")
    return post


async def _cli_test(minutes: float) -> None:
    from discord import Client, Intents
    from config import TOKEN
    loop, task = get_running_loop(), current_task()
    for signal in (SIGHUP, SIGTERM):            # a dropped ssh or a SIGTERM: the test post is still closed
        loop.add_signal_handler(signal, task.cancel)
    async with Client(intents=Intents.none()) as client:
        await client.login(TOKEN)               # REST only: no gateway, so the running bot is not disturbed
        try:
            await live_test(client, minutes)
        except (CancelledError, Gone, Retry):
            raise SystemExit("Forum votes: the test stopped (see above).")


def main(args: Optional[list] = None) -> int:
    """The command line: `test --minutes N` (the jetson, in the bot's checkout and venv)."""
    parser = ArgumentParser(prog="python -m commands.forum_votes", description="The proposal forum's votes.")
    steps = parser.add_subparsers(dest="command", required=True)
    test = steps.add_parser("test", help="a test post in the watched forum, opened, then closed N minutes later")
    test.add_argument("--minutes", type=float, default=5.0, help="minutes before the close (default 5)")
    options = parser.parse_args(args)
    if options.minutes < 0:
        parser.error("--minutes must be 0 or more")
    run(_cli_test(options.minutes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'close_post',
    'forum_votes',
    'live_test',
    'open_post',
    'outcome',
    'ping_post',
    'post_created',
    'run_pass',
    'status_text',
    'tally',
    'main',
]
