from collections import defaultdict
from datetime import date, datetime, timedelta
from discord import AllowedMentions, Interaction
from discord.app_commands import Choice
from inspect import Parameter, isawaitable, signature
from json import dumps, loads
from os import O_RDONLY, close as os_close, fsync, open as os_open, remove, replace
from os.path import dirname, exists
from pickle import load, dump
from re import compile as re_compile
from shlex import quote as shlex_quote, split as shlex_split
from threading import Lock, RLock
from traceback import format_exc
from types import UnionType
from typing import Any, Literal, Optional, Union, get_args, get_origin, get_type_hints
from uuid import uuid4

from commands.utils import (
    slash_registry,
    queue_message,
    queue_command,
    make_fake_interaction,
    lacks_admin,
    queue_long_message,
)
from global_config import REMIND_JSON, REMIND_PKL, TIMEZONE

# ========================
# Bot adapter (the only intended differences from OJWikibot's copy of this module)
# ========================
STORE_PATH = REMIND_JSON                    # reminder.json: gitignored, per machine (as reminder.pkl was)
LEGACY_PATH = REMIND_PKL                    # converted once to STORE_PATH, then kept as reminder.pkl.bak
STORE_SCHEMA = "autocroissant.reminder-store"
TIME_SEMANTICS = "when: Pacific wall time with that instant's offset (America/Los_Angeles); repeats keep the wall time"
ZONE_LABEL = "Pacific time"
RUN_AS_CREATOR = True                       # a reminder's command runs with its creator's permissions
LEGACY_RUNS_AS_BOT = True                   # reminders saved before creator tracking keep running as the bot, as before
ALLOW_POSITIONAL = True                     # commands may give values by position before name:value options
ON_LOAD_ERROR = "off"                       # an unreadable store: reminders off, the bot keeps running ("raise": exit)
CHANNEL_RETRY = timedelta(hours=1)          # a one-shot whose channel is gone waits this long and tries again
# Options a command no longer has. A stored reminder that still sets one drops it instead of failing to run.
# Each entry: (removed options, the command's option order before they were removed, because pre-v2 reminders
# could give options by position).
RETIRED_OPTIONS = {
    # local mode was archived on 2026-10-20 (branch archive/local-update-stats); /update_stats always reads GitHub
    "update_stats": (("use_local_repo", "use_local_timestamp"),
                     ("output_problematic_cards", "use_local_repo", "use_local_timestamp", "force_update")),
}


def is_admin(interaction) -> bool:
    return not lacks_admin(interaction)


# ========================
# Configuration
# ========================
# A reminder dict in memory: id, channel_id (int), msg, when (aware datetime in TIMEZONE), frequency (timedelta or
# None), command (None or canonical "/name key:value ..."), enabled (False: kept but never fires), creator_id (None =
# saved before creator tracking), creator_name, v (REMINDER_VERSION once migrated), problem (why the scheduler skips
# it, or None). reminder.json holds the same fields (SCHEMA_VERSION; ReminderStore).
SCHEMA_VERSION = 2
KEYS = ("id", "channel_id", "msg", "when", "frequency", "command", "enabled", "creator_id", "creator_name", "v", "problem")
REMINDER_VERSION = 3                        # 3: retired options dropped from stored commands (RETIRED_OPTIONS)
MAX_INTERVAL = timedelta(days=366)          # longer offsets/frequencies are typos, and huge ones overflow datetime
NOT_SCHEDULABLE = {"reminders"}             # the panel needs a real user
PANEL_TIMEOUT = 600                         # an idle /reminders panel closes (interaction tokens last 15 minutes)
SELECT_SIZE = 25                            # options in a Discord select
INTERVAL_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
TIME_FORMATS = ("%H:%M", "%I%p", "%I:%M%p")
TRUE_WORDS = {"true", "yes", "y", "on", "1"}
FALSE_WORDS = {"false", "no", "n", "off", "0"}

DATE_PREFIX = re_compile(r"(\d{4})-(\d{1,2})-(\d{1,2})[\st,]*(.*)")
WORD_PREFIX = re_compile(r"([a-z]+)\.?[\s,]*(.*)")
INTERVAL_FULL = re_compile(r"(?:\s*\d+\s*[smhdw])+\s*")
INTERVAL_PART = re_compile(r"(\d+)\s*([smhdw])")
OPTION_NAME = re_compile(r"[A-Za-z_][A-Za-z0-9_]*")
SNOWFLAKE = re_compile(r"[1-9][0-9]*")

# ========================
# Global State
# ========================
reminders = defaultdict(list)
_client = None              # discord.Client, set by bind_reminder_client; used to spot deleted channels
_loaded = False
_load_error = None          # why reminders are off (the store can't be read): the file is left untouched
_backend = "json"           # "pkl" while reminder.pkl could not be converted (this session saves to it)
_conversion_error = None    # why the conversion failed (status())
_lock = RLock()             # guards reminders and the file: commands and the panel run in threads next to check_reminder
_check_lock = Lock()        # single-flight: a pass is queued every second, and two overlapping passes would double-fire


class _ReminderUser:
    """Stands in for the reminder's creator, so lacks_admin sees who scheduled the command."""
    def __init__(self, user_id, name):
        self.id = user_id
        self.name = name
        self.display_name = name
        self.global_name = name
        self.mention = f"<@{user_id}>"
        self.bot = False

    def __str__(self):
        return self.name


class ReminderError(Exception):
    """A refusal; its text is the reply."""


class ReminderOff(ReminderError):
    """Reminders are off this session (the store can't be read)."""


class StoreError(Exception):
    """reminder.json can't be read or written as a reminder store."""


# ========================
# Time Parsing
# ========================
def _parse_clock(text):
    """'13:00', '1PM', '1:30 pm' -> time, or None."""
    text = text.strip().upper().replace(" ", "")
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    return None


def _weekday_index(word):
    """'sun', 'tues', 'thursday' -> 0-6 (Monday = 0), or None."""
    if len(word) < 3:
        return None
    for index, name in enumerate(WEEKDAYS):
        if name.startswith(word):
            return index
    return None


def parse_when(when_str, now):
    """
    Parse a reminder start time in TIMEZONE.

    Args:
        when_str: "13:00" / "1PM" / "1:30PM" (today), optionally after "today", "tomorrow", a weekday
                  ("sun 3AM": the next one, never in the past) or a date ("2026-12-25 9AM")
        now: Current time in TIMEZONE

    Returns:
        (datetime, fixed_day) where fixed_day is True when the text named a date, or (None, False) if unparseable
    """
    text = (when_str or "").strip().lower()
    day = now.date()
    weekday = None
    fixed_day = False

    match = DATE_PREFIX.fullmatch(text)
    if match:
        try:
            day = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None, False
        text = match.group(4)
        fixed_day = True
    else:
        match = WORD_PREFIX.fullmatch(text)
        if match:
            word = match.group(1)
            if word in ("tomorrow", "tmrw"):
                day = day + timedelta(days=1)
            elif word != "today":
                weekday = _weekday_index(word)
                if weekday is None:
                    return None, False
            text = match.group(2)

    clock = _parse_clock(text)
    if clock is None:
        return None, False
    if weekday is not None:
        day = day + timedelta(days=(weekday - day.weekday() + 7) % 7)
        if datetime.combine(day, clock, tzinfo=TIMEZONE) <= now:
            day = day + timedelta(days=7)
    return datetime.combine(day, clock, tzinfo=TIMEZONE), fixed_day


def parse_interval(time_str: Optional[str]) -> Optional[timedelta]:
    """
    Parse an interval like "5s", "10m", "2h", "1d", "3w" or a combination such as "1d12h".

    Args:
        time_str: Interval string (s=seconds, m=minutes, h=hours, d=days, w=weeks)

    Returns:
        timedelta, or None if blank or unparseable
    """
    if not time_str:
        return None
    text = time_str.strip().lower()
    if not INTERVAL_FULL.fullmatch(text):
        return None
    seconds = 0
    for amount, unit in INTERVAL_PART.findall(text):
        seconds = seconds + int(amount) * INTERVAL_UNITS[unit]
    try:
        return timedelta(seconds=seconds)
    except OverflowError:
        return None


def checked_interval(label, raw):
    """parse_interval that raises ValueError with a user-facing reason; blank -> None."""
    if not raw or not raw.strip():
        return None
    interval = parse_interval(raw)
    if interval is None:
        raise ValueError(f"invalid {label} `{raw}`. Use a number plus s/m/h/d/w, e.g. `30m`, `1d`, `1w` or `1d12h`")
    if interval <= timedelta(0):
        raise ValueError(f"{label} must be longer than zero")
    if interval > MAX_INTERVAL:
        raise ValueError(f"{label} `{raw}` is longer than the {MAX_INTERVAL.days}-day maximum")
    return interval


def format_interval(interval):
    """timedelta -> '1w', '1d12h', '30m'."""
    seconds = int(interval.total_seconds())
    parts = []
    for unit, size in (("w", 604800), ("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        if seconds >= size:
            parts.append(f"{seconds // size}{unit}")
            seconds = seconds - (seconds // size) * size
    return "".join(parts) or "0s"


def format_when(when):
    """Pacific wall time plus a Discord relative timestamp that each reader sees in their own time zone."""
    return f"{when.strftime('%Y-%m-%d %H:%M %Z')} (<t:{int(when.timestamp())}:R>)"


def next_run(when, frequency, now):
    """First when + n*frequency after now, in one step (a stalled bot must not replay every missed run)."""
    return when + frequency * ((now - when) // frequency + 1)


def parse_schedule(when, offset, frequency, now):
    """
    The first run and repeat of a new or rescheduled reminder.

    Args:
        when: parse_when text; blank means now (only together with an offset)
        offset: Delay added to the first run ("30m", "1d")
        frequency: Repeat interval, blank = once
        now: Current time in TIMEZONE

    Returns:
        (first run, interval or None, [notes]); raises ValueError with a user-facing reason
    """
    notes = []
    offset_delta = checked_interval("offset", offset)
    interval = checked_interval("frequency", frequency)
    if when and when.strip():
        remind_at, fixed_day = parse_when(when, now)
        if remind_at is None:
            raise ValueError(f"invalid time `{when}`. Try `13:00`, `1PM`, `sun 3AM`, `tomorrow 9AM` "
                             f"or `2026-12-25 9AM` ({ZONE_LABEL})")
    elif offset_delta:
        remind_at, fixed_day = now.replace(microsecond=0), False
    else:
        raise ValueError("give a time with `when` (e.g. `13:00`, `1PM`, `sun 3AM`) or a delay with `offset`")
    if offset_delta:
        remind_at = remind_at + offset_delta
    if remind_at - now > MAX_INTERVAL:
        raise ValueError(f"{remind_at.strftime('%Y-%m-%d')} is more than {MAX_INTERVAL.days} days away")
    if remind_at <= now:
        if fixed_day:
            raise ValueError(f"{remind_at.strftime('%Y-%m-%d %H:%M %Z')} is in the past")
        if when and when.strip():
            notes.append("That time already passed today, so it fires right away "
                         "(use `tomorrow ...` or an offset of `1d` to start tomorrow).")
    return remind_at, interval, notes


# ========================
# Command Parsing
# ========================
def tokenize(command):
    """
    Split a stored command into its name, positional values and name:value options.

    Args:
        command: e.g. '/update_stats force_update:True' or '/query query:"The Freezer"'

    Returns:
        (name, [positional values], {option: value}); raises ValueError on bad quoting or ordering
    """
    try:
        parts = shlex_split(command)
    except ValueError as e:
        raise ValueError(f"can't parse the command ({e}); check the quotes") from None
    if not parts:
        raise ValueError("the command is empty")
    name = parts[0].lstrip("/").strip().lower()
    if not name:
        raise ValueError("the command has no name")

    args = []
    kwargs = {}
    i = 1
    while i < len(parts):
        token = parts[i]
        key, sep, value = token.partition(":")
        if sep and OPTION_NAME.fullmatch(key):
            # "key: value" typed with a space arrives as two tokens
            if value == "" and i + 1 < len(parts):
                next_key, next_sep, _ = parts[i + 1].partition(":")
                if not (next_sep and OPTION_NAME.fullmatch(next_key)):
                    value = parts[i + 1]
                    i += 1
            if key in kwargs:
                raise ValueError(f"`{key}` is given twice")
            kwargs[key] = value
        else:
            if kwargs:
                raise ValueError("a value with no `name:` comes after a name:value option")
            args.append(token)
        i += 1
    return name, args, kwargs


def _command_params(func):
    """[(name, annotation, default)] for a slash callback's options, i.e. everything after interaction."""
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    params = []
    for param in list(signature(func).parameters.values())[1:]:
        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            continue
        params.append((param.name, hints.get(param.name, param.annotation), param.default))
    return params


def _type_name(annotation):
    return getattr(annotation, "__name__", None) or str(annotation)


def _to_bool(raw, option, legacy):
    word = raw.strip().lower()
    if legacy and word == "true" and raw.strip() != "true":
        # The old parser turned every "True" except lowercase "true" into False; keep what actually ran.
        return False
    if word in TRUE_WORDS:
        return True
    if word in FALSE_WORDS:
        return False
    raise ValueError(f"`{option}` must be True or False")


def _convert(raw, annotation, option, legacy):
    """Turn a text value into what the slash callback's annotation expects; ValueError (no values) if it can't."""
    if annotation is Parameter.empty or annotation is Any or annotation is str:
        return raw
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        choices = [arg for arg in get_args(annotation) if arg is not type(None)]
        problem = f"`{option}` can't take that value"
        for choice in choices:
            try:
                return _convert(raw, choice, option, legacy)
            except ValueError as e:
                problem = str(e)
        raise ValueError(problem)
    if origin is Literal:
        for allowed in get_args(annotation):
            if str(allowed) == raw:
                return allowed
        raise ValueError(f"`{option}` must be one of {', '.join(str(a) for a in get_args(annotation))}")
    if annotation is Choice or origin is Choice:
        inner = get_args(annotation)
        value = _convert(raw, inner[0], option, legacy) if inner else raw
        return Choice(name=str(value), value=value)
    if annotation is bool:
        return _to_bool(raw, option, legacy)
    if annotation is int:
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"`{option}` must be a whole number") from None
    if annotation is float:
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"`{option}` must be a number") from None
    raise ValueError(f"`{option}` ({_type_name(annotation)}) can't be filled in by a reminder")


def _bind(func, args, kwargs, legacy=False):
    """
    Match a command's values to its slash callback's options and convert them.

    Args:
        func: Slash callback from slash_registry
        args: Positional values, in option order
        kwargs: name:value options
        legacy: Read booleans the way the pre-v2 parser did

    Returns:
        {option: converted value}, in option order, only for options that were given
    """
    params = _command_params(func)
    names = [param[0] for param in params]
    if len(args) > len(names):
        raise ValueError(f"too many values; its options are {', '.join(names) or 'none'}")
    raw = {}
    for i in range(len(args)):
        raw[names[i]] = args[i]
    for key, value in kwargs.items():
        if key not in names:
            raise ValueError(f"`{key}` is not one of its options ({', '.join(names) or 'none'})")
        if key in raw:
            raise ValueError(f"`{key}` is given twice")
        raw[key] = value

    values = {}
    for option, annotation, default in params:
        if option in raw:
            values[option] = _convert(raw[option], annotation, option, legacy)
        elif default is Parameter.empty:
            raise ValueError(f"`{option}` is required")
    return values


def render(name, values):
    """Canonical stored form: '/name option:value ...', quoted where needed."""
    parts = [f"/{name}"]
    for option, value in values.items():
        if isinstance(value, Choice):
            value = value.value
        parts.append(f"{option}:{shlex_quote(str(value))}")
    return " ".join(parts)


def _drop_retired(name, args, kwargs, legacy):
    """
    Remove options the command no longer has (RETIRED_OPTIONS), so an old reminder keeps running.

    Args:
        legacy: Pre-v2 reminder, whose positional values follow the command's option order of that time

    Returns:
        (args, kwargs, ["option:value" that were dropped]); raises ValueError if an option is given twice
    """
    retired = RETIRED_OPTIONS.get(name)
    if not retired:
        return args, kwargs, []
    removed, old_order = retired
    if legacy and args and len(args) <= len(old_order):
        named = {}
        for i in range(len(args)):
            named[old_order[i]] = args[i]
        for key, value in kwargs.items():
            if key in named:
                raise ValueError(f"`{key}` is given twice")
            named[key] = value
        args, kwargs = [], named
    dropped = []
    for option in removed:
        if option in kwargs:
            dropped.append(f"{option}:{kwargs.pop(option)}")
    return args, kwargs, dropped


def resolve_command(command, legacy=False):
    """
    Look up and bind a stored command.

    Returns:
        (callback, name, {option: value}, ["option:value" dropped as retired]); raises ValueError with a
        user-facing reason that names options but never their values (it is posted in the channel)
    """
    name, args, kwargs = tokenize(command)
    if name in NOT_SCHEDULABLE:
        raise ValueError(f"`/{name}` can't run from a reminder")
    func = slash_registry.get(name)
    if func is None:
        raise ValueError(f"`/{name}` is not a bot command")
    args, kwargs, dropped = _drop_retired(name, args, kwargs, legacy)
    if args and not ALLOW_POSITIONAL and not legacy:
        raise ValueError("give every option as name:value")
    return func, name, _bind(func, args, kwargs, legacy), dropped


def _store_command(command):
    """A new command checked and in canonical form: (stored text or None, [notes]); ValueError if it can't run."""
    if not command or not command.strip():
        return None, []
    if not slash_registry:                  # before on_ready: checked when it fires
        return command.strip(), []
    _, name, values, dropped = resolve_command(command.strip())
    notes = []
    if dropped:
        notes.append(f"Dropped `{' '.join(dropped)}`: /{name} no longer has "
                     f"{'that option' if len(dropped) == 1 else 'those options'}.")
    return render(name, values), notes


# ========================
# Panel Helpers (plain functions; the View is commands/reminder_panel.py)
# ========================
def can_edit(reminder, user_id, admin) -> bool:
    """Admins, or the reminder's creator (a reminder with no creator: admins only)."""
    creator_id = reminder.get("creator_id")
    return bool(admin) or (creator_id is not None and creator_id == user_id)


def locate(ids, current_id, last_index) -> int:
    """Index of current_id in ids, else last_index clamped (the neighbour of a reminder that is gone); -1 if empty."""
    if not ids:
        return -1
    if current_id in ids:
        return ids.index(current_id)
    return min(max(last_index, 0), len(ids) - 1)


def step(n, index, delta) -> int:
    """index moved by delta, wrapping around n items; -1 if there are none."""
    if n <= 0:
        return -1
    return (index + delta) % n


def select_page(n, index, size=SELECT_SIZE):
    """
    The window of the jump select that holds index.

    Returns:
        (start, stop, has_prev, has_next): up to size reminders on one page, else size - 2 per page plus
        'earlier'/'later' entries
    """
    if n <= size:
        return 0, n, False, False
    per_page = size - 2
    index = min(max(index, 0), n - 1)
    start = (index // per_page) * per_page
    stop = min(n, start + per_page)
    return start, stop, start > 0, stop < n


def _clip(text, limit):
    text = str(text)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _state(reminder):
    if reminder.get("problem"):
        return "problem"
    return "on" if reminder.get("enabled", True) else "off"


def card(reminder, now):
    """
    The texts of a reminder's panel card.

    Returns:
        {"title", "description", "fields": [(name, value)], "state": "on"/"off"/"problem"}
    """
    when = reminder.get("when")
    frequency = reminder.get("frequency")
    state = _state(reminder)
    if isinstance(when, datetime):
        next_text = format_when(when)
        if state == "off":
            next_text = f"Off (next due {next_text})"
        elif when <= now:
            next_text = f"Due now ({next_text})"
    else:
        next_text = "unreadable"
    command = reminder.get("command")
    if command:
        fence = "``" if "`" in command else "`"
        command_text = f"{fence}{_clip(command, 1000)}{fence}"
    else:
        command_text = "none"
    creator_id = reminder.get("creator_id")
    if creator_id is not None:
        creator_text = f"<@{creator_id}>"
    elif RUN_AS_CREATOR and not LEGACY_RUNS_AS_BOT:
        creator_text = "unknown"
    else:
        creator_text = "unknown (runs as the bot)"
    status = {"on": "On", "off": "Off", "problem": f"⚠ {_clip(reminder.get('problem'), 200)}"}[state]
    return {
        "title": f"Reminder {reminder.get('id')}",
        "description": _clip(reminder["msg"], 4000) if reminder.get("msg") else "*No message*",
        "fields": [
            ("Command", command_text),
            ("Next run", next_text),
            ("Repeats", f"every {format_interval(frequency)}" if isinstance(frequency, timedelta) else "once"),
            ("Channel", f"<#{reminder.get('channel_id')}>"),
            ("Creator", creator_text),
            ("Status", status),
        ],
        "state": state,
    }


def option_texts(reminder):
    """(label, description) of a reminder in the jump select (each at most 100 characters)."""
    command = reminder.get("command")
    what = command.split(" ")[0] if command else _clip(reminder.get("msg") or "message", 60)
    when = reminder.get("when")
    parts = [when.strftime("%b %d %H:%M") if isinstance(when, datetime) else "unreadable time"]
    frequency = reminder.get("frequency")
    parts.append(f"every {format_interval(frequency)}" if isinstance(frequency, timedelta) else "once")
    state = _state(reminder)
    if state != "on":
        parts.append(state)
    return _clip(f"{reminder.get('id')} · {what}", 100), _clip(" · ".join(parts), 100)


# ========================
# Storage
# ========================
def _new_id():
    """8 hex characters, unique across all stored reminders."""
    taken = {r.get("id") for reminder_list in reminders.values() for r in reminder_list}
    while True:
        reminder_id = uuid4().hex[:8]
        if reminder_id not in taken:
            return reminder_id


def _snowflake_in(value, what):
    if not isinstance(value, str) or not SNOWFLAKE.fullmatch(value):
        raise StoreError(f"{what} must be a decimal string")
    try:
        return int(value)
    except ValueError:
        raise StoreError(f"{what} is too long") from None


def _snowflake_out(value, what):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StoreError(f"{what} is not a Discord id")
    return str(value)


def _atomic_write(path, data):
    """Write bytes to path: temp file, fsync, rename, fsync the folder (a crash leaves the old file or the new one)."""
    temp_path = f"{path}.tmp"
    try:
        with open(temp_path, "wb") as f:
            f.write(data)
            f.flush()
            fsync(f.fileno())
        replace(temp_path, path)
    except BaseException:
        try:
            remove(temp_path)
        except OSError:
            pass
        raise
    try:
        folder = os_open(dirname(path) or ".", O_RDONLY)
        try:
            fsync(folder)
        finally:
            os_close(folder)
    except OSError:
        pass


class ReminderStore:
    """reminder.json (SCHEMA_VERSION): exact keys, canonical (a no-op save is a zero diff), written atomically."""

    def __init__(self, path):
        self.path = path

    def serialize(self, data) -> str:
        """{guild_id: [reminder]} -> the file's text; StoreError if a reminder can't be stored."""
        guilds = {}
        for guild_id in sorted(data, key=lambda g: (not isinstance(g, int), g if isinstance(g, int) else 0)):
            if data[guild_id]:
                guilds[_snowflake_out(guild_id, "a guild id")] = [self._entry_out(r) for r in data[guild_id]]
        doc = {"schema": STORE_SCHEMA, "schema_version": SCHEMA_VERSION, "time_semantics": TIME_SEMANTICS,
               "guilds": guilds}
        return dumps(doc, indent=2, ensure_ascii=False) + "\n"

    def parse(self, text):
        """The file's text -> defaultdict(list) {guild_id: [reminder]}; StoreError on anything unexpected."""
        try:
            doc = loads(text)
        except ValueError as e:
            raise StoreError(f"not JSON ({e})") from None
        if not isinstance(doc, dict) or doc.get("schema") != STORE_SCHEMA:
            raise StoreError(f"not a {STORE_SCHEMA} file")
        version = doc.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise StoreError("schema_version is missing")
        if version > SCHEMA_VERSION:
            raise StoreError(f"written by newer code (schema_version {version}); update the bot")
        if version != SCHEMA_VERSION:
            raise StoreError(f"unknown schema_version {version}")
        if set(doc) != {"schema", "schema_version", "time_semantics", "guilds"} or not isinstance(doc["guilds"], dict):
            raise StoreError("unexpected top-level keys")
        data = defaultdict(list)
        seen = set()
        for guild_key, entries in doc["guilds"].items():
            guild_id = _snowflake_in(guild_key, "a guild id")
            if not isinstance(entries, list):
                raise StoreError(f"guild {guild_key} does not hold a list")
            for entry in entries:
                reminder = self._entry_in(entry)
                if reminder["id"] in seen:
                    raise StoreError(f"reminder {reminder['id']} appears twice")
                seen.add(reminder["id"])
                data[guild_id].append(reminder)
        return data

    def load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            return self.parse(f.read())

    def save(self, data):
        _atomic_write(self.path, self.serialize(data).encode("utf-8"))

    @staticmethod
    def _entry_out(reminder):
        rid = reminder.get("id")
        if not isinstance(rid, str) or not rid:
            raise StoreError("a reminder has no id")
        when = reminder.get("when")
        if not isinstance(when, datetime) or when.tzinfo is None:
            raise StoreError(f"reminder {rid}: unreadable time")
        frequency = reminder.get("frequency")
        seconds = None
        if frequency is not None:
            if not isinstance(frequency, timedelta) or frequency <= timedelta(0) \
                    or frequency != timedelta(seconds=int(frequency.total_seconds())):
                raise StoreError(f"reminder {rid}: frequency is not a whole number of seconds")
            seconds = int(frequency.total_seconds())
        creator_id = reminder.get("creator_id")
        version = reminder.get("v")
        return {
            "id": rid,
            "channel_id": _snowflake_out(reminder.get("channel_id"), f"reminder {rid}: the channel"),
            "msg": str(reminder.get("msg") or ""),
            "when": when.astimezone(TIMEZONE).isoformat(),
            "frequency": seconds,
            "command": reminder.get("command") or None,
            "enabled": bool(reminder.get("enabled", True)),
            "creator_id": None if creator_id is None else _snowflake_out(creator_id, f"reminder {rid}: the creator"),
            "creator_name": str(reminder.get("creator_name") or ""),
            "v": version if isinstance(version, int) and not isinstance(version, bool) else None,
            "problem": reminder.get("problem") or None,
        }

    @staticmethod
    def _entry_in(entry):
        if not isinstance(entry, dict) or set(entry) != set(KEYS):
            raise StoreError(f"a reminder's keys must be {', '.join(KEYS)}")
        rid = entry["id"]
        if not isinstance(rid, str) or not rid:
            raise StoreError("a reminder has no id")
        try:
            parsed = datetime.fromisoformat(entry["when"])
        except (TypeError, ValueError):
            raise StoreError(f"reminder {rid}: unreadable time") from None
        if parsed.tzinfo is None:
            raise StoreError(f"reminder {rid}: the time has no UTC offset")
        # Keep the stored wall time (a spring-forward gap time stays as written); fold picks the offset.
        when = None
        for fold in (0, 1):
            candidate = parsed.replace(tzinfo=TIMEZONE, fold=fold)
            if candidate.utcoffset() == parsed.utcoffset():
                when = candidate
                break
        if when is None:
            raise StoreError(f"reminder {rid}: the time's offset is not {ZONE_LABEL}'s at that time")
        frequency = entry["frequency"]
        if frequency is not None and (isinstance(frequency, bool) or not isinstance(frequency, int) or frequency <= 0):
            raise StoreError(f"reminder {rid}: frequency must be whole seconds or null")
        try:
            frequency = timedelta(seconds=frequency) if frequency else None
        except OverflowError:
            raise StoreError(f"reminder {rid}: frequency is too large") from None
        for key in ("msg", "creator_name"):
            if not isinstance(entry[key], str):
                raise StoreError(f"reminder {rid}: {key} must be text")
        for key in ("command", "problem"):
            if entry[key] is not None and (not isinstance(entry[key], str) or not entry[key]):
                raise StoreError(f"reminder {rid}: {key} must be text or null")
        if not isinstance(entry["enabled"], bool):
            raise StoreError(f"reminder {rid}: enabled must be true or false")
        version = entry["v"]
        if version is not None and (isinstance(version, bool) or not isinstance(version, int)):
            raise StoreError(f"reminder {rid}: v must be a number or null")
        creator_id = entry["creator_id"]
        return {
            "id": rid,
            "channel_id": _snowflake_in(entry["channel_id"], f"reminder {rid}: channel_id"),
            "msg": entry["msg"],
            "when": when,
            "frequency": frequency,
            "command": entry["command"],
            "enabled": entry["enabled"],
            "creator_id": None if creator_id is None else _snowflake_in(creator_id, f"reminder {rid}: creator_id"),
            "creator_name": entry["creator_name"],
            "v": version,
            "problem": entry["problem"],
        }


_store = ReminderStore(STORE_PATH)


def _migrate_command(reminder):
    """
    Rewrite a stored command into canonical form, keeping what it actually did.

    Pre-v2 booleans are read the way the old parser read them, and options the command no longer has are dropped
    (RETIRED_OPTIONS: /update_stats use_local_repo and use_local_timestamp since local mode was archived).
    Returns False (leave the reminder unmigrated, retry later) while slash_registry is still empty.
    """
    command = reminder.get("command")
    if not command:
        return True
    if not slash_registry:
        return False
    try:
        _, name, values, dropped = resolve_command(command, legacy=reminder.get("v") is None)
    except ValueError as e:
        print(f"Reminder {reminder['id']}: stored command `{command}` can't run ({e}); it will report this when due.")
        return True
    canonical = render(name, values)
    if canonical != command:
        reminder["command"] = canonical
        note = f" (dropped retired `{' '.join(dropped)}`)" if dropped else ""
        print(f"Reminder {reminder['id']}: migrated command `{command}` -> `{canonical}`{note}")
    return True


def _outdated(reminder):
    """True while a stored reminder still needs _migrate: no version yet, or an older one."""
    version = reminder.get("v")
    return not isinstance(version, int) or version < REMINDER_VERSION


def _migrate(reminder):
    """
    Bring one stored reminder up to REMINDER_VERSION in place. Never drops it: anything unreadable gets a
    'problem' so the scheduler skips it and /reminders shows it for removal.

    Returns:
        True if the reminder changed
    """
    renamed = "disabled" in reminder
    if renamed:                                 # reminder.pkl's name for problem
        reminder["problem"] = reminder.pop("disabled") or None
    reminder.setdefault("problem", None)        # defaults alone are no change (no needless save)
    reminder.setdefault("enabled", True)
    before = dict(reminder)
    if not isinstance(reminder.get("id"), str) or not reminder.get("id"):
        reminder["id"] = _new_id()
    try:
        reminder["channel_id"] = int(reminder.get("channel_id"))
    except (TypeError, ValueError):
        reminder["problem"] = "unreadable channel"

    msg = reminder.get("msg")
    reminder["msg"] = msg if isinstance(msg, str) else ("" if msg is None else str(msg))

    when = reminder.get("when")
    if isinstance(when, datetime):
        if when.tzinfo is None:
            when = when.replace(tzinfo=TIMEZONE)
        elif when.tzinfo is not TIMEZONE:
            when = when.astimezone(TIMEZONE)
        reminder["when"] = when
    else:
        reminder["problem"] = "unreadable time"

    frequency = reminder.get("frequency")
    if isinstance(frequency, (int, float)) and not isinstance(frequency, bool):
        frequency = timedelta(seconds=frequency) if frequency else None
    if frequency is not None and not isinstance(frequency, timedelta):
        reminder["problem"] = "unreadable frequency"
    elif frequency is not None and frequency <= timedelta(0):
        reminder["problem"] = "frequency is not positive"
    else:
        reminder["frequency"] = frequency or None

    command = reminder.get("command")
    if command is not None and not isinstance(command, str):
        command = str(command)
    reminder["command"] = command.strip() if command and command.strip() else None
    reminder.setdefault("creator_id", None)
    reminder.setdefault("creator_name", "")

    if _outdated(reminder) and _migrate_command(reminder):
        reminder["v"] = REMINDER_VERSION
    if reminder.get("problem") and before.get("problem") != reminder["problem"]:
        print(f"WARNING: reminder {reminder['id']} can't run ({reminder['problem']}); delete it with /remove_reminder.")
    return renamed or reminder != before


def _save_pickle():
    """reminder.pkl, atomically: the fallback while it could not be converted."""
    _atomic_write(LEGACY_PATH, _pickle_bytes(reminders))


def _pickle_bytes(data):
    from io import BytesIO
    buffer = BytesIO()
    dump(data, buffer)
    return buffer.getvalue()


def _store_name():
    return LEGACY_PATH if _backend == "pkl" else STORE_PATH


def save_reminders():
    """Write reminders to disk atomically (reminder.json, or reminder.pkl while it could not be converted)."""
    with _lock:
        if _load_error:
            raise RuntimeError(_load_error)
        if _backend == "pkl":
            _save_pickle()
        else:
            _store.save(reminders)


def _settle(data):
    """Migrate old entries and roll past-due repeating ones forward (off ones stay frozen). True if anything changed."""
    changed = False
    now = datetime.now(TIMEZONE)
    for reminder_list in data.values():
        for reminder in reminder_list:
            changed = _migrate(reminder) or changed
            if reminder.get("frequency") and reminder.get("enabled", True) and not reminder.get("problem") \
                    and reminder["when"] <= now:
                original_time = reminder["when"]
                reminder["when"] = next_run(original_time, reminder["frequency"], now)
                print(f"Rescheduled reminder {reminder['id']} from {original_time} to {reminder['when']}")
                changed = True
    return changed


def _read_pickle():
    """
    reminder.pkl as {guild_id: [reminder]}; None when it can't be read (sets _load_error).

    A missing or empty file is an empty schedule.
    """
    global _load_error
    try:
        with open(LEGACY_PATH, "rb") as f:
            data = load(f)
    except (EOFError, FileNotFoundError):
        print(f"{LEGACY_PATH} is empty or missing.")
        return defaultdict(list)
    except Exception as e:
        _load_error = f"{LEGACY_PATH} could not be read ({type(e).__name__}: {e})"
    else:
        if not isinstance(data, dict) or not _all_lists_of_dicts(data):
            _load_error = f"{LEGACY_PATH} does not hold a reminder dictionary"
    if _load_error:
        _load_error = f"{_load_error}. The file was left untouched; move it aside and restart the bot for a fresh schedule"
        return None
    loaded = defaultdict(list)
    for guild_id, reminder_list in data.items():
        loaded[guild_id] = list(reminder_list)
    return loaded


def _comparable(data):
    """Every stored field of every reminder, with the time's instant, wall time and offset (the conversion check)."""
    out = {}
    for guild_id, reminder_list in data.items():
        if not reminder_list:
            continue
        rows = []
        for r in reminder_list:
            when = r.get("when")
            rows.append((r.get("id"), r.get("channel_id"), r.get("msg") or "", when.timestamp(),
                         when.replace(tzinfo=None), when.utcoffset(), r.get("frequency"), r.get("command") or None,
                         bool(r.get("enabled", True)), r.get("creator_id"), r.get("creator_name") or "", r.get("v"),
                         r.get("problem") or None))
        out[guild_id] = rows
    return out


def _convert_locked():
    """
    reminder.pkl (already loaded into reminders) -> reminder.json, checked by reading it back, then the pickle becomes
    reminder.pkl.bak. Any failure removes the half-made JSON and keeps the pickle (this session saves to it).
    """
    global reminders, _backend, _conversion_error
    created = False
    try:
        _atomic_write(STORE_PATH, _store.serialize(reminders).encode("utf-8"))
        created = True
        again = _store.load()
        if _comparable(again) != _comparable(reminders):
            raise StoreError("the written file reads back different")
        replace(LEGACY_PATH, f"{LEGACY_PATH}.bak")
    except Exception as e:
        if created:
            try:
                remove(STORE_PATH)
            except OSError:
                pass
        _backend = "pkl"
        _conversion_error = f"{type(e).__name__}: {e}"
        print(f"ERROR: could not convert {LEGACY_PATH} to {STORE_PATH} ({_conversion_error}); "
              f"reminders run from {LEGACY_PATH} this session and the next start tries again.")
        return False
    reminders = again
    _backend = "json"
    count = sum(len(v) for v in again.values())
    print(f"Converted {LEGACY_PATH} to {STORE_PATH} ({count} reminders; backup {LEGACY_PATH}.bak).")
    return True


def _load_locked():
    """
    Load the reminders once (call with _lock held): reminder.json, else reminder.pkl converted to it.

    Old entries are migrated and past-due repeating reminders roll forward. An unreadable store turns reminders
    off and leaves the file alone, so a later save can't overwrite reminders that are still recoverable.
    """
    global reminders, _loaded, _load_error, _backend
    if _loaded:
        return
    _loaded = True

    if exists(STORE_PATH):
        print(f"Trying to open {STORE_PATH}.")
        if exists(LEGACY_PATH):
            print(f"{LEGACY_PATH} is ignored: {STORE_PATH} exists (move {STORE_PATH} aside to convert the pickle again).")
        try:
            loaded = _store.load()
            changed = _settle(loaded)
        except Exception as e:
            reason = e if isinstance(e, StoreError) else f"{type(e).__name__}: {e}"
            _load_error = (f"{STORE_PATH} could not be read ({reason}). The file was left untouched; "
                           f"fix it, or move it aside and restart the bot for a fresh schedule")
        else:
            _backend = "json"
            reminders = loaded
            if changed:
                _save_after_load()
            return
    elif exists(LEGACY_PATH):
        print(f"Trying to open {LEGACY_PATH}.")
        loaded = _read_pickle()
        if loaded is not None:
            reminders = loaded
            changed = _settle(reminders)
            if not _convert_locked() and changed:
                _save_after_load()
            return
    elif exists(f"{LEGACY_PATH}.bak"):
        _load_error = (f"{STORE_PATH} is missing; restore it, or rename {LEGACY_PATH}.bak to {LEGACY_PATH} "
                       f"and restart the bot")
    else:
        print(f"No {STORE_PATH} yet: an empty schedule.")
        return

    print(f"ERROR: {_load_error}. Reminders are OFF.")
    if ON_LOAD_ERROR == "raise":
        raise StoreError(_load_error)


def _save_after_load():
    try:
        save_reminders()
        print("Saved rescheduled reminders to disk.")
    except Exception:
        print(f"ERROR: could not save {_store_name()}; the loaded schedule still runs:\n{format_exc()}")


def _all_lists_of_dicts(data):
    """True when every value of data is a list of dicts (the only shape _migrate can handle)."""
    for reminder_list in data.values():
        if not isinstance(reminder_list, list):
            return False
        for reminder in reminder_list:
            if not isinstance(reminder, dict):
                return False
    return True


def _ready_locked():
    """Load on first use (call with _lock held). Returns the reason reminders are off, or None."""
    _load_locked()
    return _load_error


def _off_message():
    return f"✗ Reminders are off: {_load_error}."


def status():
    """A line about the store when something needs the owner (the conversion failed), else None."""
    if _conversion_error:
        return f"Reminders run from {LEGACY_PATH}: conversion failed ({_conversion_error})"
    return None


# ========================
# Scheduler
# ========================
def _channel_missing(channel_id):
    """True when the bot is connected but can't see the channel (deleted, or access lost)."""
    client = _client
    if client is None:
        return False
    try:
        return client.is_ready() and client.get_channel(channel_id) is None
    except Exception:
        return False


def _reminder_interaction(reminder, guild_id):
    """FakeInteraction for a firing reminder, acting as the reminder's creator."""
    fake = make_fake_interaction(reminder["channel_id"], guild_id)
    if RUN_AS_CREATOR:
        creator_id = reminder.get("creator_id")
        if creator_id is not None:
            fake.user = _ReminderUser(creator_id, reminder.get("creator_name") or str(creator_id))
        elif not LEGACY_RUNS_AS_BOT:
            fake.user = _ReminderUser(0, "unknown reminder creator")
    return fake


async def _run_reminder_command(func, interaction, values, reminder_id, name):
    """Await a reminder's slash callback; report a crash in the channel instead of losing it in a background task."""
    try:
        result = func(interaction, **values)
        if isawaitable(result):
            await result
    except Exception as e:
        print(f"ERROR: reminder {reminder_id} command /{name} raised:\n{format_exc()}")
        queue_message(interaction, f"⚠ Reminder `{reminder_id}`: `/{name}` failed ({type(e).__name__}: {str(e)[:300]}).")


def _fire(reminder, guild_id):
    """
    Queue one due reminder's message and command.

    Returns:
        False if its channel is gone (nothing was queued), else True
    """
    reminder_id = reminder["id"]
    if _channel_missing(reminder["channel_id"]):
        print(f"Reminder {reminder_id}: channel {reminder['channel_id']} not found (deleted or no access); skipped.")
        return False

    fake = _reminder_interaction(reminder, guild_id)
    if reminder.get("msg"):
        queue_long_message(fake, reminder["msg"])

    command = reminder.get("command")
    if command:
        try:
            func, name, values, _ = resolve_command(command, legacy=reminder.get("v") is None)
        except ValueError as e:
            queue_message(fake, f"⚠ Reminder `{reminder_id}` skipped its command: {e}. "
                                f"Re-create it, then `/remove_reminder reminder_id:{reminder_id}`.")
            return True
        from commands.card_sync import on_use      # lazy: card_sync imports the card modules
        on_use(name)                                # a card command: new card data looked for, as when typed
        queue_command(_run_reminder_command, func, fake, values, reminder_id, name)
    return True


def _process_due(reminder_list, reminder, guild_id, now):
    """Fire a reminder if it is due, then reschedule or remove it. Returns True if anything changed."""
    if not reminder.get("enabled", True) or reminder.get("problem") or reminder["when"] > now:
        return False
    if _outdated(reminder):
        _migrate(reminder)
        if reminder.get("problem"):
            return True
    try:
        delivered = _fire(reminder, guild_id)
    except Exception:
        print(f"ERROR: reminder {reminder.get('id')} failed to fire:\n{format_exc()}")
        delivered = True    # a reminder that crashes must not re-fire every second

    if reminder.get("frequency"):
        reminder["when"] = next_run(reminder["when"], reminder["frequency"], now)
    elif delivered:
        if reminder in reminder_list:
            reminder_list.remove(reminder)
    else:
        reminder["when"] = now + CHANNEL_RETRY
    return True


def _check_pass_locked():
    now = datetime.now(TIMEZONE)
    changed = False
    for guild_id, reminder_list in list(reminders.items()):
        for reminder in list(reminder_list):
            try:
                changed = _process_due(reminder_list, reminder, guild_id, now) or changed
            except Exception:
                print(f"ERROR: reminder {reminder.get('id')} is broken and was stopped:\n{format_exc()}")
                reminder["problem"] = "crashed the scheduler"
                changed = True
        if not reminder_list:
            reminders.pop(guild_id, None)

    if changed:
        try:
            save_reminders()
        except Exception:
            print(f"ERROR: could not save {_store_name()}:\n{format_exc()}")


def check_reminder():
    """
    Fire every due reminder: post its message, queue its command, then repeat or remove it.

    Called every second by the bot's task loop. A pass that finds another still running is skipped, a reminder that
    fails never stops the others, and a repeating reminder jumps straight past now instead of replaying missed runs.
    """
    if not _check_lock.acquire(blocking=False):
        return
    try:
        with _lock:
            if _ready_locked():
                return
            _check_pass_locked()
    finally:
        _check_lock.release()


# ========================
# Changes (the panel and the slash shortcuts; each holds _lock, re-checks rights and rolls back a failed save)
# ========================
def _ready_or_raise():
    if _ready_locked():
        raise ReminderOff(_off_message())


def _find_locked(guild_id, reminder_id):
    wanted = (reminder_id or "").strip().strip("`").lower()
    for reminder in reminders.get(guild_id, []):
        if str(reminder.get("id", "")).lower() == wanted:
            return reminder
    raise ReminderError(f"No reminder found with ID `{wanted}`." if wanted else "Give a reminder ID.")


def _check_edit(reminder, interaction):
    if not can_edit(reminder, interaction.user.id, is_admin(interaction)):
        raise ReminderError("Only an admin or the reminder's creator can change it.")


def _commit_locked(undo):
    """Save; on failure undo the change in memory and refuse."""
    try:
        save_reminders()
    except Exception as e:
        undo()
        raise ReminderError(f"Could not save {_store_name()} ({e}); nothing changed.") from None


def _restore(reminder, before):
    reminder.clear()
    reminder.update(before)


def snapshot(guild_id):
    """Copies of a server's reminders in creation order; ReminderOff when reminders are off."""
    with _lock:
        _ready_or_raise()
        return [dict(r) for r in reminders.get(guild_id, [])]


def create(interaction, msg="", when="", offset="", frequency="", command=""):
    """
    A new reminder in the interaction's channel, created by its user (admins only).

    Returns:
        (copy of the reminder, [notes]); ReminderError with the reason it was not set
    """
    guild_id = interaction.guild_id
    if guild_id is None:
        raise ReminderError("reminders work in servers only")
    if not is_admin(interaction):
        raise ReminderError("only admins can add reminders")
    with _lock:
        _ready_or_raise()
    now = datetime.now(TIMEZONE)
    try:
        remind_at, interval, notes = parse_schedule(when, offset, frequency, now)
        stored_command, command_notes = _store_command(command)
        if not (msg or "").strip() and not stored_command:
            raise ValueError("give a `msg` to post and/or a `command` to run")
    except ValueError as e:
        raise ReminderError(str(e)) from None

    with _lock:
        _ready_or_raise()
        reminder = {
            "id": _new_id(),
            "channel_id": interaction.channel_id,
            "msg": msg or "",
            "when": remind_at,
            "frequency": interval,
            "command": stored_command,
            "enabled": True,
            "creator_id": interaction.user.id,
            "creator_name": str(getattr(interaction.user, "name", "") or ""),
            "v": REMINDER_VERSION,
            "problem": None,
        }
        reminders[guild_id].append(reminder)

        def undo():
            reminders[guild_id].remove(reminder)
            if not reminders[guild_id]:
                reminders.pop(guild_id, None)
        _commit_locked(undo)
        return dict(reminder), command_notes + notes


def delete(interaction, reminder_id):
    """Delete a reminder (admins or its creator). Returns the deleted reminder."""
    guild_id = interaction.guild_id
    with _lock:
        _ready_or_raise()
        target = _find_locked(guild_id, reminder_id)
        _check_edit(target, interaction)
        reminder_list = reminders[guild_id]
        position = reminder_list.index(target)
        reminder_list.remove(target)
        if not reminder_list:
            reminders.pop(guild_id, None)
        _commit_locked(lambda: reminders[guild_id].insert(position, target))
        return dict(target)


def set_enabled(interaction, reminder_id, enabled):
    """
    Turn a reminder on or off (admins or its creator). Off: kept, never fires. On again: a repeating one skips the
    runs it missed; a one-shot whose time passed is refused.

    Returns:
        (copy of the reminder, note)
    """
    guild_id = interaction.guild_id
    enabled = bool(enabled)
    with _lock:
        _ready_or_raise()
        target = _find_locked(guild_id, reminder_id)
        _check_edit(target, interaction)
        if bool(target.get("enabled", True)) == enabled:
            return dict(target), f"Already {'on' if enabled else 'off'}."
        before = dict(target)
        note = "Turned off."
        if enabled:
            if target.get("problem"):
                raise ReminderError(f"It can't run ({target['problem']}). Reschedule or delete it.")
            note = "Turned on."
            now = datetime.now(TIMEZONE)
            if target["when"] <= now:
                if not target.get("frequency"):
                    raise ReminderError("Its time passed while it was off. Reschedule it first.")
                target["when"] = next_run(target["when"], target["frequency"], now)
                note = "Turned on; missed runs skipped."
        target["enabled"] = enabled
        _commit_locked(lambda: _restore(target, before))
        return dict(target), note


def reschedule(interaction, reminder_id, when, frequency):
    """
    New first run and repeat for a reminder (admins or its creator); on/off stays as it is.

    Args:
        when: parse_when text, or a delay such as "30m"
        frequency: Repeat interval, blank = once

    Returns:
        (copy of the reminder, [notes])
    """
    guild_id = interaction.guild_id
    now = datetime.now(TIMEZONE)
    try:
        if when and parse_when(when, now)[0] is None and parse_interval(when) is not None:
            remind_at, interval, notes = parse_schedule("", when, frequency, now)
        else:
            remind_at, interval, notes = parse_schedule(when, "", frequency, now)
    except ValueError as e:
        raise ReminderError(str(e)) from None
    with _lock:
        _ready_or_raise()
        target = _find_locked(guild_id, reminder_id)
        _check_edit(target, interaction)
        before = dict(target)
        target["when"] = remind_at
        target["frequency"] = interval
        if isinstance(target.get("channel_id"), int):
            target["problem"] = None            # its time and repeat were what could be wrong
        _commit_locked(lambda: _restore(target, before))
        if not target.get("enabled", True):
            notes.append("It is still off.")
        return dict(target), notes


# ========================
# Slash Shortcuts
# ========================
def _set_reply(reminder, notes):
    desc = f"Reminder `{reminder['id']}` set for **{format_when(reminder['when'])}**"
    if reminder.get("frequency"):
        desc += f", repeats every {format_interval(reminder['frequency'])}"
    if reminder.get("command"):
        desc += f", runs `{reminder['command']}` with your permissions"
    desc += ". Output goes to this channel."
    for note in notes:
        desc += f"\n{note}"
    return desc


def set_reminder(interaction: Interaction,
                 msg: str = "",
                 when: str = "",
                 offset: str = "",
                 frequency: str = "",
                 command: str = ""):
    """
    Create a reminder that posts a message and/or runs a bot command in this channel.

    The command runs with the creator's permissions (re-checked by the command itself every time it fires).

    Args:
        interaction: Discord interaction
        msg: Message to send when the reminder fires
        when: Start time in Pacific time ("13:00", "1PM", "sun 3AM", "tomorrow 9AM", "2026-12-25 9AM");
              blank means now (only together with an offset)
        offset: Delay added to the first run (e.g. "5m", "2h", "1d")
        frequency: How often to repeat (e.g. "1h", "1d", "1w", "1d12h")
        command: Bot command with name:value options (e.g. "/sync_emojis refresh:True")
    """
    try:
        reminder, notes = create(interaction, msg, when, offset, frequency, command)
    except ReminderOff as e:
        return queue_message(interaction, str(e))
    except ReminderError as e:
        return queue_message(interaction, f"✗ Reminder NOT set: {e}")
    if status():
        notes = notes + [f"⚠ {status()}."]
    queue_message(interaction, _set_reply(reminder, notes), allowed_mentions=AllowedMentions.none())


def remove_reminder(interaction: Interaction, reminder_id: str = ""):
    """
    Delete a reminder by its ID. Admins can remove any reminder, everyone else only their own.

    Args:
        interaction: Discord interaction
        reminder_id: The 8-character ID of the reminder to delete
    """
    try:
        target = delete(interaction, reminder_id)
    except ReminderOff as e:
        return queue_message(interaction, str(e))
    except ReminderError as e:
        return queue_message(interaction, f"✗ {e}")
    queue_message(interaction, f"Reminder `{target['id']}` has been removed.")


# ========================
# Initialization
# ========================
def bind_reminder_client(client) -> None:
    """Called once from main.py: lets the scheduler skip reminders whose channel no longer exists."""
    global _client
    _client = client


def init_reminder():
    """
    Load reminders from disk (called from on_ready via queue_command), converting reminder.pkl once.

    Old entries are migrated in place and past-due repeating reminders roll forward to their next future slot;
    a one-shot reminder that came due while the bot was down fires right after startup.
    """
    with _lock:
        _ready_locked()


# ========================
# Module Exports
# ========================
__all__ = [
    'bind_reminder_client',
    'init_reminder',
    'set_reminder',
    'check_reminder',
    'remove_reminder',
    'snapshot',
    'create',
    'delete',
    'set_enabled',
    'reschedule',
    'status',
]
