from collections import defaultdict
from datetime import date, datetime, timedelta
from discord import AllowedMentions, Interaction
from discord.app_commands import Choice
from inspect import Parameter, isawaitable, signature
from os import fsync, remove, replace
from pickle import load, dump
from re import compile as re_compile
from shlex import quote as shlex_quote, split as shlex_split
from threading import Lock, RLock
from traceback import format_exc
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints
from uuid import uuid4

from commands.utils import (
    slash_registry,
    queue_message,
    queue_command,
    make_fake_interaction,
    perms_check,
    split_long_message,
)
from global_config import REMIND_PKL, TIMEZONE

# ========================
# Configuration
# ========================
# reminder.pkl keeps the original layout (defaultdict(list) of {guild_id: [reminder dict]}) so older code can still
# read it. Keys of a reminder dict:
#   id, channel_id, msg, when (aware datetime in TIMEZONE), frequency (timedelta or None),
#   command (None or canonical "/name key:value ..."), creator_id (None = saved before creator tracking),
#   creator_name, v (REMINDER_VERSION once migrated), disabled (only present when the scheduler must skip it: the reason)
REMINDER_VERSION = 3                        # 3: retired options dropped from stored commands (RETIRED_OPTIONS)
MAX_INTERVAL = timedelta(days=366)          # longer offsets/frequencies are typos, and huge ones overflow datetime
CHANNEL_RETRY = timedelta(hours=1)          # a one-shot whose channel is gone waits this long and tries again
LEGACY_RUNS_AS_BOT = True                   # reminders saved before creator tracking keep running as the bot, as before
# Options a command no longer has. A stored reminder that still sets one drops it instead of failing to run.
# Each entry: (removed options, the command's option order before they were removed, because pre-v2 reminders
# could give options by position).
RETIRED_OPTIONS = {
    # local mode was archived on 2026-10-20 (branch archive/local-update-stats); /update_stats always reads GitHub
    "update_stats": (("use_local_repo", "use_local_timestamp"),
                     ("output_problematic_cards", "use_local_repo", "use_local_timestamp", "force_update")),
}
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

# ========================
# Global State
# ========================
reminders = defaultdict(list)
_client = None              # discord.Client, set by bind_reminder_client; used to spot deleted channels
_loaded = False
_load_error = None          # set when reminder.pkl exists but can't be read: reminders stay off, file untouched
_lock = RLock()             # guards reminders and the file: set/list/remove run in threads next to check_reminder
_check_lock = Lock()        # single-flight: a pass is queued every second, and two overlapping passes would double-fire


class _ReminderUser:
    """Stands in for the reminder's creator, so perms_check sees who scheduled the command."""
    def __init__(self, user_id, name):
        self.id = user_id
        self.name = name
        self.display_name = name
        self.global_name = name
        self.mention = f"<@{user_id}>"
        self.bot = False

    def __str__(self):
        return self.name


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


def _parse_when(when_str, now):
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


def parse_time_str(when_str: str) -> datetime | None:
    """
    Parse a time string into a datetime in TIMEZONE (Pacific, follows daylight saving).

    Args:
        when_str: "13:00", "1PM", "1:30PM", optionally with "tomorrow", a weekday or a date in front

    Returns:
        datetime in TIMEZONE, or None if parsing fails
    """
    return _parse_when(when_str, datetime.now(TIMEZONE))[0]


def parse_interval(time_str: str) -> timedelta | None:
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


def _checked_interval(label, raw):
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


def _format_interval(interval):
    """timedelta -> '1w', '1d12h', '30m'."""
    seconds = int(interval.total_seconds())
    parts = []
    for unit, size in (("w", 604800), ("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        if seconds >= size:
            parts.append(f"{seconds // size}{unit}")
            seconds = seconds - (seconds // size) * size
    return "".join(parts) or "0s"


def _format_when(when):
    """Pacific wall time plus a Discord relative timestamp that each reader sees in their own time zone."""
    return f"{when.strftime('%Y-%m-%d %H:%M %Z')} (<t:{int(when.timestamp())}:R>)"


def _next_run(when, frequency, now):
    """First when + n*frequency after now, in one step (a stalled bot must not replay every missed run)."""
    return when + frequency * ((now - when) // frequency + 1)


# ========================
# Command Parsing
# ========================
def _tokenize(command):
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
                raise ValueError(f"`{token}` has no `name:` but comes after a name:value option")
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
    raise ValueError(f"`{option}` must be True or False, got `{raw}`")


def _convert(raw, annotation, option, legacy):
    """Turn a text value into what the slash callback's annotation expects; ValueError if it can't."""
    if annotation is Parameter.empty or annotation is Any or annotation is str:
        return raw
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        choices = [arg for arg in get_args(annotation) if arg is not type(None)]
        problem = f"`{option}` can't take `{raw}`"
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
            raise ValueError(f"`{option}` must be a whole number, got `{raw}`") from None
    if annotation is float:
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"`{option}` must be a number, got `{raw}`") from None
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


def _render(name, values):
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


def _resolve_command(command, legacy=False):
    """
    Look up and bind a stored command.

    Returns:
        (callback, name, {option: value}, ["option:value" dropped as retired]); raises ValueError with a
        user-facing reason
    """
    name, args, kwargs = _tokenize(command)
    func = slash_registry.get(name)
    if func is None:
        raise ValueError(f"`/{name}` is not a bot command")
    args, kwargs, dropped = _drop_retired(name, args, kwargs, legacy)
    return func, name, _bind(func, args, kwargs, legacy), dropped


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
        _, name, values, dropped = _resolve_command(command, legacy=reminder.get("v") is None)
    except ValueError as e:
        print(f"Reminder {reminder['id']}: stored command `{command}` can't run ({e}); it will report this when due.")
        return True
    canonical = _render(name, values)
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
    'disabled' reason so the scheduler skips it and /list_reminders shows it for removal.

    Returns:
        True if the reminder changed
    """
    before = dict(reminder)
    if not isinstance(reminder.get("id"), str) or not reminder.get("id"):
        reminder["id"] = _new_id()
    try:
        reminder["channel_id"] = int(reminder.get("channel_id"))
    except (TypeError, ValueError):
        reminder["disabled"] = "unreadable channel"

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
        reminder["disabled"] = "unreadable time"

    frequency = reminder.get("frequency")
    if isinstance(frequency, (int, float)) and not isinstance(frequency, bool):
        frequency = timedelta(seconds=frequency) if frequency else None
    if frequency is not None and not isinstance(frequency, timedelta):
        reminder["disabled"] = "unreadable frequency"
    elif frequency is not None and frequency <= timedelta(0):
        reminder["disabled"] = "frequency is not positive"
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
    if reminder.get("disabled") and before.get("disabled") != reminder["disabled"]:
        print(f"WARNING: reminder {reminder['id']} is disabled ({reminder['disabled']}); remove it with /remove_reminder.")
    return reminder != before


def save_reminders():
    """Write reminders to disk atomically (temp file + fsync + rename), so a crash can't leave a truncated file."""
    with _lock:
        if _load_error:
            raise RuntimeError(_load_error)
        temp_path = f"{REMIND_PKL}.tmp"
        try:
            with open(temp_path, "wb") as f:
                dump(reminders, f)
                f.flush()
                fsync(f.fileno())
            replace(temp_path, REMIND_PKL)
        except BaseException:
            try:
                remove(temp_path)
            except OSError:
                pass
            raise


def _load_locked():
    """
    Load reminder.pkl once (call with _lock held), migrate old entries and roll past-due repeating reminders forward.

    A missing or empty file is an empty schedule. Any other read error turns reminders off and leaves the file alone,
    so a later save can't overwrite reminders that are still recoverable.
    """
    global reminders, _loaded, _load_error
    if _loaded:
        return
    _loaded = True

    print(f"Trying to open {REMIND_PKL}.")
    try:
        with open(REMIND_PKL, "rb") as f:
            data = load(f)
    except (EOFError, FileNotFoundError):
        print(f"{REMIND_PKL} is empty or missing.")
        return
    except Exception as e:
        _load_error = f"{REMIND_PKL} could not be read ({type(e).__name__}: {e})"
    else:
        if not isinstance(data, dict) or not _all_lists_of_dicts(data):
            _load_error = f"{REMIND_PKL} does not hold a reminder dictionary"
    if _load_error:
        print(f"ERROR: {_load_error}. Reminders are OFF and the file was left untouched; "
              f"move it aside and restart the bot to start a fresh schedule.")
        return

    print(f"Existing dict found in {REMIND_PKL}, updating entries...")
    loaded = defaultdict(list)
    changed = False
    now = datetime.now(TIMEZONE)
    for guild_id, reminder_list in data.items():
        for reminder in reminder_list:
            changed = _migrate(reminder) or changed
            if reminder.get("frequency") and not reminder.get("disabled") and reminder["when"] <= now:
                original_time = reminder["when"]
                reminder["when"] = _next_run(original_time, reminder["frequency"], now)
                print(f"Rescheduled reminder {reminder['id']} from {original_time} to {reminder['when']}")
                changed = True
            loaded[guild_id].append(reminder)
    reminders = loaded

    if changed:
        try:
            save_reminders()
            print("Saved rescheduled reminders to disk.")
        except Exception:
            print(f"ERROR: could not save {REMIND_PKL}; the loaded schedule still runs:\n{format_exc()}")


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
    return (f"✗ Reminders are off: {_load_error}. The file was left untouched; "
            f"move it aside and restart the bot to start a fresh schedule.")


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
    fake.scheduled = True
    fake.reminder_id = reminder["id"]
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
        for chunk in split_long_message(reminder["msg"]):
            queue_message(fake, chunk)

    command = reminder.get("command")
    if command:
        try:
            func, name, values, _ = _resolve_command(command, legacy=reminder.get("v") is None)
        except ValueError as e:
            queue_message(fake, f"⚠ Reminder `{reminder_id}` skipped its command: {e}. "
                                f"Re-create it, then `/remove_reminder reminder_id:{reminder_id}`.")
            return True
        queue_command(_run_reminder_command, func, fake, values, reminder_id, name)
    return True


def _process_due(reminder_list, reminder, guild_id, now):
    """Fire a reminder if it is due, then reschedule or remove it. Returns True if anything changed."""
    if reminder.get("disabled") or reminder["when"] > now:
        return False
    if _outdated(reminder):
        _migrate(reminder)
        if reminder.get("disabled"):
            return True
    try:
        delivered = _fire(reminder, guild_id)
    except Exception:
        print(f"ERROR: reminder {reminder.get('id')} failed to fire:\n{format_exc()}")
        delivered = True    # a reminder that crashes must not re-fire every second

    if reminder.get("frequency"):
        reminder["when"] = _next_run(reminder["when"], reminder["frequency"], now)
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
                print(f"ERROR: reminder {reminder.get('id')} is broken and was disabled:\n{format_exc()}")
                reminder["disabled"] = "crashed the scheduler"
                changed = True
        if not reminder_list:
            reminders.pop(guild_id, None)

    if changed:
        try:
            save_reminders()
        except Exception:
            print(f"ERROR: could not save {REMIND_PKL}:\n{format_exc()}")


# ========================
# Public API Functions
# ========================
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
    with _lock:
        if _ready_locked():
            return queue_message(interaction, _off_message())

    now = datetime.now(TIMEZONE)
    notes = []
    try:
        offset_delta = _checked_interval("offset", offset)
        interval = _checked_interval("frequency", frequency)

        if when and when.strip():
            remind_at, fixed_day = _parse_when(when, now)
            if remind_at is None:
                raise ValueError(f"invalid time `{when}`. Try `13:00`, `1PM`, `sun 3AM`, `tomorrow 9AM` "
                                 f"or `2026-12-25 9AM` (Pacific time)")
        elif offset_delta:
            remind_at, fixed_day = now, False
        else:
            raise ValueError("give a time with `when` (e.g. `13:00`, `1PM`, `sun 3AM`) or a delay with `offset`")
        if offset_delta:
            remind_at = remind_at + offset_delta
        if remind_at <= now:
            if fixed_day:
                raise ValueError(f"{remind_at.strftime('%Y-%m-%d %H:%M %Z')} is in the past")
            if when and when.strip():
                notes.append("That time already passed today, so it fires right away "
                             "(use `when:tomorrow ...` or `offset:1d` to start tomorrow).")

        stored_command = None
        if command and command.strip():
            if slash_registry:
                _, name, values, dropped = _resolve_command(command.strip())
                if dropped:
                    notes.append(f"Dropped `{' '.join(dropped)}`: /{name} no longer has "
                                 f"{'that option' if len(dropped) == 1 else 'those options'}.")
                stored_command = _render(name, values)
            else:
                stored_command = command.strip()
        if not (msg or "").strip() and not stored_command:
            raise ValueError("give a `msg` to post and/or a `command` to run")
    except ValueError as e:
        return queue_message(interaction, f"✗ Reminder NOT set: {e}")

    guild_id = interaction.guild_id
    with _lock:
        reminder = {
            "id": _new_id(),
            "channel_id": interaction.channel_id,
            "msg": msg or "",
            "when": remind_at,
            "frequency": interval,
            "command": stored_command,
            "creator_id": interaction.user.id,
            "creator_name": str(getattr(interaction.user, "name", "") or ""),
            "v": REMINDER_VERSION,
        }
        reminders[guild_id].append(reminder)
        try:
            save_reminders()
        except Exception as e:
            reminders[guild_id].remove(reminder)
            if not reminders[guild_id]:
                reminders.pop(guild_id, None)
            return queue_message(interaction, f"✗ Reminder NOT set: could not save {REMIND_PKL} ({e}).")

    desc = f"Reminder `{reminder['id']}` set for **{_format_when(remind_at)}**"
    if interval:
        desc += f", repeats every {_format_interval(interval)}"
    if stored_command:
        desc += f", runs `{stored_command}` with your permissions"
    desc += ". Output goes to this channel."
    for note in notes:
        desc += f"\n{note}"
    queue_message(interaction, desc, allowed_mentions=AllowedMentions.none())


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


def list_reminders(interaction: Interaction, all: bool = False, hidden: bool = False):
    """
    List reminders for the current channel or entire server.

    Args:
        interaction: Discord interaction
        all: If True, show all reminders in the server; if False, only show current channel
        hidden: If True, send as ephemeral message (only visible to user)
    """
    guild_id = interaction.guild_id
    channel_id = interaction.channel_id
    with _lock:
        if _ready_locked():
            return queue_message(interaction, _off_message(), ephemeral=hidden)
        server_reminders = [dict(r) for r in reminders.get(guild_id, [])]

    if not server_reminders:
        return queue_message(interaction, "No reminders set for this server.", ephemeral=hidden)

    if all:
        filtered = server_reminders
        scope_text = "this server"
    else:
        filtered = [r for r in server_reminders if r.get("channel_id") == channel_id]
        scope_text = f"this channel (<#{channel_id}>)"

    if not filtered:
        return queue_message(interaction, f"No reminders found for {scope_text}.", ephemeral=hidden)

    lines = []
    for r in filtered:
        msg_str = f"**{r['msg']}**" if r.get("msg") else "*<no message>*"
        line = f"`{r.get('id')}` — {msg_str}"
        if isinstance(r.get("when"), datetime):
            line += f" at {_format_when(r['when'])}"
        if isinstance(r.get("frequency"), timedelta):
            line += f", repeats every {_format_interval(r['frequency'])}"
        if r.get("command"):
            line += f", runs `{r['command']}`"
        if r.get("creator_id") is not None:
            line += f", by <@{r['creator_id']}>"
        else:
            line += ", by unknown (older reminder, runs as the bot)"
        if all:
            line += f" in <#{r.get('channel_id')}>"
        if r.get("disabled"):
            line += f" ⚠ DISABLED: {r['disabled']}"
        lines.append(line)

    output = "\n".join(lines)
    for chunk in split_long_message(f"Reminders for {scope_text}:\n{output}"):
        queue_message(interaction, chunk, ephemeral=hidden, allowed_mentions=AllowedMentions.none())


def remove_reminder(interaction: Interaction, reminder_id: str = ""):
    """
    Delete a reminder by its ID. Admins can remove any reminder, everyone else only their own.

    Args:
        interaction: Discord interaction
        reminder_id: The 8-character ID of the reminder to delete
    """
    guild_id = interaction.guild_id
    wanted = (reminder_id or "").strip().strip("`").lower()
    with _lock:
        if _ready_locked():
            return queue_message(interaction, _off_message())
        reminder_list = reminders.get(guild_id)
        if not reminder_list:
            return queue_message(interaction, "No reminders found for this server.")

        target = None
        for reminder in reminder_list:
            if str(reminder.get("id", "")).lower() == wanted:
                target = reminder
                break
        if target is None:
            return queue_message(interaction, f"No reminder found with ID `{wanted}`.")
        if perms_check(interaction) and target.get("creator_id") != interaction.user.id:
            return queue_message(interaction, "Only an admin or the reminder's creator can remove it.")

        position = reminder_list.index(target)
        reminder_list.remove(target)
        if not reminder_list:
            reminders.pop(guild_id, None)
        try:
            save_reminders()
        except Exception as e:
            reminders[guild_id].insert(position, target)
            return queue_message(interaction, f"✗ Reminder NOT removed: could not save {REMIND_PKL} ({e}).")

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
    Load reminders from disk (called from on_ready via queue_command).

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
    'list_reminders',
    'remove_reminder',
]
