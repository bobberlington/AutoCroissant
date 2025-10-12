from datetime import datetime, timedelta
from discord import Interaction
from pickle import load, dump
from zoneinfo import ZoneInfo

from commands.utils import messages
from global_config import REMIND_PKL

reminders = {}


def save_reminders():
    """Save the reminder dict to a PICKLE."""
    with open(REMIND_PKL, "wb") as f:
        dump(reminders, f)


def init_reminder():
    """Load reminders from pickle or initialize empty store."""
    global reminders

    print(f"Trying to open {REMIND_PKL}")
    try:
        with open(REMIND_PKL, 'rb') as f:
            reminders = load(f)
        print(f"Existing dict found in {REMIND_PKL}, updating entries...")
    except (EOFError, FileNotFoundError):
        print(f"{REMIND_PKL} is completely empty or doesn't exist, rebuilding entire dict...")


def parse_time_str(when_str: str) -> datetime | None:
    """Parse a time string like '13:00' or '1PM' (PST) into a datetime."""
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    when_str = when_str.strip().upper().replace(" ", "")
    target_time = None

    try:
        # Try formats like 13:00, 1PM, 1:30PM
        for fmt in ("%H:%M", "%I%p", "%I:%M%p"):
            try:
                t = datetime.strptime(when_str, fmt).time()
                target_time = t
                break
            except ValueError:
                continue
        if not target_time:
            return None

        remind_at = datetime.combine(now.date(), target_time, tzinfo=ZoneInfo("America/Los_Angeles"))
        return remind_at
    except Exception:
        return None


def parse_interval(how_often: str) -> timedelta | None:
    """Parse an interval like '1h', '30m', '2d', '1w'."""
    if not how_often:
        return None
    how_often = how_often.strip().lower()
    try:
        unit = how_often[-1]
        val = int(how_often[:-1])
        if unit == "s":
            return timedelta(seconds=val)
        if unit == "m":
            return timedelta(minutes=val)
        if unit == "h":
            return timedelta(hours=val)
        if unit == "d":
            return timedelta(days=val)
        if unit == "w":
            return timedelta(weeks=val)
        return None
    except Exception:
        return None


def set_reminder(interaction: Interaction, msg: str = "", when: str = "", how_often: str = ""):
    """Register a new reminder."""
    global reminders

    guild_id = interaction.guild_id
    channel_id = interaction.channel_id

    remind_at = parse_time_str(when)
    interval = parse_interval(how_often)

    if not remind_at:
        messages.append((interaction, "Invalid time format. Use something like `13:00` or `1PM` (PST)."))
        return

    if guild_id not in reminders:
        reminders[guild_id] = []

    reminders[guild_id].append({
        "channel_id": channel_id,
        "msg": msg,
        "when": remind_at,
        "how_often": interval,
    })

    save_reminders()

    messages.append((interaction, f"Reminder set for **{remind_at.strftime('%Y-%m-%d %H:%M %Z')}**" + (f" repeating every {how_often}" if how_often else "")))


def check_reminder():
    """Check reminders and send messages when due."""
    global reminders

    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    changed = False

    for guild_id, reminder_list in list(reminders.items()):
        for reminder in list(reminder_list):
            remind_time = reminder["when"]
            channel_id = reminder["channel_id"]
            msg = reminder["msg"]
            how_often = reminder.get("how_often")

            if remind_time <= now:
                messages.append((type("TempInteraction", (), {"channel_id": channel_id})(), msg))
                
                if how_often:
                    reminder["when"] = remind_time + how_often
                    changed = True
                else:
                    reminder_list.remove(reminder)
                    changed = True

        # Clean up empty lists
        if not reminder_list:
            reminders.pop(guild_id, None)

    if changed:
        save_reminders()
