from datetime import datetime, timedelta
from discord import Interaction
from pickle import load, dump
from uuid import uuid4
from zoneinfo import ZoneInfo

from commands.utils import messages
from global_config import REMIND_PKL

reminders = {}  # {guild_id: [ {id, channel_id, msg, when, how_often} ]}


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

    remind_at = parse_time_str(when)

    if not remind_at:
        messages.append((interaction, "Invalid time format. Use something like `13:00` or `1PM` (PST)."))
        return

    reminder_id = str(uuid4())[:8]
    guild_id = interaction.guild_id
    channel_id = interaction.channel_id
    interval = parse_interval(how_often)

    if guild_id not in reminders:
        reminders[guild_id] = []

    reminders[guild_id].append({
        "id": reminder_id,
        "channel_id": channel_id,
        "msg": msg,
        "when": remind_at,
        "how_often": interval,
    })

    save_reminders()

    messages.append((interaction,
                     f"Reminder set for **{remind_at.strftime('%Y-%m-%d %H:%M %Z')}**"
                     + (f" repeating every {how_often}" if how_often else "")))


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

        if not reminder_list:
            reminders.pop(guild_id, None)

    if changed:
        save_reminders()

def list_reminders(interaction: Interaction, all: bool = False):
    """List reminders for the current channel or entire server."""
    guild_id = interaction.guild_id
    channel_id = interaction.channel_id

    if guild_id not in reminders or len(reminders[guild_id]) == 0:
        messages.append((interaction, "No reminders set for this server."))
        return

    if all:
        filtered = reminders[guild_id]
        scope_text = "this server"
    else:
        filtered = [r for r in reminders[guild_id] if r["channel_id"] == channel_id]
        scope_text = f"this channel (<#{channel_id}>)"

    if not filtered:
        messages.append((interaction, f"No reminders found for {scope_text}."))
        return

    lines = []
    for r in filtered:
        when_str = r["when"].strftime("%Y-%m-%d %H:%M %Z")
        repeat_str = f", repeats every {r['how_often']}" if r["how_often"] else ""
        lines.append(f"`{r['id']}` — **{r['msg']}** at {when_str}{repeat_str}")

    output = "\n".join(lines)
    messages.append((interaction, f"Reminders for {scope_text}:\n{output}"))


def remove_reminder(interaction: Interaction, reminder_id: str = ""):
    """Remove a reminder by its ID."""
    global reminders

    guild_id = interaction.guild_id

    if guild_id not in reminders:
        messages.append((interaction, f"No reminders found for this server."))
        return

    removed = False
    for reminder in list(reminders[guild_id]):
        if reminder["id"] == reminder_id:
            reminders[guild_id].remove(reminder)
            removed = True
            break

    # Clean up empty lists
    if guild_id in reminders and len(reminders[guild_id]) == 0:
        reminders.pop(guild_id)

    if removed:
        save_reminders()
        messages.append((interaction, f"Reminder `{reminder_id}` has been removed."))
    else:
        messages.append((interaction, f"No reminder found with ID `{reminder_id}`."))
