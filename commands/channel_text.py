from collections import defaultdict
from datetime import datetime, timedelta
from discord import Interaction
from pickle import load, dump
from shlex import split as shlex_split
from uuid import uuid4
from zoneinfo import ZoneInfo

from commands.utils import slash_registry, queue_message, queue_command, make_fake_interaction, convert_value
from global_config import REMIND_PKL

reminders = defaultdict(list)


def save_reminders():
    with open(REMIND_PKL, "wb") as f:
        dump(reminders, f)


def init_reminder():
    global reminders
    print(f"Trying to open {REMIND_PKL}.")
    try:
        with open(REMIND_PKL, 'rb') as f:
            reminders = load(f)
        print(f"Existing dict found in {REMIND_PKL}, updating entries...")
    except (EOFError, FileNotFoundError):
        print(f"{REMIND_PKL} is empty or missing.")


def parse_time_str(when_str: str) -> datetime | None:
    when_str = when_str.strip().upper().replace(" ", "")
    for fmt in ("%H:%M", "%I%p", "%I:%M%p"):
        try:
            pst = ZoneInfo("America/Los_Angeles")
            return datetime.combine(datetime.now(pst).date(), datetime.strptime(when_str, fmt).time(), tzinfo=pst)
        except ValueError:
            continue
    return None


def parse_interval(time_str: str) -> timedelta | None:
    if not time_str:
        return None
    time_str = time_str.strip().lower()
    try:
        val = int(time_str[:-1])
        unit = time_str[-1]
        return {
            "s": timedelta(seconds=val),
            "m": timedelta(minutes=val),
            "h": timedelta(hours=val),
            "d": timedelta(days=val),
            "w": timedelta(weeks=val)
        }.get(unit)
    except Exception:
        return None


def set_reminder(interaction: Interaction, msg: str = "", when: str = "",
                 offset: str = "", frequency: str = "", command: str = ""):
    remind_at = parse_time_str(when)
    offset_delta = parse_interval(offset)
    interval = parse_interval(frequency)

    if not remind_at:
        return queue_message(interaction, "Invalid time format. Try `13:00` or `1PM` (PST).")

    if offset_delta:
        remind_at += offset_delta

    reminder = {
        "id": str(uuid4())[:8],
        "channel_id": interaction.channel_id,
        "msg": msg,
        "when": remind_at,
        "frequency": interval,
        "command": command.strip() or None,
    }

    reminders[interaction.guild_id].append(reminder)
    save_reminders()

    desc = f"Reminder set for **{remind_at.strftime('%Y-%m-%d %H:%M %Z')}**"
    if offset:
        desc += f" (offset {offset})"
    if frequency:
        desc += f", repeats every {frequency}"
    if command:
        desc += f", runs `{command}`"
    queue_message(interaction, desc)


def parse_named_args(parts):
    """
    Split a list like ['prompt:sunset', 'height:512'] into args=[], kwargs={...},
    converting numeric and boolean values automatically.
    """
    args = []
    kwargs = {}
    for p in parts:
        if ":" in p:
            key, value = p.split(":", 1)
            kwargs[key] = convert_value(value)
        else:
            args.append(convert_value(p))
    return args, kwargs


def check_reminder():
    """Trigger reminders and execute queued async/sync slash commands."""
    global reminders
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    changed = False

    for guild_id, reminder_list in list(reminders.items()):
        for reminder in reminder_list[:]: # copy since we may modify it
            if reminder["when"] <= now:
                channel_id = reminder["channel_id"]
                msg = reminder["msg"]
                cmd_text = reminder.get("command")

                # Create a fake interaction for queued sends
                fake_interaction = make_fake_interaction(channel_id, guild_id)

                # 1. Queue message send
                if msg:
                    queue_message(fake_interaction, msg)

                # 2. Execute stored slash command (if any)
                if cmd_text:
                    try:
                        parts = shlex_split(cmd_text)
                    except ValueError:
                        queue_message(fake_interaction, f"Could not parse command: `{cmd_text}`")
                        continue

                    if not parts:
                        continue

                    cmd_name = parts[0].lstrip("/")
                    raw_args = parts[1:]
                    args, kwargs = parse_named_args(raw_args)
                    func = slash_registry.get(cmd_name)

                    if func:
                        # Queue the function for deferred execution
                        queue_command(func, fake_interaction, *args, **kwargs)
                        # Optionally notify about command dispatch:
                        # queue_message(fake_interaction, f"Executing `{cmd_text}`")
                    else:
                        queue_message(fake_interaction, f"Unknown command: `{cmd_name}`")

                # 3. Repeat or remove reminder
                if reminder["frequency"]:
                    reminder["when"] = reminder["when"] + reminder["frequency"]
                else:
                    reminder_list.remove(reminder)
                changed = True

        # Clean up empty guild reminder lists
        if not reminder_list:
            reminders.pop(guild_id, None)

    if changed:
        save_reminders()


def list_reminders(interaction: Interaction, all: bool = False, hidden: bool = False):
    """List reminders for the current channel or entire server (includes commands and channel info)."""
    guild_id = interaction.guild_id
    channel_id = interaction.channel_id

    if guild_id not in reminders or len(reminders[guild_id]) == 0:
        return queue_message(interaction, "No reminders set for this server.", ephemeral=hidden)

    if all:
        filtered = reminders[guild_id]
        scope_text = "this server"
    else:
        filtered = [r for r in reminders[guild_id] if r["channel_id"] == channel_id]
        scope_text = f"this channel (<#{channel_id}>)"

    if not filtered:
        return queue_message(interaction, f"No reminders found for {scope_text}.", ephemeral=hidden)

    lines = []
    for r in filtered:
        when_str = r["when"].strftime("%Y-%m-%d %H:%M %Z")
        repeat_str = f", repeats every {r['frequency']}" if r["frequency"] else ""
        cmd_str = f", runs `{r['command']}`" if r.get("command") else ""
        msg_str = f"**{r['msg']}**" if r["msg"] else "*<no message>*"
        chan_str = f" in <#{r['channel_id']}>" if all else ""

        lines.append(f"`{r['id']}` — {msg_str} at {when_str}{repeat_str}{cmd_str}{chan_str}")

    output = "\n".join(lines)
    queue_message(interaction, f"Reminders for {scope_text}:\n{output}", ephemeral=hidden)


def remove_reminder(interaction: Interaction, reminder_id: str = ""):
    """Remove a reminder by its ID."""
    global reminders

    guild_id = interaction.guild_id

    if guild_id not in reminders:
        return queue_message(interaction, f"No reminders found for this server.")

    removed = False
    for reminder in list(reminders[guild_id]):
        if reminder["id"] == reminder_id:
            reminders[guild_id].remove(reminder)
            removed = True
            break

    if guild_id in reminders and len(reminders[guild_id]) == 0:
        reminders.pop(guild_id)

    if removed:
        save_reminders()
        queue_message(interaction, f"Reminder `{reminder_id}` has been removed.")
    else:
        queue_message(interaction, f"No reminder found with ID `{reminder_id}`.")
