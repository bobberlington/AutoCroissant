"""
The /reminders panel: one reminder per card (only the invoker sees it), with Previous/Next, a jump select and
Reschedule, Turn off/on, Delete and New. The logic and the rights live in commands/analytics.py; this module is the
Discord UI (not compiled: discord.ui subclasses).

It answers each interaction directly (not through the queues): a View, a modal and edit_message need the live
interaction. Every click re-reads the list by reminder id, so a reminder that fired or was deleted elsewhere shows
up as it is now. Panels left open across a restart stop working (run /reminders again).
"""
from asyncio import to_thread
from datetime import datetime
from traceback import format_exc

from discord import AllowedMentions, ButtonStyle, Color, Embed, HTTPException, Interaction, SelectOption, TextStyle
from discord.ui import Button, Modal, Select, TextInput, View

import commands.analytics as rem

# ========================
# Configuration
# ========================
COLORS = {"on": Color.green(), "off": Color.light_grey(), "problem": Color.red()}
PAGE_PREV, PAGE_NEXT = "page:-1", "page:+1"
CLOSED = "Closed. Run /reminders again."


def _upper(text):
    text = str(text)
    return text[:1].upper() + text[1:]


def _act(guild_id, action, args):
    """Run a change, then read the list again (one thread hop). Returns (result or the ReminderError, snapshot)."""
    try:
        result = action(*args)
    except rem.ReminderOff:
        raise
    except rem.ReminderError as e:
        result = e
    return result, rem.snapshot(guild_id)


# ========================
# Panel
# ========================
class ReminderPanel(View):
    """One /reminders panel. State is the shown reminder's id (the list is re-read on every click)."""

    def __init__(self, interaction: Interaction):
        super().__init__(timeout=rem.PANEL_TIMEOUT)
        self.guild_id = interaction.guild_id
        self.owner_id = interaction.user.id
        self.admin = rem.is_admin(interaction)
        self.current_id = None
        self.index = 0
        self.anchor = None                      # an index on the select page being shown; None: the current one's
        self.confirming = False
        self.shown = None                       # copy of the reminder on the card
        self.embed = None
        self.last_interaction = interaction

    # --- rendering ---
    def start(self, snap, channel_id):
        """Open on the first reminder in this channel, else the first."""
        for reminder in snap:
            if reminder.get("channel_id") == channel_id:
                self.current_id = reminder["id"]
                return
        self.current_id = snap[0]["id"] if snap else None

    def render(self, snap, notice=""):
        """Rebuild the card and the components for snap. Returns the send/edit keyword arguments."""
        ids = [r["id"] for r in snap]
        if self.current_id is not None and self.current_id not in ids and not notice:
            notice = "That reminder is gone."
        self.index = rem.locate(ids, self.current_id, self.index)
        self.shown = snap[self.index] if snap else None
        self.current_id = self.shown["id"] if self.shown else None
        self.clear_items()
        if self.shown is None:
            self.confirming = False
            embed = Embed(title="Reminders", description="No reminders in this server.", color=Color.light_grey())
            footer = rem.ZONE_LABEL
            self._button("New", ButtonStyle.success, "rem:new", self.on_new, 0, not self.admin)
        else:
            texts = rem.card(self.shown, datetime.now(rem.TIMEZONE))
            embed = Embed(title=texts["title"], description=texts["description"], color=COLORS[texts["state"]])
            for name, value in texts["fields"]:
                embed.add_field(name=name, value=value, inline=True)
            footer = f"{self.index + 1} of {len(snap)} · {rem.ZONE_LABEL}"
            self._nav(snap)
            self._actions()
        line = rem.status()
        if line:
            footer += f" · {line}"
        embed.set_footer(text=footer[:2048])
        self.embed = embed
        return {"content": notice or None, "embed": embed, "view": self, "allowed_mentions": AllowedMentions.none()}

    def _button(self, label, style, custom_id, callback, row, disabled=False):
        button = Button(label=label, style=style, custom_id=custom_id, row=row, disabled=disabled)
        button.callback = callback
        self.add_item(button)

    def _nav(self, snap):
        n = len(snap)
        locked = self.confirming
        self._button("◀", ButtonStyle.secondary, "rem:prev", self.on_prev, 0, locked or n < 2)
        self._button("↻", ButtonStyle.secondary, "rem:refresh", self.on_refresh, 0, locked)
        self._button("▶", ButtonStyle.secondary, "rem:next", self.on_next, 0, locked or n < 2)
        start, stop, has_prev, has_next = rem.select_page(n, self.index if self.anchor is None else self.anchor)
        options = []
        if has_prev:
            options.append(SelectOption(label="◀ earlier", value=PAGE_PREV))
        for reminder in snap[start:stop]:
            label, description = rem.option_texts(reminder)
            options.append(SelectOption(label=label, value=reminder["id"], description=description,
                                        default=reminder["id"] == self.current_id))
        if has_next:
            options.append(SelectOption(label="later ▶", value=PAGE_NEXT))
        select = Select(custom_id="rem:jump", placeholder="Jump to a reminder", options=options, row=1,
                        disabled=locked)
        select.callback = self.on_jump
        self.add_item(select)

    def _actions(self):
        if self.confirming:
            self._button(f"Delete {self.current_id}", ButtonStyle.danger, "rem:confirm", self.on_confirm, 2)
            self._button("Cancel", ButtonStyle.secondary, "rem:cancel", self.on_cancel, 2)
            return
        editable = rem.can_edit(self.shown, self.owner_id, self.admin)
        self._button("Reschedule", ButtonStyle.primary, "rem:resched", self.on_reschedule, 2, not editable)
        if self.shown.get("enabled", True):
            self._button("Turn off", ButtonStyle.secondary, "rem:off", self.on_off, 2, not editable)
        else:
            self._button("Turn on", ButtonStyle.success, "rem:on", self.on_on, 2, not editable)
        self._button("Delete", ButtonStyle.danger, "rem:delete", self.on_delete, 2, not editable)
        self._button("New", ButtonStyle.success, "rem:new", self.on_new, 2, not self.admin)

    # --- plumbing ---
    async def interaction_check(self, interaction: Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("Use /reminders.", ephemeral=True)
            return False
        self.track(interaction)
        return True

    def track(self, interaction):
        self.last_interaction = interaction
        self.admin = rem.is_admin(interaction)

    async def show(self, interaction, snap, notice=""):
        await interaction.response.edit_message(**self.render(snap, notice))

    async def off(self, interaction, e):
        """Reminders went off: say so and close."""
        self.stop()
        await interaction.response.edit_message(content=str(e), embed=None, view=None)

    async def refresh(self, interaction, notice=""):
        try:
            snap = await to_thread(rem.snapshot, self.guild_id)
        except rem.ReminderOff as e:
            return await self.off(interaction, e)
        await self.show(interaction, snap, notice)

    async def change(self, interaction, action, args, done, failed="✗ "):
        """Run a change in a thread and show the result; done(result) is the success notice."""
        try:
            result, snap = await to_thread(_act, self.guild_id, action, args)
        except rem.ReminderOff as e:
            return await self.off(interaction, e)
        if isinstance(result, rem.ReminderError):
            notice = failed + (_upper(result) if failed == "✗ " else str(result))
        else:
            notice = done(result)
        await self.show(interaction, snap, notice)

    async def on_timeout(self):
        if self.embed is not None:
            self.embed.set_footer(text=CLOSED)
        try:
            await self.last_interaction.edit_original_response(embed=self.embed, view=None)
        except HTTPException:
            pass

    async def on_error(self, interaction, error, item):
        print(f"ERROR: /reminders panel ({getattr(item, 'custom_id', item)}):\n{format_exc()}")
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message("✗ Something went wrong; run /reminders again.", ephemeral=True)
        except HTTPException:
            pass

    # --- callbacks ---
    async def on_prev(self, interaction):
        await self.move(interaction, -1)

    async def on_next(self, interaction):
        await self.move(interaction, 1)

    async def move(self, interaction, delta):
        try:
            snap = await to_thread(rem.snapshot, self.guild_id)
        except rem.ReminderOff as e:
            return await self.off(interaction, e)
        ids = [r["id"] for r in snap]
        if self.current_id in ids:              # else the card says it is gone and shows its neighbour
            self.index = rem.step(len(ids), ids.index(self.current_id), delta)
            self.current_id = ids[self.index]
        self.anchor = None
        await self.show(interaction, snap)

    async def on_refresh(self, interaction):
        await self.refresh(interaction)

    async def on_jump(self, interaction):
        value = (interaction.data or {}).get("values", [""])[0]
        try:
            snap = await to_thread(rem.snapshot, self.guild_id)
        except rem.ReminderOff as e:
            return await self.off(interaction, e)
        if value in (PAGE_PREV, PAGE_NEXT):
            start, stop, _, _ = rem.select_page(len(snap), self.index if self.anchor is None else self.anchor)
            self.anchor = start - 1 if value == PAGE_PREV else stop
        else:
            self.current_id, self.anchor = value, None
        await self.show(interaction, snap)

    async def on_reschedule(self, interaction):
        if self.shown is None or not rem.can_edit(self.shown, self.owner_id, self.admin):
            return await self.refresh(interaction, "✗ Only an admin or the reminder's creator can change it.")
        await interaction.response.send_modal(RescheduleModal(self, self.shown))

    async def on_off(self, interaction):
        await self.toggle(interaction, False)

    async def on_on(self, interaction):
        await self.toggle(interaction, True)

    async def toggle(self, interaction, enabled):
        # the state the button showed, not a flip: a stale panel never undoes itself
        await self.change(interaction, rem.set_enabled, (interaction, self.current_id, enabled),
                          lambda result: f"✓ {result[1]}")

    async def on_delete(self, interaction):
        self.confirming = True
        await self.refresh(interaction)

    async def on_cancel(self, interaction):
        self.confirming = False
        await self.refresh(interaction)

    async def on_confirm(self, interaction):
        self.confirming = False
        await self.change(interaction, rem.delete, (interaction, self.current_id),
                          lambda result: f"✓ Deleted `{result['id']}`.")

    async def on_new(self, interaction):
        if not self.admin:
            return await self.refresh(interaction, "✗ Only admins can add reminders.")
        await interaction.response.send_modal(NewModal(self))


# ========================
# Modals
# ========================
class RescheduleModal(Modal):
    def __init__(self, panel: ReminderPanel, reminder: dict):
        super().__init__(title=f"Reschedule {reminder['id']}"[:45], timeout=rem.PANEL_TIMEOUT,
                         custom_id=f"rem:modal:resched:{reminder['id']}")
        self.panel = panel
        self.reminder_id = reminder["id"]
        when = reminder.get("when")
        frequency = reminder.get("frequency")
        self.when = TextInput(label=f"When ({rem.ZONE_LABEL})"[:45], max_length=100,
                              default=when.strftime("%Y-%m-%d %H:%M") if isinstance(when, datetime) else None,
                              placeholder="13:00, sun 3AM, tomorrow 9AM, 2026-12-25 9AM, or a delay like 30m")
        self.repeat = TextInput(label="Repeat every (blank = once)", required=False, max_length=50,
                                default=rem.format_interval(frequency) if frequency else None,
                                placeholder="1d, 1w, 1d12h")
        self.add_item(self.when)
        self.add_item(self.repeat)

    async def on_submit(self, interaction: Interaction):
        self.panel.track(interaction)
        self.panel.current_id = self.reminder_id

        def done(result):
            return " ".join(["✓ Rescheduled."] + result[1])
        await self.panel.change(interaction, rem.reschedule,
                                (interaction, self.reminder_id, self.when.value, self.repeat.value), done,
                                failed="✗ Not rescheduled: ")


class NewModal(Modal):
    def __init__(self, panel: ReminderPanel):
        super().__init__(title="New reminder in this channel", timeout=rem.PANEL_TIMEOUT, custom_id="rem:modal:new")
        self.panel = panel
        self.msg = TextInput(label="Message", style=TextStyle.paragraph, required=False, max_length=2000)
        self.command = TextInput(label="Command (runs with your permissions)", required=False, max_length=1000,
                                 placeholder="/update_stats force_update:True")
        self.when = TextInput(label=f"When ({rem.ZONE_LABEL})"[:45], required=False, max_length=100,
                              placeholder="13:00, sun 3AM, tomorrow 9AM, 2026-12-25 9AM")
        self.offset = TextInput(label="Offset (delay added to the first run)", required=False, max_length=50,
                                placeholder="30m, 1d")
        self.repeat = TextInput(label="Repeat every (blank = once)", required=False, max_length=50,
                                placeholder="1d, 1w, 1d12h")
        for item in (self.msg, self.command, self.when, self.offset, self.repeat):
            self.add_item(item)

    async def on_submit(self, interaction: Interaction):
        self.panel.track(interaction)

        def done(result):
            reminder, notes = result
            self.panel.current_id, self.panel.anchor = reminder["id"], None
            return " ".join([f"✓ Reminder `{reminder['id']}` added."] + notes)
        await self.panel.change(interaction, rem.create,
                                (interaction, self.msg.value, self.when.value, self.offset.value, self.repeat.value,
                                 self.command.value), done, failed="✗ Reminder NOT set: ")


# ========================
# Entry Point
# ========================
async def open_panel(interaction: Interaction):
    """/reminders: the panel, on the first reminder in this channel."""
    if interaction.guild_id is None:
        await interaction.response.send_message("Reminders work in servers only.", ephemeral=True)
        return
    try:
        snap = await to_thread(rem.snapshot, interaction.guild_id)
    except rem.ReminderOff as e:
        await interaction.response.send_message(str(e), ephemeral=True)
        return
    panel = ReminderPanel(interaction)
    panel.start(snap, interaction.channel_id)
    await interaction.response.send_message(ephemeral=True, **panel.render(snap))


__all__ = ['open_panel', 'ReminderPanel']
