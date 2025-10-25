from discord import Interaction, TextChannel, Forbidden, ClientException

from commands.utils import queue_message, split_long_message


# ========================
# Guild Management
# ========================
async def purge(interaction: Interaction,
                limit : int,
                id: int,
                bulk: bool = False):
    """
    Deletes a bunch of messages from the channel this command is invoked from.

    Args:
        interaction: Discord interaction
        limit: Number of messages to delete (-1 means infinite)
        id: ID of the user who you want to delete messages from
        bulk: Delete messages faster, but requires 'manage messages' permission
    """
    if limit == -1:
        limit = 1000000

    channel = interaction.channel
    await interaction.followup.send(f"Purging messages from user {id}.")
    num_messages = len(await channel.purge(limit=limit, check=lambda message: message.author.id == id, bulk=bulk))
    await channel.send(f"Deleted {num_messages} message(s) from user {id}.")


async def list_guilds(interaction: Interaction) -> None:
    """
    List all guilds the bot is currently a member of.

    Args:
        interaction: Discord interaction
    """
    guilds = interaction.client.guilds
    if not guilds:
        await interaction.response.send_message("Bot is not in any guilds.")
        return

    guild_list = "\n".join(
        f"• {guild.name} (ID: {guild.id}, Members: {guild.member_count})"
        for guild in guilds
    )

    message_chunks = split_long_message(
        f"**Bot is in {len(guilds)} guild(s):**\n```\n{guild_list}\n```"
    )

    await interaction.response.send_message(message_chunks[0])
    for chunk in message_chunks[1:]:
        await interaction.followup.send(chunk)


async def leave_guild(interaction: Interaction, guild_id: str) -> None:
    """
    Force the bot to leave a specified guild.

    Args:
        interaction: Discord interaction
        guild_id: ID of the guild to leave
    """
    guild = interaction.client.get_guild(int(guild_id))
    if guild is None:
        await interaction.response.send_message(f"Guild with ID `{guild_id}` not found or bot is not a member.")
        return

    await guild.leave()
    await interaction.response.send_message(f"Left guild: **{guild.name}** (ID: {guild_id})")


async def sync_commands_global(interaction: Interaction, tree, client) -> None:
    """
    Sync all slash commands globally and per guild.

    Args:
        interaction: Discord interaction
        tree: Command tree to sync
        client: Discord client instance
    """
    await interaction.response.send_message("Resyncing all commands globally and per guild...")

    # Clear and sync global commands
    tree.clear_commands(guild=None)
    await tree.sync(guild=None)

    # Sync per-guild commands
    guild_count = 0
    for guild in client.guilds:
        tree.copy_global_to(guild=guild)
        await tree.sync(guild=guild)
        guild_count += 1

    queue_message(interaction, f"Commands synced globally and to {guild_count} guilds.")


# ========================
# Guild Inspection
# ========================
async def list_guild_members(interaction: Interaction, guild_id: str) -> None:
    """
    List all members in a guild by its ID.

    Args:
        interaction: Discord interaction
        guild_id: ID of the guild
    """
    guild = interaction.client.get_guild(int(guild_id))
    if guild is None:
        await interaction.response.send_message(f"Guild with ID `{guild_id}` not found or bot is not a member.")
        return

    # Defer response as this might take time
    await interaction.response.defer(ephemeral=True, thinking=False)

    try:
        # Force member cache fill if not loaded
        await guild.chunk()

        # Build member list
        member_list = [
            f"• {member.name}#{member.discriminator} (ID: {member.id})"
            for member in guild.members
        ]

        message_chunks = split_long_message(
            f"**Members in {guild.name} ({len(member_list)} total):**\n"
            + "\n".join(member_list)
        )

        for chunk in message_chunks:
            await interaction.followup.send(chunk)
    except ClientException:
        queue_message(interaction, "Intents.members must be enabled to use this.")


async def list_guild_channels(interaction: Interaction, guild_id: str) -> None:
    """
    List all channels in a guild by its ID.

    Args:
        interaction: Discord interaction
        guild_id: ID of the guild
    """
    guild = interaction.client.get_guild(int(guild_id))
    if guild is None:
        await interaction.response.send_message(f"Guild with ID `{guild_id}` not found or bot is not a member.")
        return

    channels = []
    for channel in guild.channels:
        channel_type = str(channel.type).replace('_', ' ').title()
        channels.append(f"[{channel_type}] {channel.name} (ID: {channel.id})")

    message_chunks = split_long_message(
        f"**Channels in {guild.name} ({len(channels)} total):**\n"
        + "\n".join(channels)
    )

    await interaction.response.send_message(message_chunks[0])
    for chunk in message_chunks[1:]:
        await interaction.followup.send(chunk)


async def get_channel_messages(
    interaction: Interaction,
    guild_id: str,
    channel_id: str,
    limit: int = 50) -> None:
    """
    Get the last X messages from a specific channel in a guild.

    Args:
        interaction: Discord interaction
        guild_id: ID of the guild
        channel_id: ID of the channel
        limit: Number of messages to fetch (-1 means infinite)
    """
    if limit == -1:
        limit = 1000000

    guild = interaction.client.get_guild(int(guild_id))
    if guild is None:
        await interaction.response.send_message(f"Guild with ID `{guild_id}` not found or bot is not in that guild.")
        return

    channel = guild.get_channel(int(channel_id))
    if channel is None:
        await interaction.response.send_message(f"Channel with ID `{channel_id}` not found in guild **{guild.name}**.")
        return

    if not isinstance(channel, TextChannel):
        await interaction.response.send_message("That channel is not a text channel.")
        return

    # Defer as fetching messages may take time
    await interaction.response.defer(thinking=False, ephemeral=True)

    try:
        # Fetch messages
        messages = []
        async for msg in channel.history(limit=limit):
            author = f"{msg.author.name}#{msg.author.discriminator}"
            timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
            content = msg.content or "[No text content]"

            # Include attachment info
            if msg.attachments:
                attachment_info = ", ".join(f"[{att.filename}]" for att in msg.attachments)
                content += f" {attachment_info}"
            messages.append(f"[{timestamp}] {author}: {content}")

        if not messages:
            await interaction.followup.send(f"No messages found in {channel.mention}.")
            return

        # Build output (reverse to show oldest first)
        message_chunks = split_long_message(
            f"Last {len(messages)} messages from #{channel.name} "
            f"in {guild.name}:\n\n" + "\n".join(reversed(messages))
        )

        for chunk in message_chunks:
            await interaction.followup.send(chunk)
    except Forbidden:
        queue_message(interaction, "I don't have permission to read messages in that channel.")


# ========================
# Module Exports
# ========================
__all__ = [
    'list_guilds',
    'leave_guild',
    'sync_commands_global',
    'list_guild_members',
    'list_guild_channels',
    'get_channel_messages',
    'purge',
]
