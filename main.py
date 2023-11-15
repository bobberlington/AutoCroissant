import discord
from discord import app_commands
import config
import aliases
import requests
import difflib

intents = discord.Intents.default()
intents.message_content = True


client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.content.startswith('?'):
        # If it's just a ?, ignore everything
        if len(message.content) == 1:
            return
        # Grab repo
        repo = requests.get("https://api.github.com/repos/MichaelJSr/TTSCardMaker/git/trees/main?recursive=1")
        text = repo.json()
        file_alias = aliases.aliases
        files = {}
        # Make a dictionary of all list of pngs in the github
        for i in text["tree"]:
            if ".png" in i["path"]:
                filepath = i["path"]
                files[filepath[filepath.rindex("/") + 1:]] = filepath.replace(" ", "%20")
        # Fill up the aliases
        # keys of aliases are the aliases, values are the filenames they point to
        for i in file_alias.keys():
            files[i] = files[file_alias[i]]
        filenames = list(files.keys())
        card = message.content[1:].replace(" ", "_")
        try:
            closest = difflib.get_close_matches(card, filenames, n=1, cutoff=0.3)[0]
        except IndexError:
            await message.channel.send("No card found!")
            return
        await message.channel.send(f"https://raw.githubusercontent.com/MichaelJSr/TTSCardMaker/main/{files[closest]}")

client.run(config.token)
