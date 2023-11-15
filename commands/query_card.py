import aliases
import requests
import difflib

async def query_card(message):
    # If it's just a ?, ignore everything
    if len(message.content) == 1:
        return

    # Grab repo
    repo = requests.get("https://api.github.com/repos/MichaelJSr/TTSCardMaker/git/trees/main?recursive=1")
    if repo.status_code != 200:
        await message.channel.send(f"Error {repo.status_code} when requesting github.")
        return

    text = repo.json()
    file_alias = aliases.aliases
    files = {}
    ambiguous_names = {}
    # Make a dictionary of all list of pngs in the github
    # Keys consist of the filename, for example Bomb.png
    # Values consist of the whole path, for example Items/Attack/2 Stars/Bomb.png
    for i in text["tree"]:
        if i["path"].endswith(".png"):
            filepath = i["path"]
            png_filename = filepath[filepath.rindex("/") + 1:].lower()
            # If we're putting the same name twice, it will change both names to include the top level folder
            if png_filename in files:
                new_filename = f"{filepath[0:filepath.index('/')]}/{png_filename}".lower()
                old_filename = f"{files[png_filename][0:files[png_filename].index('/')]}/{png_filename}".lower()
                files[new_filename] = filepath.replace(" ", "%20")
                files[old_filename] = files[png_filename]

                # Mark which files are ambiguous
                if png_filename in ambiguous_names.keys():
                    if old_filename not in ambiguous_names[png_filename]:
                        ambiguous_names[png_filename].append(old_filename)
                    ambiguous_names[png_filename].append(new_filename)
                else:
                    ambiguous_names[png_filename] = []
                    ambiguous_names[png_filename].append(old_filename)
                    ambiguous_names[png_filename].append(new_filename)
            else:
                files[png_filename] = filepath.replace(" ", "%20")

    # Fill up the aliases
    # keys of aliases are the aliases, values are the filenames they point to
    for i in file_alias.keys():
        files[f"{i}.png".lower()] = files[file_alias[i].lower()]

    filenames = list(files.keys())
    card = message.content[1:].replace(" ", "_").lower()
    if not card.endswith(".png"):
        card += ".png"
    try:
        closest = difflib.get_close_matches(card, filenames, n=1, cutoff=0.6)[0]
    except IndexError:
        await message.channel.send("No card found!")
        return
    await message.channel.send(f"https://raw.githubusercontent.com/MichaelJSr/TTSCardMaker/main/{files[closest]}")

    # If the filename was ambiguous, make a note of that.
    if closest in ambiguous_names:
        ambiguous_message = "Ambiguous name found. If this wasn't the card you wanted, try typing: \n"
        for i in ambiguous_names[closest]:
            ambiguous_message += f"{i}\n"
        await message.channel.send(ambiguous_message)
