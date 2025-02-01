from cv2 import resize, INTER_AREA
from difflib import get_close_matches
from discord import Interaction
from numpy import ndarray

from commands.utils import url_to_cv2image, cv2discordfile, messages, files

def frankenstein(interaction: Interaction, cards: str):
    from commands.query_card import repository, git_files, git_filenames, ambiguous_names, match_ratio
    if len(cards.split()) < 1:
        interaction.followup.send("Must specify at least one argument to frankenstein.")
        return

    images: list[ndarray] = []
    for part in cards.strip().lower().split(","):
        creature = part.strip().replace(" ", "_")
        if not creature.endswith(".png"):
            creature += ".png"
        try:
            closest = get_close_matches(creature, git_filenames, n=1, cutoff=match_ratio)[0]
        except IndexError:
            messages.append((interaction, f"No card found for query {creature}!"))
            return
        images.append(url_to_cv2image(f"https://raw.githubusercontent.com/{repository}/main/{git_files[closest]}"))

        # If the filename was ambiguous, make a note of that.
        if closest in ambiguous_names:
            ambiguous_message = f"Ambiguous name found for {closest}. If this wasn't the card you wanted, try typing: \n"
            for i in ambiguous_names[closest]:
                ambiguous_message += f"{i}\n"
            messages.append((interaction, ambiguous_message))

    frankensteins_monster = images[-1]
    total_images = len(images)
    cur_image = total_images
    height = frankensteins_monster.shape[0]
    width = frankensteins_monster.shape[1]
    height_part = int(height / total_images)
    for image in reversed(images):
        image = resize(image, (width, height), interpolation=INTER_AREA)
        frankensteins_monster[0:height_part * cur_image] = image[0:height_part * cur_image]
        cur_image -= 1

    files.append((interaction, cv2discordfile(frankensteins_monster)))
