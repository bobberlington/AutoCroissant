from cv2 import resize, INTER_AREA
from discord import Interaction
from numpy import ndarray, mean
from typing import Optional

from commands.query_card import card_repo
from commands.utils import (
    url_to_cv2image,
    cv2discordfile,
    queue_message,
    queue_file,
)


class CardMerger:
    """Handles merging multiple card images into a single composite image."""
    def __init__(self):
        self.card_repo = card_repo

    def find_card_image(self, query: str) -> tuple[Optional[ndarray], Optional[str]]:
        """
        Find and download a card image by name.

        Args:
            query: Card name to search for

        Returns:
            Tuple of (image array, card filename) or (None, None) if not found
        """
        # Find closest match
        closest = card_repo._find_closest_match(query)
        if not closest:
            return None, None

        # Download image
        url = card_repo.get_card_url(closest)
        image = url_to_cv2image(url)
        return image, closest

    def merge_cards_horizontal(
        self,
        images: list[ndarray],
        blend_mode: str = "slice"
    ) -> ndarray:
        """
        Merge multiple card images horizontally.

        Args:
            images: List of card images to merge
            blend_mode: How to blend images ("slice", "average")

        Returns:
            Merged image array
        """
        if not images:
            raise ValueError("No images provided to merge")

        if len(images) == 1:
            return images[0]

        # Use last image as base (determines final dimensions)
        base_image = images[len(images) - 1]
        height, width = base_image.shape[:2]

        # Resize all images to match base dimensions
        resized_images = [
            resize(img, (width, height), interpolation=INTER_AREA)
            for img in images
        ]

        # Merge based on blend mode
        if blend_mode == "slice":
            return self._merge_slice(resized_images, height, width)
        elif blend_mode == "average":
            return self._merge_average(resized_images)

    def _merge_slice(self, images: list[ndarray], height: int, width: int) -> ndarray:
        """
        Merge images by slicing them horizontally into equal parts.

        Each image contributes a horizontal slice from top to bottom.
        """
        result = images[len(images) - 1].copy()
        total_images = len(images)
        slice_height = height // total_images

        for idx, image in enumerate(reversed(images)):
            y_end = slice_height * (total_images - idx)
            result[0:y_end] = image[0:y_end]

        return result

    def _merge_average(self, images: list[ndarray]) -> ndarray:
        """
        Merge images by averaging pixel values.

        Creates a ghostly blend of all images.
        """
        return mean(images, axis=0).astype(images[0].dtype)

merger = CardMerger()


def frankenstein(
    interaction: Interaction,
    cards: str,
    blend_mode: str = "slice"
) -> None:
    """
    Create a Frankenstein's monster by merging multiple card images.

    Args:
        interaction: Discord interaction
        cards: Comma-separated list of card names
        blend_mode: How to blend images ("slice", "average")
    """
    # Validate input
    card_names = [name.strip() for name in cards.split(",") if name.strip()]

    if len(card_names) == 0:
        queue_message(interaction, "Must specify at least one card to frankenstein.")
        return

    # Find and download all card images
    images: list[ndarray] = []
    ambiguous_warnings: list[str] = []

    for card_name in card_names:
        image, closest = merger.find_card_image(card_name)

        if image is None:
            queue_message(interaction, f"No card found matching '{card_name}'.")
            return

        images.append(image)

        # Check for ambiguous names
        if closest in card_repo.ambiguous_names:
            options = "\n".join(card_repo.ambiguous_names[closest])
            ambiguous_warnings.append(f"Ambiguous name for '{card_name}'. If this wasn't the card you wanted, try:\n{options}")

    # Merge images
    merged_image = merger.merge_cards_horizontal(images, blend_mode)
    queue_file(interaction, cv2discordfile(merged_image, "frankenstein.png"))

    queue_message(interaction, "\n\n".join(ambiguous_warnings))


# ========================
# Module Exports
# ========================
__all__ = [
    'frankenstein',
]
