from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from discord import VoiceClient, PCMVolumeTransformer, Interaction, FFmpegOpusAudio
import logging
from os import listdir, makedirs, sep
from os.path import join, isdir
from pathlib import Path
from random import shuffle
from shutil import rmtree
from typing import Optional
from yt_dlp import YoutubeDL

import config
from commands.utils import (
    BREAK_LEN,
    queue_message,
    queue_command,
    split_long_message,
    recursively_traverse,
    make_simple_fake_interaction,
)

# Suppress FFmpeg process warnings
logging.getLogger('discord.player').setLevel(logging.ERROR)

# ========================
# Configurable Parameters
# ========================
MUSIC_BASE_DIR = "music/"
SKIP_FILES = {".gitignore", ".DS_Store"}

music_queue: deque[str] = deque()
prev_music: deque[str] = deque(maxlen=1000)

executor = ThreadPoolExecutor(max_workers=2)
cookies_from_browser: bool = getattr(config, "vram_usage", False) == "mps"

@dataclass
class MusicState:
    vc: VoiceClient | None = None
    last_channel_id: int | None = None
    loop_song: bool = False
state = MusicState()


# ========================
# YT-DLP Setup
# ========================
ydl_opts = {
    "format":             "bestaudio/best",
    "restrictfilenames":  True,
    "outtmpl":            join(MUSIC_BASE_DIR, "%(title)s-%(id)s.%(ext)s"),
    "keepvideo":          getattr(config, "KEEPVIDEO", True),
    "cookiesfrombrowser": ("safari", None, None, None) if cookies_from_browser else None,
    "quiet":              True,  # Suppress console spam
    "no_warnings":        True,
}
ydl_preprocess_opts = {**ydl_opts, "playlistend": 1}

if getattr(config, "POSTPROCESS", False):
    post_proc = {
        "key":              "FFmpegExtractAudio",
        "preferredcodec":   "mp3",
        "preferredquality": "320",
    }
    ydl_opts["postprocessors"] = [post_proc]
dl_pre = YoutubeDL(ydl_preprocess_opts, auto_init="no_verbose_header")


# ========================
# Helper Functions
# ========================
def send_to_last_channel(message: str) -> None:
    """Send a message to the last active channel."""
    if state.last_channel_id:
        queue_message(make_simple_fake_interaction(state.last_channel_id), message)


def queue_local_song(interaction: Interaction, path: Path, play_next: bool) -> None:
    """
    Queue a local song file.

    Args:
        interaction: Discord interaction, or None for internal calls
        path: Path to the song file
        play_next: If True, add to front of queue; otherwise add to back
    """
    (music_queue.appendleft if play_next else music_queue.append)(str(path))
    queue_command(play_next_song)
    if interaction:
        queue_message(interaction, f"Queued song: {path.name}")
    else:
        send_to_last_channel(f"Queued song: {path.name}")


def after_play(err: Optional[Exception]) -> None:
    """
    Callback after a song finishes playing.

    Args:
        err: Exception if playback failed, None otherwise
    """
    queue_command(play_next_song)


def play_next_song() -> None:
    """
    Play the next song in queue if the player is idle.
    Handles looping and automatic queue progression.
    """
    if not state.vc or state.vc.is_playing() or state.vc.is_paused():
        return

    # Loop the last song if enabled
    if state.loop_song and prev_music:
        def _play_loop():
            state.vc.play(FFmpegOpusAudio(source=prev_music[-1], before_options='-nostdin', options='-vn'), after=after_play)
        queue_command(_play_loop)
        return

    if len(music_queue) == 0:
        return

    # Otherwise, play the next queued song
    cur_song = music_queue.popleft()
    if not Path(cur_song).is_file():
        send_to_last_channel(f"File not found: {cur_song}")
        queue_command(play_next_song)  # Try next song
        return

    def _play_next():
        state.vc.play(FFmpegOpusAudio(source=cur_song, before_options='-nostdin', options='-vn'), after=after_play)

    queue_command(_play_next)
    send_to_last_channel(f"Now playing: **{cur_song}**")
    prev_music.append(cur_song)


def queue_folder_songs(interaction: Interaction, folder: str, play_next: bool = False) -> None:
    """
    Queue all valid songs in a folder.

    Args:
        interaction: Discord interaction
        folder: Path to folder containing songs
        play_next: If True, add songs to front of queue
    """
    if not folder.endswith(sep):
        folder += sep
    if not isdir(folder):
        queue_message(interaction, "Folder not found.")
        return

    songs = []
    for s in sorted(listdir(folder)):
        if s in SKIP_FILES:
            continue
        songs.append(s)
        (music_queue.appendleft if play_next else music_queue.append)(join(folder, s))

    # Send all queued songs at once
    if songs:
        queue_msg = "\n".join(songs)
        for part in split_long_message(queue_msg):
            queue_message(interaction, f"Queued songs:\n```{part}```")
    else:
        queue_message(interaction, "No valid songs found in folder.")


# ========================
# Music Functions
# ========================
async def init_vc(interaction: Interaction) -> None:
    """Join user's voice channel if not already connected."""
    if not interaction.user.voice:
        await queue_message(interaction, "You are not in a voice channel.")
        return
    if state.vc is None or not state.vc.is_connected():
        state.vc = await interaction.user.voice.channel.connect()
    state.last_channel_id = interaction.channel_id


def list_all_music(interaction: Interaction) -> None:
    """List all songs recursively."""
    all_music = recursively_traverse(MUSIC_BASE_DIR)
    for part in split_long_message(all_music):
        queue_message(interaction, f"```{part}```")


def play_all(interaction: Interaction) -> None:
    """Queue all songs recursively from MUSIC_BASE_DIR using queue_folder_songs."""
    def _recurse(folder: str) -> None:
        # Queue songs in the current folder
        queue_folder_songs(interaction, folder)

        # Recurse into subfolders
        for item in sorted(listdir(folder)):
            path = join(folder, item)
            if isdir(path):
                _recurse(path)

    _recurse(MUSIC_BASE_DIR)


def play_music(interaction: Interaction, song: str, play_next: bool = False) -> None:
    """
    Handle playback requests for URLs, local files, or folders.
    Automatically downloads YouTube content and queues it when ready.

    Args:
        interaction: Discord interaction
        song: URL, file path, or folder path
        play_next: If True, add to front of queue
    """
    path = Path(song)

    # --- Handle Local Folder ---
    if path.is_dir():
        queue_folder_songs(interaction, str(path), play_next)
        return

    # --- Handle Local File ---
    if path.exists() and path.is_file():
        queue_local_song(interaction, path, play_next)
        return

    # --- Handle URL ---
    if song.startswith("http"):
        if "&list=" not in song:
            pre_info = dl_pre.extract_info(song, download=False)

            # --- Playlist or Mix ---
            if pre_info.get("entries"):
                playlist_title = pre_info.get("title", "playlist")
                folder = Path(MUSIC_BASE_DIR) / playlist_title
                makedirs(folder, exist_ok=True)

                def _playlist_download_hook(d: dict) -> None:
                    if d.get("status") == "finished":
                        queue_local_song(None, Path(d.get("filename")), play_next)

                def _download_playlist() -> None:
                    YoutubeDL({**ydl_opts,
                               "progress_hooks": [_playlist_download_hook],
                               "outtmpl": str(folder / Path(ydl_opts.get("outtmpl")).name)},
                               auto_init="no_verbose_header").extract_info(url=song, download=True)

                queue_message(interaction, f"Starting playlist download: {playlist_title}")
                queue_command(_download_playlist, executor=executor)
                return

        # --- Single Video URL ---
        def _single_download_hook(d: dict) -> None:
            if d.get("status") == "finished":
                queue_local_song(None, Path(d.get("filename")), play_next)

        def _download_song() -> None:
            YoutubeDL({**ydl_opts,
                       "noplaylist": True,
                       "progress_hooks": [_single_download_hook]},
                       auto_init="no_verbose_header").extract_info(url=song, download=True)

        queue_command(_download_song, executor=executor)
        queue_message(interaction, f"Downloading: {song}")
        return

    # --- If nothing matched ---
    queue_message(interaction, f"File or folder not found: {song}")


def replay(interaction: Interaction, rep_index: int = 0) -> None:
    """
    Replay a previously played song.

    Args:
        interaction: Discord interaction
        rep_index: Index of song to replay (0=current, 1=previous, etc.)
    """
    prev_music_len = len(prev_music)
    if prev_music_len > 0:
        if abs(rep_index) >= prev_music_len:
            rep_index = prev_music_len - 1
        queue_message(interaction, f"Replaying song: {prev_music[-1 - rep_index]}")
        music_queue.appendleft(prev_music[-1 - rep_index])
    else:
        queue_message(interaction, "No previously played songs.")


def replay_all(interaction: Interaction) -> None:
    """Replay all previously played songs."""
    if len(prev_music) > 0:
        queue_message(interaction, "Replaying all songs.")
        # Remove duplicates while preserving order
        unique_prev = deque(dict.fromkeys(prev_music))
        music_queue.extend(unique_prev)
        prev_music.clear()
        if music_queue:
            prev_music.append(music_queue[-1])
    else:
        queue_message(interaction, "No previously played songs.")


def set_volume(interaction: Interaction, volume: float) -> None:
    """
    Set the playback volume (note: not supported with FFmpegOpusAudio).

    Args:
        interaction: Discord interaction
        volume: Volume multiplier (0.5=half, 1.0=default, 2.0=double)
    """
    if state.vc and (state.vc.is_playing() or state.vc.is_paused()):
        if state.vc.source.is_opus():
            queue_message(interaction, "Volume control not available with FFmpegOpusAudio. Switch to FFmpegPCMAudio to adjust volume.")
            return
        queue_message(interaction, f"Setting volume: {str(volume)}")
        state.vc.source = PCMVolumeTransformer(state.vc.source, volume=volume)
    else:
        queue_message(interaction, "Not currently playing a song.")


def shuffle_music(interaction: Interaction) -> None:
    """Shuffle the current queue."""
    queue_message(interaction, "Shuffling songs.")
    shuffle(music_queue)


def skip(interaction: Interaction) -> None:
    """Skip the current song."""
    if state.vc and (state.vc.is_playing() or state.vc.is_paused()):
        queue_message(interaction, f"Skipping song: {prev_music[-1] if prev_music else 'unknown'}")
        state.vc.stop()
    else:
        queue_message(interaction, "Not currently playing a song.")


def loop(interaction: Interaction) -> None:
    """Toggle looping of the current song."""
    if state.vc and (state.vc.is_playing() or state.vc.is_paused()):
        state.loop_song = not state.loop_song
        if state.loop_song:
            queue_message(interaction, "Looping song.")
        else:
            queue_message(interaction, "Stopping looping.")
    else:
        queue_message(interaction, "Not currently playing a song.")


def pause(interaction: Interaction) -> None:
    """Pause or resume playback."""
    if state.vc and state.vc.is_playing():
        if not state.vc.is_paused():
            queue_message(interaction, f"Paused song: {prev_music[-1] if prev_music else 'unknown'}")
            state.vc.pause()
        else:
            queue_message(interaction, f"Unpaused song: {prev_music[-1] if prev_music else 'unknown'}")
            state.vc.resume()
    else:
        queue_message(interaction, "Not currently playing a song.")


def print_queue(interaction: Interaction) -> None:
    """Print the current queue."""
    queue_msg = "```"
    for q in music_queue:
        queue_msg += q + '\n'
        if len(queue_msg) > BREAK_LEN:
            queue_message(interaction, f"{queue_msg}```")
            queue_msg = "```"
    queue_message(interaction, f"{queue_msg}```")


def print_prev_queue(interaction: Interaction) -> None:
    """Print the previously played songs."""
    queue_msg = "```"
    for q in prev_music:
        queue_msg += q + '\n'
        if len(queue_msg) > BREAK_LEN:
            queue_message(interaction, f"{queue_msg}```")
            queue_msg = "```"
    queue_message(interaction, f"{queue_msg}```")


def clear_queue(interaction: Interaction) -> None:
    """Clear the current queue."""
    queue_message(interaction, "Cleared queue")
    prev_music.extend(music_queue)
    music_queue.clear()


def stop(interaction: Interaction) -> None:
    """Stop playback and clear the queue."""
    if state.vc:
        prev_song = prev_music[-1] if prev_music else ""
        queue_message(interaction, f"Clearing queue and stopping song: {prev_song}")
        prev_music.extend(music_queue)
        music_queue.clear()
        state.vc.stop()
    else:
        queue_message(interaction, "Not currently in a voice channel.")


async def disconnect(interaction: Interaction) -> None:
    """Disconnect from voice and reset state."""
    if state.vc:
        await state.vc.disconnect()
        state.vc = None
        state.last_channel_id = None
        state.loop_song = False
        music_queue.clear()
        prev_music.clear()
        await interaction.response.send_message("Disconnected from voice channel.")
    else:
        await interaction.response.send_message("Not currently in a voice channel. If stuck, use `/play` then `/disconnect`.")


def delete_song(interaction: Interaction, song_name: str) -> None:
    """
    Delete a song/folder under MUSIC_BASE_DIR.

    Args:
        interaction: Discord interaction
        song_name: Name or path of song/folder to delete
    """
    path = song_name if song_name.startswith(MUSIC_BASE_DIR) else join(MUSIC_BASE_DIR, song_name)
    full_path = Path(path).resolve()
    base_path = Path(MUSIC_BASE_DIR).resolve()

    # Security: ensure path is under MUSIC_BASE_DIR
    if not str(full_path).startswith(str(base_path)):
        queue_message(interaction, "Invalid path: must be under music directory.")
        return

    if not full_path.exists():
        queue_message(interaction, f"Song/folder not found: {song_name}")
        return

    # Safety: prevent deleting root music directory
    if full_path == base_path:
        queue_message(interaction, "Cannot delete the root music directory. Use `/delete_all_music` instead.")
        return

    if full_path.is_file():
        full_path.unlink()
    elif full_path.is_dir():
        rmtree(full_path)
    queue_message(interaction, f"Deleted: {full_path.name}")


def delete_all_music(interaction: Interaction) -> None:
    """Delete all music files and folders under MUSIC_BASE_DIR."""
    if not Path(MUSIC_BASE_DIR).exists():
        queue_message(interaction, "Music directory not found.")
        return

    for item in listdir(MUSIC_BASE_DIR):
        if item in SKIP_FILES:
            continue

        path = join(MUSIC_BASE_DIR, item)
        if Path(path).is_file():
            Path(path).unlink()
        elif Path(path).is_dir():
            rmtree(path)

    queue_message(interaction, "All music files and folders have been deleted.")


# ========================
# Module Exports
# ========================
__all__ = [
    'clear_queue',
    'delete_all_music',
    'delete_song',
    'disconnect',
    'init_vc',
    'list_all_music',
    'loop',
    'pause',
    'play_all',
    'play_music',
    'print_prev_queue',
    'print_queue',
    'replay',
    'replay_all',
    'set_volume',
    'shuffle_music',
    'skip',
    'stop',
]
