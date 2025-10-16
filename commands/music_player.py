from asyncio import sleep
from collections import deque
from dataclasses import dataclass
from discord import VoiceClient, PCMVolumeTransformer, Interaction
from os import listdir, makedirs, sep
from os.path import join, isdir
from pathlib import Path
from random import shuffle
from re import sub
from unicodedata import normalize
from yt_dlp import YoutubeDL

from commands.utils import music_queue, prev_music, queue_message, queue_command, split_long_message, recursively_traverse, BREAK_LEN
import config

# ========================
# Configurable Parameters
# ========================
MUSIC_BASE_DIR: str = "music/"
EXT: str =            '.webm'
TIMER_WAIT: int =     30

postprocess: bool =          getattr(config, "postprocess", False)
cookies_from_browser: bool = getattr(config, "vram_usage", False) == "mps"

# ========================
# Dataclass for Music State
# ========================
@dataclass
class MusicState:
    vc: VoiceClient | None =   None
    last_channel: int | None = None
    latest_filename: str =     ""
    loop_song: bool =          False

state = MusicState()

# ========================
# YT-DLP Setup
# ========================
def get_filename(song):
    """Hook to store the latest filename from yt-dlp downloads."""
    state.latest_filename = song.get('info_dict', {}).get('_filename', "")
    if postprocess and state.latest_filename:
        state.latest_filename = ".".join(state.latest_filename.split(".")[:-1]) + EXT

ydl_opts_base = {
    "format": "bestaudio/best",
    "restrictfilenames": True,
    "outtmpl": join(MUSIC_BASE_DIR, "%(title)s.%(ext)s"),
    "progress_hooks": [get_filename],
}

ydl_opts = ydl_opts_base.copy()
ydl_opts["download_archive"] = join(MUSIC_BASE_DIR, "downloaded.txt")
ydl_preprocess_opts = ydl_opts_base.copy()
ydl_preprocess_opts["playlistend"] = 1

if cookies_from_browser:
    ydl_opts["cookiesfrombrowser"] = ("safari", None, None, None)
    ydl_preprocess_opts["cookiesfrombrowser"] = ("safari", None, None, None)

if postprocess:
    EXT = ".mp3"
    post_proc = {
        "key": "FFmpegExtractAudio",
        "preferredcodec": "mp3",
        "preferredquality": "320",
    }
    ydl_opts["postprocessors"] = [post_proc]
    ydl_preprocess_opts["postprocessors"] = [post_proc]

dl = YoutubeDL(ydl_opts)
dl_pre = YoutubeDL(ydl_preprocess_opts)


def sanitize_title(title: str) -> str:
    """Make a title safe for Discord messages and ffmpeg-compatible filenames."""
    # Normalize Unicode (remove accents, etc.)
    title = normalize("NFKD", title).encode("ascii", "ignore").decode("ascii", "ignore")

    # Replace illegal filename characters with underscores
    title = sub(r'[<>:"/\\|?*]', "_", title)

    # Replace all non-alphanumeric or space/underscore characters
    title = sub(r"[^\w\s-]", "_", title)

    # Collapse multiple underscores/spaces
    title = sub(r"[\s_]+", "_", title).strip("_")

    # Limit filename length to avoid OS issues (e.g., 255 chars)
    title = title[:100]

    return title


def queue_folder_songs(interaction: Interaction, folder: str, play_next: bool = False):
    """Queue all valid songs in a folder."""
    if not folder.endswith(sep):
        folder += sep
    if not isdir(folder):
        return queue_message(interaction, "Folder not found.")

    queue_msg = "```"
    for s in sorted(listdir(folder)):
        full_path = join(folder, s)
        if full_path in prev_music or (postprocess and not s.endswith(EXT)):
            continue
        queue_msg += s + "\n"
        if len(queue_msg) > BREAK_LEN:
            queue_message(interaction, f"Queued songs:{queue_msg}```")
            queue_msg = "```"
        (music_queue.appendleft if play_next else music_queue.append)(full_path)
    queue_message(interaction, f"Queued songs:{queue_msg}```")


def list_all_music(interaction: Interaction):
    """List all songs recursively."""
    all_music = recursively_traverse(MUSIC_BASE_DIR)
    for part in split_long_message(all_music):
        queue_message(interaction, f"```{part}```")


async def init_vc(interaction: Interaction):
    """Join user's voice channel if not already connected."""
    if not interaction.user.voice:
        return await queue_message(interaction, "You are not in a voice channel.")
    if state.vc is None or not state.vc.is_connected():
        state.vc = await interaction.user.voice.channel.connect()
    state.last_channel = interaction.channel_id


async def queue_song_async(song: str, interaction: Interaction, sleep_mode=None, play_next=False):
    """Wait conditionally and queue a song or folder."""
    timer = 0
    if isinstance(sleep_mode, int):
        await sleep(sleep_mode)
    elif sleep_mode == "until_filename_available":
        while not state.latest_filename and timer < TIMER_WAIT:
            await sleep(0.25)
            timer += 0.25
        song = state.latest_filename or song
        await sleep(max(0, 15 - int(timer)))
    elif sleep_mode == "until_file_exists":
        while not Path(song).is_file() and timer < TIMER_WAIT:
            await sleep(1)
            timer += 1
        await sleep(5)
    elif sleep_mode == "until_song_finishes" and state.vc:
        while state.vc.is_playing():
            await sleep(1)

    if Path(song).is_dir():
        queue_folder_songs(interaction, song, play_next)
    else:
        (music_queue.appendleft if play_next else music_queue.append)(song)
        queue_message(interaction, f"Queued song: {song}")


def play_all(interaction: Interaction):
    all_songs = recursively_traverse(MUSIC_BASE_DIR, '').replace('\t', '')
    queue_message(interaction, "Queueing all songs.")
    for song in all_songs.split('\n'):
        if song == ".gitignore" or song == "downloaded.txt":
            continue
        music_queue.append(song)


def play_music(interaction: Interaction, song: str, play_next=False):
    """Play music from URL or local file."""
    if song.startswith("http"):
        try:
            pre_info = dl_pre.extract_info(song, download=False)
        except Exception as e:
            return queue_message(interaction, f"Error fetching song: {e}")

        if "entries" in pre_info and pre_info["entries"]:
            folder = ""
            first_song_title = sanitize_title(pre_info["entries"][0]["title"])
            temp_opts = ydl_opts.copy()
            # Don't queue the playlist if it is not a proper playlist link
            if "&list=" in song or "?list=" in song:
                temp_opts["noplaylist"] = True
            else:
                playlist_title = sanitize_title(pre_info["title"])
                folder = join(MUSIC_BASE_DIR, playlist_title)
                makedirs(folder, exist_ok=True)
                temp_opts["outtmpl"] = join(folder, "%(title)s.%(ext)s")

            queue_command(YoutubeDL(temp_opts).extract_info, song, True)
            queue_command(queue_song_async, join(folder, first_song_title + EXT),
                          interaction, "until_filename_available", play_next)
            if not temp_opts.get("noplaylist"):
                queue_command(queue_song_async, folder, interaction, "until_song_finishes", play_next)
        else:
            queue_command(dl.extract_info, song, True)
            title = sanitize_title(pre_info["title"])
            queue_command(queue_song_async, join(MUSIC_BASE_DIR, title + EXT),
                          interaction, "until_filename_available", play_next)
    else:
        path = song if song.startswith(MUSIC_BASE_DIR) else join(MUSIC_BASE_DIR, song)
        if Path(path).is_dir():
            queue_folder_songs(interaction, path, play_next)
        else:
            (music_queue.appendleft if play_next else music_queue.append)(path)
            queue_message(interaction, f"Queued song: {Path(path).name}")


def replay(interaction: Interaction, rep_index: int = 0):
    prev_music_len = len(prev_music)
    if prev_music_len > 0:
        if abs(rep_index) >= prev_music_len:
            rep_index = prev_music_len - 1
        queue_message(interaction, f"Replaying song: {prev_music[-1 - rep_index]}")
        music_queue.appendleft(prev_music[-1 - rep_index])
    else:
        queue_message(interaction, "No previously played songs.")


def replay_all(interaction: Interaction):
    global music_queue, prev_music
    if len(prev_music) > 0:
        queue_message(interaction, "Replaying all songs.")
        prev_music = deque(set(prev_music))
        music_queue += prev_music
        prev_music.clear()
        prev_music.append(music_queue[-1])
    else:
        queue_message(interaction, "No previously played songs.")


def set_volume(interaction: Interaction, volume: float):
    if state.vc and (state.vc.is_playing() or state.vc.is_paused()):
        queue_message(interaction, f"Setting volume: {str(volume)}")
        state.vc.source = PCMVolumeTransformer(state.vc.source, volume=volume)
    else:
        queue_message(interaction, "Not currently in a voice channel OR not currently playing a song, I think?")


def shuffle_music(interaction: Interaction):
    queue_message(interaction, "Shuffling songs.")
    shuffle(music_queue)


def skip(interaction: Interaction):
    if state.vc and (state.vc.is_playing() or state.vc.is_paused()):
        queue_message(interaction, f"Skipping song: {prev_music[-1]}")
        state.vc.stop()
    else:
        queue_message(interaction, "Not currently in a voice channel OR not currently playing a song, I think?")


def loop(interaction: Interaction):
    if state.vc and (state.vc.is_playing() or state.vc.is_paused()):
        state.loop_song = not state.loop_song
        if state.loop_song:
            if len(music_queue) == 0:
                music_queue.append("dummy")
            queue_message(interaction, f"Looping song: {prev_music[-1]}")
        else:
            queue_message(interaction, "Stopping looping.")
    else:
        queue_message(interaction, "Not currently in a voice channel OR not currently playing a song, I think?")


def pause(interaction: Interaction):
    if state.vc and state.vc.is_playing():
        if not state.vc.is_paused():
            queue_message(interaction, f"Paused song: {prev_music[-1]}")
            state.vc.pause()
        else:
            queue_message(interaction, f"Unpaused song: {prev_music[-1]}")
            state.vc.resume()
    else:
        queue_message(interaction, "Not currently in a voice channel OR not currently playing a song, I think?")


def print_queue(interaction: Interaction):
    queue_msg = "```"
    for q in music_queue:
        queue_msg += q + '\n'
        if len(queue_msg) > BREAK_LEN:
            queue_message(interaction, f"{queue_msg}```")
            queue_msg = "```"
    queue_message(interaction, f"{queue_msg}```")


def print_prev_queue(interaction: Interaction):
    queue_msg = "```"
    for q in prev_music:
        queue_msg += q + '\n'
        if len(queue_msg) > BREAK_LEN:
            queue_message(interaction, f"{queue_msg}```")
            queue_msg = "```"
    queue_message(interaction, f"{queue_msg}```")


def clear_queue(interaction: Interaction):
    global prev_music
    queue_message(interaction, "Cleared queue")
    prev_music += music_queue
    music_queue.clear()


def stop(interaction: Interaction):
    global prev_music
    if state.vc:
        prev_song = prev_music[-1] if prev_music else ""
        queue_message(interaction, f"Clearing queue and stopping song: {prev_song}")
        prev_music += music_queue
        music_queue.clear()
        state.vc.stop()
    else:
        queue_message(interaction, "Not currently in a voice channel OR not currently playing a song, I think?")


def disconnect(interaction: Interaction):
    global state, music_queue, prev_music
    if state.vc:
        queue_command(state.vc.disconnect)
        state = MusicState()
        music_queue = deque()
        prev_music = deque()
        queue_message(interaction, "Succesfully disconnected.", ephemeral=True)
    else:
        queue_message(interaction, "Not currently in a voice channel. If I am in a voice channel, do ```/play``` and then ```/disconnect```")
