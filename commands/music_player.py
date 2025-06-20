from collections import deque
from discord import VoiceClient, PCMVolumeTransformer, Interaction, channel
from os import listdir, makedirs
from os.path import getctime
from pathlib import Path
from random import shuffle
from re import sub
from time import sleep
import yt_dlp

from commands.utils import music, prev_music, messages, commands

vc: VoiceClient | None = None
last_channel: channel = None
latest_filename: str = ""
ext: str = '.webm'
loop_song: bool = False

music_base_dir: str = "music/"
break_len: int = 1500
postprocess: bool = False
cookies_from_browser: bool = False
try:
    import config
    postprocess = config.postprocess
    if config.vram_usage == "mps":
        cookies_from_browser = True
except AttributeError:
    print("No music params in config, skipping.")

def get_filename(song):
    global latest_filename
    latest_filename = song.get('info_dict').get('_filename')
    if postprocess:
        latest_filename = '.'.join(latest_filename.split('.')[:-1]) + ext

ydl_preprocess_opts = {
    'format': 'bestaudio/best',
    'restrictfilenames': True,
    'playlistend': 1,
    'outtmpl': music_base_dir + '%(title)s.%(ext)s',
    "progress_hooks": [get_filename],
}

ydl_opts = {
    'format': 'bestaudio/best',
    'restrictfilenames': True,
    'outtmpl': music_base_dir + '%(title)s.%(ext)s',
    'download_archive': music_base_dir + 'downloaded.txt',
    "progress_hooks": [get_filename],
}

if cookies_from_browser:
    ydl_preprocess_opts['cookiesfrombrowser'] = ('safari', None, None, None)
    ydl_opts['cookiesfrombrowser'] = ('safari', None, None, None)
if postprocess:
    ext = '.mp3'
    ydl_preprocess_opts['postprocessors'] = [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '320',
    }]
    ydl_opts['postprocessors'] = [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '320',
    }]
    dl_pre = yt_dlp.YoutubeDL(ydl_preprocess_opts)
    dl = yt_dlp.YoutubeDL(ydl_opts)
else:
    dl_pre = yt_dlp.YoutubeDL(ydl_preprocess_opts)
    dl = yt_dlp.YoutubeDL(ydl_opts)


def recursively_traverse(dir: str, tabs: str):
    msg = ""
    for file in listdir(dir):
        if Path(dir + file).is_dir():
            msg += '\n' + recursively_traverse(dir + file + '/', tabs + '\t') + '\n'
        else:
            msg += tabs + dir + file + '\n'
    return msg

def list_all_music(interaction: Interaction):
    full_msg = recursively_traverse(music_base_dir, '')
    while len(full_msg) > break_len:
        i = full_msg.index('\n', break_len)
        part_msg = full_msg[:i]
        full_msg = full_msg[i:]
        messages.append((interaction, "```" + part_msg + "```"))
    messages.append((interaction, "```" + full_msg + "```"))

def queue_song_async(song: str, interaction: Interaction, sleep_timer: int | str | None = None, play_next: bool = False):
    if type(sleep_timer) == int:
        sleep(sleep_timer)
    else:
        timer = 0
        if sleep_timer == "until_filename_available":
            sleep(3)
            while not latest_filename:
                sleep(0.25)
                timer += 1
                if timer > 40:
                    break
            if music_base_dir.strip('/') in latest_filename:
                song = latest_filename
            sleep(15 - int(timer / 4))
        elif sleep_timer == "until_file_exists":
            while True:
                if Path(song).is_file() or timer > 30:
                    break
                sleep(1)
                timer += 1
            sleep(5)
            if timer > 30:
                song = '/'.join(song.split('/')[:-1])
                song = sorted([song + '/' + name for name in listdir(song)], key=getctime)[1]
        elif sleep_timer == "until_song_finishes":
            sleep(30)
            while vc.is_playing():
                sleep(1)

    if not song:
        messages.append((interaction, "Invalid song or error?"))
        return

    if Path(song).is_dir():
        if not song.endswith('/'):
            song += '/'
        queue_msg = "```"
        for s in listdir(song):
            if (song + s in prev_music) or (postprocess and not s.endswith(ext)):
                continue
            queue_msg += s + '\n'
            if len(queue_msg) > break_len:
                messages.append((interaction, "Queued songs:" + queue_msg + "```"))
                queue_msg = "```"
            if play_next:
                music.appendleft(song + s)
            else:
                music.append(song + s)
        messages.append((interaction, "Queued songs:" + queue_msg + "```"))
    else:
        if play_next:
            music.appendleft(song)
        else:
            music.append(song)
        messages.append((interaction, "Queued song: " + song))

def play_all_async(interaction: Interaction):
    all_songs = recursively_traverse(music_base_dir, '').replace('\t', '')
    messages.append((interaction, "Queueing all songs"))
    for song in all_songs.split('\n'):
        music.append(song)

def play_song_async(interaction: Interaction, song: str, play_next: bool):
    if song.startswith("http"):
        pre_extract = dl_pre.extract_info(url=song, download=False)
        if "playlist" in song or "album" in song:
            playlist_title = pre_extract['title'].replace(':', '')
            if len(pre_extract['entries']) > 0:
                first_song_title = sub('\W+','_', pre_extract['entries'][0]['title'].replace(':', '_')).replace('__', '_-_').strip('_')
                if not Path(music_base_dir + playlist_title).is_dir():
                    makedirs(music_base_dir + playlist_title)

                temp_ydl_opts = ydl_opts.copy()
                temp_ydl_opts['outtmpl'] = music_base_dir + playlist_title + '/' +  '%(title)s.%(ext)s'

                commands.append(((song, True), yt_dlp.YoutubeDL(temp_ydl_opts).extract_info))
                commands.append(((music_base_dir + playlist_title + '/' + first_song_title + ext, interaction, "until_filename_available", play_next), queue_song_async))
            commands.append(((music_base_dir + playlist_title + '/', interaction, "until_song_finishes" if len(pre_extract['entries']) > 0 else None, play_next), queue_song_async))
        else:
            commands.append(((song, True), dl.extract_info))
            commands.append(((music_base_dir + sub('\W+','_', pre_extract['title'].replace(':', '_')).replace('__', '_-_').strip('_') + ext, interaction, "until_filename_available", play_next), queue_song_async))
    else:
        if not song.startswith(music_base_dir.strip('/')):
            song = music_base_dir + song
        if Path(song).is_dir():
            if not song.endswith('/'):
                song += '/'
            queue_msg = "```"
            for s in sorted(listdir(song)):
                queue_msg += s + '\n'
                if len(queue_msg) > break_len:
                    messages.append((interaction, "Queued songs:" + queue_msg + "```"))
                    queue_msg = "```"
                if play_next:
                    music.appendleft(song + s)
                else:
                    music.append(song + s)
            messages.append((interaction, "Queued songs:" + queue_msg + "```"))
        else:
            if play_next:
                music.appendleft(song)
            else:
                music.append(song)
            messages.append((interaction, "Queued song: " + song))

async def init_vc(interaction: Interaction):
    global vc, last_channel
    last_channel = interaction.channel
    if interaction.user.voice:
        vc = await interaction.user.voice.channel.connect()
    else:
        messages.append((interaction, "User not connected to voice channel."))

async def play_music(interaction: Interaction, song: str, play_next: bool = False):
    if vc is None or not vc.is_connected():
        await init_vc(interaction)
    commands.append(((interaction, song, play_next), play_song_async))

async def play_all(interaction: Interaction):
    if vc is None or not vc.is_connected():
        await init_vc(interaction)
    commands.append(((interaction,), play_all_async))

def replay(interaction: Interaction, rep_index: int = 0):
    if len(prev_music) > 0:
        messages.append((interaction, "Replaying song: " + prev_music[-1 - rep_index]))
        music.appendleft(prev_music[-1 - rep_index])
    else:
        messages.append((interaction, "No previously played songs."))

def replay_all(interaction: Interaction):
    global music, prev_music
    if len(prev_music) > 0:
        messages.append((interaction, "Replaying all songs."))
        prev_music = deque(set(prev_music))
        music += prev_music
        prev_music.clear()
        prev_music.append(music[-1])
    else:
        messages.append((interaction, "No previously played songs."))

def set_volume(interaction: Interaction, volume: float):
    if vc and (vc.is_playing() or vc.is_paused()):
        messages.append((interaction, "Setting volume: " + str(volume)))
        vc.source = PCMVolumeTransformer(vc.source, volume=volume)
    else:
        messages.append((interaction, "Not currently in a voice channel OR not currently playing a song, I think?"))

def shuffle_music(interaction: Interaction):
    messages.append((interaction, "Shuffling songs."))
    shuffle(music)

def skip(interaction: Interaction):
    if vc and (vc.is_playing() or vc.is_paused()):
        messages.append((interaction, "Skipping song: " + prev_music[-1]))
        vc.stop()
    else:
        messages.append((interaction, "Not currently in a voice channel OR not currently playing a song, I think?"))

def loop(interaction: Interaction):
    global loop_song
    if vc and (vc.is_playing() or vc.is_paused()):
        loop_song = not loop_song
        if loop_song:
            if len(music) == 0:
                music.append("dummy")
            messages.append((interaction, "Looping song: " + prev_music[-1]))
        else:
            messages.append((interaction, "Stopping looping."))
    else:
        messages.append((interaction, "Not currently in a voice channel OR not currently playing a song, I think?"))

def pause(interaction: Interaction):
    if vc and vc.is_playing():
        if not vc.is_paused():
            messages.append((interaction, "Paused song: " + prev_music[-1]))
            vc.pause()
        else:
            messages.append((interaction, "Unpaused song: " + prev_music[-1]))
            vc.resume()
    else:
        messages.append((interaction, "Not currently in a voice channel OR not currently playing a song, I think?"))

def print_queue(interaction: Interaction):
    queue_msg = "```"
    for q in music:
        queue_msg += q + '\n'
        if len(queue_msg) > break_len:
            messages.append((interaction, queue_msg + "```"))
            queue_msg = "```"
    messages.append((interaction, queue_msg + "```"))

def print_prev_queue(interaction: Interaction):
    queue_msg = "```"
    for q in prev_music:
        queue_msg += q + '\n'
        if len(queue_msg) > break_len:
            messages.append((interaction, queue_msg + "```"))
            queue_msg = "```"
    messages.append((interaction, queue_msg + "```"))

def clear_queue(interaction: Interaction):
    global prev_music
    messages.append((interaction, "Cleared queue"))
    prev_music += music
    music.clear()

def stop(interaction: Interaction):
    global prev_music
    if vc:
        messages.append((interaction, "Clearing queue and stopping song: " + prev_music[-1]))
        prev_music += music
        music.clear()
        vc.stop()
    else:
        messages.append((interaction, "Not currently in a voice channel OR not currently playing a song, I think?"))

def disconnect(interaction: Interaction):
    global vc, last_channel, music, prev_music
    if vc:
        commands.append((("await",), vc.disconnect))
        vc = last_channel = None
        music = deque()
        prev_music = deque()
    else:
        messages.append((interaction, "Not currently in a voice channel. If I am in a voice channel, do ```/play``` and then ```/disconnect```"))
