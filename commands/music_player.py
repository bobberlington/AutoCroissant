from collections import deque
from discord import Message, VoiceClient, PCMVolumeTransformer
from os import listdir, makedirs
from os.path import getctime
from pathlib import Path
from random import shuffle
from re import sub
from time import sleep
import yt_dlp

from commands.tools import music, prev_music, messages, commands
import config

music_base_dir = "music/"
vc: VoiceClient | None = None
last_channel_id: id = None
loop_song: bool = False
postprocess = False
latest_filename = ""
ext = '.webm'
try:
    postprocess = config.postprocess
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

def list_all_music(message: Message):
    full_msg = recursively_traverse(music_base_dir, '')
    while len(full_msg) > 1500:
        i = full_msg.index('\n', 1500)
        part_msg = full_msg[:i]
        full_msg = full_msg[i:]
        messages.append((message.channel.id, "```" + part_msg + "```"))
    messages.append((message.channel.id, "```" + full_msg + "```"))

async def init_vc(message: Message):
    global vc, last_channel_id
    last_channel_id = message.channel.id
    if message.author.voice:
        vc = await message.author.voice.channel.connect()
    else:
        messages.append((message.channel.id, "User not connected to voice channel."))

def queue_song_async(song: str, channel_id: int, sleep_timer: int | str | None = None):
    if type(sleep_timer) == int:
        sleep(sleep_timer)
    else:
        if sleep_timer == "latest":
            sleep(5)
            if latest_filename.find(music_base_dir) != -1:
                song = latest_filename
        elif sleep_timer == "exists":
            timer = 0
            while True:
                if Path(song).is_file():
                    break
                sleep(1)
                timer += 1
                if timer > 30:
                    break
            sleep(5)
            if timer > 30:
                song = '/'.join(song.split('/')[:-1])
                song = sorted([song + '/' + name for name in listdir(song)], key=getctime)[1]
        elif sleep_timer == "finish":
            sleep(30)
            while vc.is_playing():
                sleep(1)

    if not song:
        song = music_base_dir
    if Path(song).is_dir():
        if not song.endswith('/'):
            song += '/'
        queue_msg = "```"
        for s in listdir(song):
            if (song + s in prev_music) or (postprocess and not s.endswith(ext)):
                continue
            queue_msg += s + '\n'
            music.append(song + s)
        messages.append((channel_id, "Queued songs:" + queue_msg + "```"))
    else:
        music.append(song)
        messages.append((channel_id, "Queued song: " + song))

def play_song_async(message: Message):
    song = ' '.join(message.content.split()[1:])
    if not song:
        messages.append((message.channel.id, "Please specify a song."))
    else:
        if song.startswith("http"):
            pre_extract = dl_pre.extract_info(url=song, download=False)
            if song.find("playlist") != -1 or song.find("album") != -1:
                playlist_title = pre_extract['title']
                if len(pre_extract['entries']) > 0:
                    if not Path(music_base_dir + playlist_title).is_dir():
                        makedirs(music_base_dir + playlist_title)

                    temp_ydl_opts = ydl_opts
                    temp_ydl_opts['outtmpl'] = music_base_dir + playlist_title + '/' +  '%(title)s.%(ext)s'
                    temp_dl = yt_dlp.YoutubeDL(temp_ydl_opts)

                    commands.append(((song, True), temp_dl.extract_info))
                    commands.append((('', message.channel.id, "latest"), queue_song_async))
                commands.append(((music_base_dir + playlist_title + '/', message.channel.id, "finish" if len(pre_extract['entries']) > 0 else None), queue_song_async))
            else:
                commands.append(((song, True), dl.extract_info))
                commands.append(((music_base_dir + sub('\W+','_', pre_extract['title'].replace(':', '_')).replace('__', '_-_').strip('_') + ext, message.channel.id, "latest"), queue_song_async))
        else:
            song = music_base_dir + song
            if Path(song).is_dir():
                if not song.endswith('/'):
                    song += '/'
                queue_msg = "```"
                for s in listdir(song):
                    queue_msg += s + '\n'
                    music.append(song + s)
                messages.append((message.channel.id, "Queued songs:" + queue_msg + "```"))
            else:
                music.append(song)
                messages.append((message.channel.id, "Queued song: " + song))

async def play_music(message: Message):
    if not vc or not vc.is_connected():
        await init_vc(message)
    commands.append(((message,), play_song_async))

def replay_last(message: Message):
    if len(prev_music) > 0:
        messages.append((message.channel.id, "Replaying song: " + prev_music[-1]))
        music.appendleft(prev_music[-1])
    else:
        messages.append((message.channel.id, "No previously played songs."))

def replay_all(message: Message):
    global music, prev_music
    if len(prev_music) > 0:
        messages.append((message.channel.id, "Replaying all songs."))
        prev_music = deque(set(prev_music))
        music += prev_music
        prev_music.clear()
        prev_music.append(music[-1])
    else:
        messages.append((message.channel.id, "No previously played songs."))

def set_volume(message: Message):
    msg = message.content.split()[1]
    volume = float(msg)
    if vc and (vc.is_playing() or vc.is_paused()):
        messages.append((message.channel.id, "Setting volume: " + msg))
        vc.source = PCMVolumeTransformer(vc.source, volume=volume)
    else:
        messages.append((message.channel.id, "Not currently in a voice channel OR not currently playing a song, I think?"))

def shuffle_music(message: Message):
    messages.append((message.channel.id, "Shuffling songs."))
    shuffle(music)

def skip(message: Message):
    if vc and (vc.is_playing() or vc.is_paused()):
        messages.append((message.channel.id, "Skipping song: " + prev_music[-1]))
        vc.stop()
    else:
        messages.append((message.channel.id, "Not currently in a voice channel OR not currently playing a song, I think?"))

def loop(message: Message):
    global loop_song
    if vc and (vc.is_playing() or vc.is_paused()):
        loop_song = not loop_song
        if loop_song:
            if len(music) == 0:
                music.append("dummy")
            messages.append((message.channel.id, "Looping song: " + prev_music[-1]))
        else:
            messages.append((message.channel.id, "Stopping looping."))
    else:
        messages.append((message.channel.id, "Not currently in a voice channel OR not currently playing a song, I think?"))

def pause(message: Message):
    if vc and vc.is_playing():
        if not vc.is_paused():
            messages.append((message.channel.id, "Paused song: " + prev_music[-1]))
            vc.pause()
        else:
            messages.append((message.channel.id, "Unpaused song: " + prev_music[-1]))
            vc.resume()
    else:
        messages.append((message.channel.id, "Not currently in a voice channel OR not currently playing a song, I think?"))

def print_queue(message: Message):
    queue_msg = "```"
    for q in music:
        queue_msg += q + '\n'
        if len(queue_msg) > 1500:
            messages.append((message.channel.id, queue_msg + "```"))
            queue_msg = "```"
    messages.append((message.channel.id, queue_msg + "```"))

def print_prev_queue(message: Message):
    queue_msg = "```"
    for q in prev_music:
        queue_msg += q + '\n'
        if len(queue_msg) > 1500:
            messages.append((message.channel.id, queue_msg + "```"))
            queue_msg = "```"
    messages.append((message.channel.id, queue_msg + "```"))

def clear_queue(message: Message):
    global prev_music
    messages.append((message.channel.id, "Cleared queue"))
    prev_music += music
    music.clear()

def stop(message: Message):
    global prev_music, music
    if vc:
        messages.append((message.channel.id, "Clearing queue and stopping song: " + prev_music[-1]))
        prev_music += music
        music.clear()
        vc.stop()
    else:
        messages.append((message.channel.id, "Not currently in a voice channel OR not currently playing a song, I think?"))

def disconnect(message: Message):
    global vc, last_channel_id, music, prev_music
    if vc:
        commands.append((("await",), vc.disconnect))
        vc = last_channel_id = None
        music = deque()
        prev_music = deque()
    else:
        messages.append((message.channel.id, "Not currently in a voice channel. If I am in a voice channel, do ```-play``` and then ```-disconnect```"))
