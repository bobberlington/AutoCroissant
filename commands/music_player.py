from discord import Message, VoiceClient, FFmpegPCMAudio

from commands.tools import messages, commands

vc: VoiceClient | None = None
cur_song : str = ""

async def play_music(message: Message):
    global vc, cur_song
    if not vc:
        voice_channel = None
        if message.author.voice:
            voice_channel = message.author.voice.channel
        else:
            messages.append((message.channel.id, "User not connected to voice channel."))
            return

        if voice_channel:
            vc = await voice_channel.connect()
            song = message.content.split()
            if len(song) > 1:
                cur_song = ' '.join(song[1:])
                commands.append(((FFmpegPCMAudio(source=cur_song),), vc.play))
            else:
                messages.append((message.channel.id, "Please specify a song."))
        else:
            messages.append((message.channel.id, "Invalid voice channel."))
            return
    else:
        if not vc.is_connected():
            vc = None
            await play_music(message)
        else:
            if not vc.is_playing():
                song = message.content.split()
                if len(song) > 1:
                    cur_song = ' '.join(song[1:])
                    commands.append(((FFmpegPCMAudio(source=cur_song),), vc.play))
                else:
                    messages.append((message.channel.id, "Please specify a song."))

def disconnect(message: Message):
    global vc, cur_song
    if vc:
        commands.append((("await",), vc.disconnect))
        vc = cur_song = None
    else:
        messages.append((message.channel.id, "Not currently in a voice channel. If I am in a voice channel, do ```-play``` and then ```-disconnect```"))

def pause(message: Message):
    global vc
    if vc and cur_song:
        if not vc.is_paused():
            messages.append((message.channel.id, "Paused song: " + cur_song))
            vc.pause()
        else:
            messages.append((message.channel.id, "Unpaused song: " + cur_song))
            vc.resume()
    else:
        messages.append((message.channel.id, "Not currently in a voice channel OR not currently playing a song, I think?"))

def stop(message: Message):
    global vc, cur_song
    if vc and (vc.is_playing() or vc.is_paused()):
        messages.append((message.channel.id, "Stopping song: " + cur_song))
        vc.stop()
        cur_song = None
    else:
        messages.append((message.channel.id, "Not currently in a voice channel OR not currently playing a song, I think?"))
