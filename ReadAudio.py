import numpy as np

song_dtype = np.dtype([
        ("name", "U50"),          
        ("notes", "O"),
        ("notediff", "O"),        
        ("beats", "O")
    ])

def TranformMidiNumber(value):
    midiMap = {
        65: 36,
        66: 38,
        67: 40,
        68: 41,
        69: 43,
        70: 45,
        71: 47,
        49: 48,
        50: 50,
        51: 52,
        52: 53,
        53: 55,
        54: 57,
        55: 59,
        72: 60,
        73: 62,
        74: 64,
        75: 65,
        76: 67,
        77: 69,
        78: 71,
    }

    return midiMap.get(value, value)

def ReadNote(n):
    note = []
    noteDiff = []

    for i in range(0, len(n)):
        note.append(TranformMidiNumber(ord(n[i])))

    for i in range(0, len(note)-1):
        noteDiff.append(note[i+1] - note[i])
    return (note, noteDiff)

def ReadBeat(b):
    beat = []
    for i in range(0, len(b)):
        beat.append((int)(b[i])/2)
    return beat

def AnalyzeEachSong(s):
    sp = []

    for i in range(0, len(s)):
        if ord(s[i]) == 12288: # 32 -> space
            sp.append(i)
        if ord(s[i]) == 47: # 47 -> slash
            sl = i

    (Note, NoteDiff) = ReadNote(s[sp[len(sp)-1]+1:sl])

    return np.array([(s[0:sp[0]], 
                      Note,
                      NoteDiff, 
                      ReadBeat(s[sl+1:len(s)-1]))], 
                    dtype=song_dtype)

def ReadTarget():

    t = open('Target_tempo_50.txt', 'r', encoding='big5')

    songs = np.array([], dtype=song_dtype)
    for s in t.readlines():
        songs = np.append(songs, AnalyzeEachSong(s))

    # with open('songs.txt', 'w') as file:
    #     for song in songs:
    #         file.write(f"{song}\n") 

    

    return songs
