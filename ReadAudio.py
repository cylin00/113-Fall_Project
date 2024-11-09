import numpy as np

def ReadNote(n):
    no = []
    print(f"Note: *{n}*")
    for i in range(0, len(n)):
        no.append(n[i])
    return n

def ReadBeat(b):
    be = []
    print(f"Beat: *{b}*")
    for i in range(0, len(b)):
        be.append(b[i])
    return b


def AnalyzeEachSong(s):
    sp = []

    for i in range(0, len(s)):
        if ord(s[i]) == 12288: # 32
            sp.append(i)
        if ord(s[i]) == 47:
            sl = i

    print(f"Song =======================================")
    print(f"{s}")
    print(f"Analysis -----------------------------------")
    print(f"Song: *{s[0:sp[0]]}*")
    ReadNote(s[sp[len(sp)-1]+1:sl])
    ReadBeat(s[sl+1:len(s)-1])

    song = np.array([(s[0:sp[0]], ReadNote(s[sp[len(sp)-1]+1:sl]), ReadBeat(s[sl+1:len(s)-1]))], dtype=song_dtype)
    return song

t = open('Target_tempo_50.txt', 'r', encoding='big5')

song_dtype = np.dtype([
    ("name", "U50"),          
    ("notes", "O"),        
    ("beats", "O")
])

songs = np.array([], dtype=song_dtype)
for s in t.readlines():
    songs = np.append(songs, AnalyzeEachSong(s))

