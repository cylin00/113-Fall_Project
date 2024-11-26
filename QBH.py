import DetectOnsets
import numpy as np
import ReadAudio
import os
import glob
import MidiMatrix

queryDirectory = "hummingdata/20/"
wavFiles = glob.glob(os.path.join(queryDirectory, "*.wav"))

songs = ReadAudio.ReadTarget()

rate = 0

for f in wavFiles:
    fileName = os.path.basename(f)
    prefix, originalQuery = fileName.split("_", 1)
    querySong = originalQuery.replace(".wav", "")
    
    queryNote = DetectOnsets.Query(fileName)
    targetNote = next((song[1] for song in songs if song[0] == querySong), None)
    M = MidiMatrix.GetMidiMatrix(np.diff(targetNote), np.diff(queryNote))
    ED = min(M[len(M) - 1])

    # print(f"query song: {querySong} ==============================")
    # print(f"Edit Distance of {querySong} = {ED}\n")

    rank = 1

    for t in songs:
        midiMatrix = MidiMatrix.GetMidiMatrix(t[1], queryNote)
        d = min(midiMatrix[len(midiMatrix) - 1])

        if(d < ED):
            rank += 1
            # print(f"distance of target {t[0]} = {d}")

    print(f"Rank of {querySong} = {rank}\n")
    rate += 1 / rank

print(f"Rate = {rate / len(wavFiles)}")

    
