import DetectOnsets
import numpy as np
import ReadAudio
import os
import glob
import MidiMatrix

queryDirectory = "hummingdata/10/"
wavFiles = glob.glob(os.path.join(queryDirectory, "*.wav"))

# targetNames = []

songs = ReadAudio.ReadTarget()

# for song in songs:
#     print(song[1])
#     songName = song[0]
#     targetNames.append(songName)

for f in wavFiles:
    fileName = os.path.basename(f)
    
    prefix, originalQuery = fileName.split("_", 1)
    querySong = originalQuery.replace(".wav", "")
    
    queryNote = DetectOnsets.Query(fileName)
    
    targetNote = next((song[1] for song in songs if song[0] == querySong), None)

    # print(f"Query: {querySong}, Note = {queryNote}")
    # print(f"Target: {querySong}, Note = {targetNote}\n")

    M = MidiMatrix.GetMidiMatrix(targetNote, queryNote)
    lastRow = M[len(M) - 1]
    #print(f"Midi matrix for {querySong}:\n{M}\n")
    print(f"Edit Distance of {querySong} = {min(lastRow)}\n")

# print("All target names:", targetNames)\
