import DetectOnsets
import numpy as np
import ReadAudio
import os
import glob
import MidiMatrix

queryDirectory = "hummingdata/10/"
wavFiles = glob.glob(os.path.join(queryDirectory, "*.wav"))

songs = ReadAudio.ReadTarget()

for f in wavFiles:
    fileName = os.path.basename(f)
    
    prefix, originalQuery = fileName.split("_", 1)
    querySong = originalQuery.replace(".wav", "")
    
    queryNote = DetectOnsets.Query(fileName)
    
    targetNote = next((song[1] for song in songs if song[0] == querySong), None)

    M = MidiMatrix.GetMidiMatrix(targetNote, queryNote)
    lastRow = M[len(M) - 1]
    
    print(f"Edit Distance of {querySong} = {min(lastRow)}\n")
