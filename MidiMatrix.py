import numpy as np

def GetMidiMatrix(target, query):
    midiMatrix = np.zeros((len(query) + 1, len(target) + 1))
    d = 4

    # print(f"Matrix shape: {midiMatrix.shape}")

    for i in range(len(query) + 1):
        midiMatrix[i][0] = d * i

    for j in range(len(target) + 1):
        midiMatrix[0][j] = 0

    # print(f"Initial matrix:\n{midiMatrix}")

    for i in range(1, len(query) + 1):  
        for j in range(1, len(target) + 1):  
            midiMatrix[i][j] = min(
                midiMatrix[i-1][j-1] + abs(target[j-1] - query[i-1]),  
                midiMatrix[i-1][j] + d,  
                midiMatrix[i][j-1] + d   
            )

    return midiMatrix