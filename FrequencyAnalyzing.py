import numpy as np

def DFT(segment, frame_rate):
    X_m_segment = np.fft.fft(segment)
    freqs_segment = np.fft.fftfreq(len(segment), 1/frame_rate)
    return (X_m_segment, freqs_segment)

def Smooth(X, L):

    smooth = [0] * len(X)

    for i in range(0, len(X)):
        if abs(i) < len(X) / 2:
            smooth[i] = (L - abs(i)) / ( L ^ 2)
        else:
            smooth[i] = 0

    Xs = abs(X) * smooth

    return Xs

def GetNotes(midi):
    base_notes = {
        48: "Do", 50: "Re", 52: "Mi", 53: "Fa", 55: "So", 57: "La", 59: "Ti"
    }
    notes = []

    for value in midi:

        rounded_value = round(value)
        base_note = (rounded_value - 48) % 12 + 48
        octave = (rounded_value - 48) // 12 
        
        note_name = base_notes.get(base_note, None)
        if note_name:
            note_with_octave = f"{note_name} (Octave {octave})"
            notes.append(note_with_octave)
        else:
            notes.append("Unknown")

    return notes