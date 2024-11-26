import numpy as np

def DFT(segment, frame_rate):
    # X_m_segment = np.fft.fft(segment)
    # freqs_segment = np.fft.fftfreq(len(segment), 1/frame_rate)
    # return (X_m_segment, freqs_segment)

    segment = np.piecewise(segment, 
                           [segment < 1, segment >= 1], 
                           [lambda t: (2 * t) ** 2, lambda t: (4 - 2 * t) ** 2])

    N = len(segment)  
    X_m_segment = []  

    for m in range(N):
        sum_real = 0  
        sum_imag = 0  
        for n in range(N):
            angle = -2 * np.pi * m * n / N
            sum_real += segment[n] * np.cos(angle)
            sum_imag += segment[n] * np.sin(angle)
        X_m_segment.append(complex(sum_real, sum_imag))  

    X_m_segment = np.array(X_m_segment)  
    freqs_segment = np.linspace(0, frame_rate, N, endpoint=False)

    return X_m_segment, freqs_segment

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