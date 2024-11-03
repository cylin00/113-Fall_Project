import numpy as np

class FrequencyAnalyzing:

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