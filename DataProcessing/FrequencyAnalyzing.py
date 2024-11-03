import numpy as np

class FrequencyAnalyzing:

    def DFT(segment, frame_rate):
        X_m_segment = np.fft.fft(segment)
        freqs_segment = np.fft.fftfreq(len(segment), 1/frame_rate)
        return (X_m_segment, freqs_segment)