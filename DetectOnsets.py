import numpy as np
import matplotlib.pyplot as plt
import Audio
import DataProcessing
import FrequencyAnalyzing   
import ReadAudio

def GetAmplitude(audio, n0, startTime, endTime, frame_rate):
    segment = audio[int(startTime * frame_rate // n0):int(endTime * frame_rate // n0)]
    return segment

def Query(songname):

    (amplitude_data, time, frame_rate, audio_data) = Audio.GetAudioData(10, songname)

    # Select the max-amplitude data in groups of n0 = 80
    n0 = 80
    A = DataProcessing.GetMaxAmplitude(n0, amplitude_data)
    A = DataProcessing.NormalizeData(A) # Normalization of A : mean_A -> A
    B = DataProcessing.FractionalData(A, 0.7) # Take the fractional power

    f = [3, 3, 4, 4, -1, -1, -2, -2, -2, -2, -2, -2]
    Enveloped_Data = DataProcessing.EnvelopData(B, f) # Convolution of B with an envelope match filter
            
    threshold = 15.4 # 15.3
    Predicted_onsets = []
    for i in range(1, len(Enveloped_Data) // 3 ): # Find the onsets of EnvData and > threshold value
        if (Enveloped_Data[3*i] > threshold) & (Enveloped_Data[3*i] > Enveloped_Data[3*i - 1]) & (Enveloped_Data[3*i] > Enveloped_Data[3*i - 2]):
            Predicted_onsets.append(3*i)
    onset_times = np.array(Predicted_onsets) * n0 / frame_rate

    Predicted_onsets[0] = 0
    for o in range(1, len(Predicted_onsets)):

        previous_index = o - 1
        while previous_index >= 0 and Predicted_onsets[previous_index] == 0:
            previous_index -= 1 

        if previous_index >= 0:
            if Predicted_onsets[o] - Predicted_onsets[previous_index] < 0.15 * frame_rate // n0 :
                Predicted_onsets[o] = 0

    # Remove the last onset if interval < 0.15 sec
    if Predicted_onsets[len(Predicted_onsets)-1] > (len(audio_data) - 0.15) * frame_rate // n0:
        Predicted_onsets[len(Predicted_onsets)-1] = 0

    OnsetTimes = [] # Store the onset times & onsets
    for onset_time in Predicted_onsets:
        if onset_time != 0:
            OnsetTimes.append(onset_time * n0 / frame_rate)

    OnsetIndex = []
    for onsetIndex in Predicted_onsets:
        if onsetIndex != 0:
            OnsetIndex.append(onsetIndex)

    EnvelopAmplitudeThreshold = 0.02
    for o in range(0, len(OnsetTimes)-1):
        SilenceDuration = OnsetTimes[o+1] - OnsetTimes[o]
        if SilenceDuration >= 1.5:
            StartTime = OnsetTimes[o]
            EndTime = OnsetTimes[o+1]
            
            amplitudes_in_silence = GetAmplitude(Enveloped_Data, n0, StartTime, EndTime, frame_rate)
            
            newThreshold = threshold - 0.5 # set adaptive threshold
            startIndex = 0
            for s in range(0, len(amplitudes_in_silence)-1):
                if(amplitudes_in_silence[s] >= newThreshold and (s - startIndex) >= (0.15 * frame_rate // n0)):
                    startIndex = s

    # -----------------------------------------------------

    # Discrete Fourier Transform 
    N = len(OnsetIndex) # N onsets
    Indices = []
    DFTmagnitudes = []
    OriginalData = []
    SmoothData = []

    OnsetMag = []
    FundFreq = []
    FreqSeg = []
    Interval = []
    for o in range(0, N-1):
        window_size = OnsetIndex[o+1] - OnsetIndex[o]
        Interval.append(window_size)
        segment = Enveloped_Data[o:o+window_size]
        (X_m_segment, freqs_segment) = FrequencyAnalyzing.DFT(segment, frame_rate)
        
        Xs = FrequencyAnalyzing.Smooth(X_m_segment, 20) # Add smooth filter to the DFT result

        # Store the Points satisifying the three conditions after the smooth filter
        maxIndex = np.argmax(Xs)
        fund = frame_rate
        Fup = frame_rate / 2
        Flow = 80
        for i in range(0, len(Xs)):
            if i == 0 and Xs[i] > Xs[i+1] and Xs[i] >= Xs[maxIndex]/5:
                if freqs_segment[i] <= Fup and freqs_segment[i] >= Flow:
                    if freqs_segment[i] < fund:
                        fund = freqs_segment[i]
            else:
                if Xs[i] > Xs[i-1] and Xs[i] > Xs[i+1] and Xs[i] >= Xs[maxIndex]/5:
                    if freqs_segment[i] <= Fup and freqs_segment[i] >= Flow:
                        if freqs_segment[i] < fund:
                            fund = i * freqs_segment[i]
        
        if o == 0:
            OnsetMag.append(Xs[0])
            FundFreq.append(0)
        else:
            if Xs[0] > OnsetMag[-1]:
                OnsetMag.append(Xs[0])
                FundFreq.append(o * frame_rate / len(Xs))
            else:
                OnsetMag.append(0)
                FundFreq.append(0)

        magnitude = np.abs(X_m_segment)
        indices = np.arange(OnsetIndex[o], OnsetIndex[o+1])
        Indices.extend(indices)
        DFTmagnitudes.extend(magnitude)
        OriginalData.extend(segment)
        SmoothData.extend(Xs)

        FundaFreq = min([value for value in freqs_segment if (value != 0 and value > 80 and value < 4000)])
        FreqSeg.append(FundaFreq)

    midi = 48 + 12 * np.log2(np.array(FreqSeg) / 261.63)
    notes = FrequencyAnalyzing.GetNotes(midi)
    b0 = np.mean(Interval)
    beat = 2 ** np.round(np.log2(Interval / b0))

    return midi