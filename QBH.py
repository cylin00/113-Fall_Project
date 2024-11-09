import numpy as np
import matplotlib.pyplot as plt
import AudioProcessing.Audio as Audio
import DataProcessing.DataProcessing as Data
import DataProcessing.FrequencyAnalyzing as Freq

(amplitude_data, time, frame_rate, audio_data) = Audio.AudioReader.GetAudioData(10, 'lugo_火車快飛.wav')

def GetAmplitude(audio, n0, startTime, endTime):
    segment = audio[int(startTime * frame_rate // n0):int(endTime * frame_rate // n0)]
    return segment

# Select the max-amplitude data in groups of n0 = 80
n0 = 80
A = Data.DataProcessing.GetMaxAmplitude(n0, amplitude_data)
A = Data.DataProcessing.NormalizeData(A) # Normalization of A : mean_A -> A
B = Data.DataProcessing.FractionalData(A, 0.7) # Take the fractional power

f = [3, 3, 4, 4, -1, -1, -2, -2, -2, -2, -2, -2]
Enveloped_Data = Data.DataProcessing.EnvelopData(B, f) # Convolution of B with an envelope match filter
        
threshold = 15.4 # 15.3
Predicted_onsets = []
for i in range(1, len(Enveloped_Data) // 3 ): # Find the onsets of EnvData and > threshold value
    if (Enveloped_Data[3*i] > threshold) & (Enveloped_Data[3*i] > Enveloped_Data[3*i - 1]) & (Enveloped_Data[3*i] > Enveloped_Data[3*i - 2]):
        Predicted_onsets.append(3*i)
onset_times = np.array(Predicted_onsets) * n0 / frame_rate

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6))

ax1.plot(time, audio_data) # Draw the graph with onsets
for onset_time in onset_times:
        ax1.axvline(x=onset_time, color='red', linestyle='-', linewidth = 0.5, label='Onset' if onset_time == onset_times[0] else "")
ax1.set_title('Amplitude vs. Time with Onsets')
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')


# Remove small intervals between onsets
# TmptOnset = Predicted_onsets
print(f"frame rate = {frame_rate}")
Predicted_onsets[0] = 0
for o in range(1, len(Predicted_onsets)):

    previous_index = o - 1
    print(f"o = {o}, interval: {(Predicted_onsets[o] - Predicted_onsets[previous_index]) * n0 / frame_rate} sec")
    while previous_index >= 0 and Predicted_onsets[previous_index] == 0:
        previous_index -= 1 

    if previous_index >= 0:
        print(f"previous index = {previous_index}")
        if Predicted_onsets[o] - Predicted_onsets[previous_index] < 0.15 * frame_rate // n0 :
            # print(f"interval: {(Predicted_onsets[o] - Predicted_onsets[previous_index]) * n0 / frame_rate}")
            Predicted_onsets[o] = 0

# Remove the last onset if interval < 0.15 sec
if Predicted_onsets[len(Predicted_onsets)-1] > (len(audio_data) - 0.15) * frame_rate // n0:
    Predicted_onsets[len(Predicted_onsets)-1] = 0

OnsetTimes = [] # Store the onset times & onsets
for onset_time in Predicted_onsets:
    if onset_time != 0:
        OnsetTimes.append(onset_time * n0 / frame_rate)
        print(f"onset_time: {onset_time} sec")

OnsetIndex = []
for onsetIndex in Predicted_onsets:
    if onsetIndex != 0:
        OnsetIndex.append(onsetIndex)

ax2.plot(time, audio_data)
for onset_time in OnsetTimes:
    ax2.axvline(x=onset_time, color='red', linestyle='-', linewidth = 0.5, label='Onset' if onset_time == onset_times[0] else "")
ax2.set_title('Amplitude vs. Time with Onsets (After Filtered)')
ax2.set_xlabel('time')
ax2.set_ylabel('Amplitude')


ax3.plot(time, audio_data)
for onset_time in OnsetTimes:
    ax3.axvline(x=onset_time, color='red', linestyle='-', linewidth = 0.5, label='Onset' if onset_time == onset_times[0] else "")

EnvelopAmplitudeThreshold = 0.02
for o in range(0, len(OnsetTimes)-1):
    SilenceDuration = OnsetTimes[o+1] - OnsetTimes[o]
    if SilenceDuration >= 1.5:
        StartTime = OnsetTimes[o]
        EndTime = OnsetTimes[o+1]
        
        amplitudes_in_silence = GetAmplitude(Enveloped_Data, n0, StartTime, EndTime)
        
        newThreshold = threshold - 0.5 # set adaptive threshold
        startIndex = 0
        for s in range(0, len(amplitudes_in_silence)-1):
            if(amplitudes_in_silence[s] >= newThreshold and (s - startIndex) >= (0.15 * frame_rate // n0)):
                startIndex = s
                ax3.axvline(x=StartTime + s * n0 / frame_rate, color='green', linestyle='-', linewidth = 0.5, label='Onset' if onset_time == onset_times[0] else "")

ax3.set_title('Amplitude vs. Time with Adaptive Threshold')
ax3.set_xlabel('time')
ax3.set_ylabel('Amplitude') 
plt.tight_layout()

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
    (X_m_segment, freqs_segment) = Freq.FrequencyAnalyzing.DFT(segment, frame_rate)
    
    Xs = Freq.FrequencyAnalyzing.Smooth(X_m_segment, 20) # Add smooth filter to the DFT result

    # Store the Points satisifying the three conditions after the smooth filter
    maxIndex = np.argmax(Xs)
    fund = frame_rate
    Fup = frame_rate / 2
    Flow = 80
    for i in range(0, len(Xs)):
        if i == 0 and Xs[i] > Xs[i+1] and Xs[i] >= Xs[maxIndex]/5:
            # print(f"*, i = {i}, freq = {i * frame_rate / window_size}")
            if freqs_segment[i] <= Fup and freqs_segment[i] >= Flow:
                # print(f"fundamental frequency = {freqs_segment[i]}")
                if freqs_segment[i] < fund:
                    fund = freqs_segment[i]
        else:
            if Xs[i] > Xs[i-1] and Xs[i] > Xs[i+1] and Xs[i] >= Xs[maxIndex]/5:
                # print(f"*, i = {i}, freq = {i * frame_rate / window_size}")
                if freqs_segment[i] <= Fup and freqs_segment[i] >= Flow:
                    # print(f"fundamental frequency = {i * frame_rate / window_size}")
                    if freqs_segment[i] < fund:
                        fund = i * freqs_segment[i]
    
    print(f"get fundamental frequency = {fund} between each onset")

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

print(f"Fundamental frequency = {FundFreq} of all onset")
print(f"Fundamental frequency test = {FreqSeg}")


fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

axs[0].plot(Indices, OriginalData, color='orange', label='Original Data')
for onset in OnsetIndex:
    axs[0].axvline(x=onset, color='green', linestyle='--', alpha=0.6)
    axs[0].text(onset, max(OriginalData) * 0.9, 'Onset', color='green', rotation=90, ha='right', va='center')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Original Data with Onset Lines')
axs[0].legend()

axs[1].plot(Indices, DFTmagnitudes, color='blue', label='DFT Magnitude')
for onset in OnsetIndex:
    axs[1].axvline(x=onset, color='green', linestyle='--', alpha=0.6)
    axs[1].text(onset, max(DFTmagnitudes) * 0.9, 'Onset', color='green', rotation=90, ha='right', va='center')
axs[1].set_ylabel('Amplitude')
axs[1].set_title('DFT Magnitude with Onset Lines')
axs[1].legend()

axs[2].plot(Indices, SmoothData, color='red', label='Smoothed DFT Magnitude')
for onset in OnsetIndex:
    axs[2].axvline(x=onset, color='green', linestyle='--', alpha=0.6)
    axs[2].text(onset, max(SmoothData) * 0.9, 'Onset', color='green', rotation=90, ha='right', va='center')
axs[2].set_xlabel('Original Data Index')
axs[2].set_ylabel('Amplitude')
axs[2].set_title('Smoothed DFT Magnitude with Onset Lines')
axs[2].legend()
plt.tight_layout()

# Freg Seq -> fundamental frequency of each onset
# OnsetIndex -> onset index

midi = 48 + 12 * np.log2(np.array(FreqSeg) / 261.63)
print(f"midi number = {midi}")

notes = Freq.FrequencyAnalyzing.GetNotes(midi)
print(f"notes = {notes}")

b0 = np.mean(Interval)
print(f"mean onset interval = {b0} = {b0 * n0 / frame_rate} sec")


beat = 2 ** np.round(np.log2(Interval / b0))
print(f"beat = {beat}")

plt.show()