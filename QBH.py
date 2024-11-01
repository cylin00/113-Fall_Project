import wave
import numpy as np
import matplotlib.pyplot as plt

def GetAmplitude(audio, n0, startTime, endTime):
    segment = audio[int(startTime * frame_rate // n0):int(endTime * frame_rate // n0)]
    return segment

with wave.open('hummingdata/10/lugo_新不了情.wav', 'rb') as audio:
    
    # Step 0: Read the audio file and extract the amplitude data

    params = audio.getparams()
    num_channels, sample_width, frame_rate, num_frames = params[:4]
    
    frames = audio.readframes(num_frames)
    audio_data = np.frombuffer(frames, dtype=np.int16) 
    amplitude_data = abs(audio_data)
    time = np.linspace(0, len(audio_data) / frame_rate, num=len(audio_data))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6))

    # Step 1: Select the max-amplitude data in groups of n0 = 80

    n0 = 80
    A = np.zeros(len(amplitude_data) // n0)

    for idx, i in enumerate(range(0, len(amplitude_data), n0)):
        group = amplitude_data[i:i + n0]
        A[idx] = max(group)  
    
    # Step 2: Remove the background noise by setting the threshold value
        
    rho = 0
    for i in range(0, len(A)):
        if A[i] < rho:
            A[i] = 0

    # Step 3: Perform th normalization of A
                        
    mean_A = np.mean(A)
    for i in range(0, len(A)):
        A[i] = A[i] / (0.2 + 0.1 * mean_A)

    # Step 4: Take the fractional power for the envelope amplitude

    B = np.zeros(len(A))  
    l = 0.7
    for i in range(0, len(A)):
        B[i] = A[i] ** l

    # Step 5: Convolution of B with an envelope match filter
        
    f = [3, 3, 4, 4, -1, -1, -2, -2, -2, -2, -2, -2]
    Enveloped_Data = np.zeros(len(B))
    for i in range(0, len(B)):
        for j in range(0, 11):
            Enveloped_Data[i] += B[i - j] * f[j]
        # print(f"C{i}: {C[i]}")
            
    # Step 6: Find the onsets of C and > threshold value
    #print(f"data after chosen: {Enveloped_Data}")

    threshold = 15.3
    t = 0
    Predicted_onsets = []
    for i in range(1, len(Enveloped_Data) // 3 ):
        if (Enveloped_Data[3*i] > threshold) & (Enveloped_Data[3*i] > Enveloped_Data[3*i - 1]) & (Enveloped_Data[3*i] > Enveloped_Data[3*i - 2]):
            Predicted_onsets.append(3*i)
            t += 1
            # print(f"{t}-th predicted onsets = {Enveloped_Data[3*i]}, index = {3*i}")

    # Draw the graph with onsets
    onset_times = np.array(Predicted_onsets) * n0 / frame_rate
    ax1.plot(time, audio_data)
    for onset_time in onset_times:
         ax1.axvline(x=onset_time, color='red', linestyle='-', linewidth = 0.5, label='Onset' if onset_time == onset_times[0] else "")
    ax1.set_title('Amplitude vs. Time with Onsets')
    ax1.set_xlabel('time')
    ax1.set_ylabel('Amplitude')


    # Remove small intervals between onsets
    for o in range(1, len(Predicted_onsets)):
        previous_index = o - 1
    
        while previous_index >= 0 and Predicted_onsets[previous_index] == 0:
            previous_index -= 1 

        if previous_index >= 0:
            if Predicted_onsets[o] - Predicted_onsets[previous_index] < 0.15 * frame_rate // n0:
                Predicted_onsets[o] = 0


    # Remove the last onset if interval < 0.15 sec
    if Predicted_onsets[len(Predicted_onsets)-1] > (len(audio_data) - 0.15) * frame_rate // n0:
        Predicted_onsets[len(Predicted_onsets)-1] = 0

    # Draw the graph with new onsets
    onset_times = np.array(Predicted_onsets) * n0 / frame_rate

    # Store the onset times & onsets
    OnsetTimes = []
    for onset_time in onset_times:
        if onset_time != 0:
            OnsetTimes.append(onset_time)
            print(f"onset_time: {onset_time} sec")
    
    OnsetIndex = []
    for onsetIndex in Predicted_onsets:
        if onsetIndex != 0:
            OnsetIndex.append(onsetIndex)

    # print(f"length of onset indexs = {len(OnsetIndex)}")

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
            # print(f"Silence between {OnsetTimes[o-1]} and {OnsetTimes[o]}")
            
            amplitudes_in_silence = GetAmplitude(Enveloped_Data, n0, StartTime, EndTime)
            #print(f"Amplitude in silence: {amplitudes_in_silence}")

            # set adaptive threshold
            newThreshold = threshold - 0.5
            startIndex = 0
            for s in range(0, len(amplitudes_in_silence)-1):
                if(amplitudes_in_silence[s] >= newThreshold and (s - startIndex) >= (0.15 * frame_rate // n0)):
                    startIndex = s
                    ax3.axvline(x=StartTime + s * n0 / frame_rate, color='green', linestyle='-', linewidth = 0.5, label='Onset' if onset_time == onset_times[0] else "")
                    # print(f"new possible threshold = {amplitudes_in_silence[s]}, time = {StartTime + s * n0 / frame_rate}")
    
    # print(f"predicted number of onsets = {len(OnsetTimes)}")

    ax3.set_title('Amplitude vs. Time with Adaptive Threshold')
    ax3.set_xlabel('time')
    ax3.set_ylabel('Amplitude') 

    plt.tight_layout()
    #plt.show()

    # Discrete Fourier Transform 
    
    N = len(OnsetIndex)
    fs = 8000
    Indices = []
    DFTmagnitudes = []
    OriginalData = []
    SmoothData = []

    OnsetMag = []
    FundFreq = []

    for o in range(0, N-1):
        window_size = OnsetIndex[o+1] - OnsetIndex[o]
        segment = Enveloped_Data[o:o+window_size]
        X_m_segment = np.fft.fft(segment)
        freqs_segment = np.fft.fftfreq(len(segment), 1/fs)
        # print(f"Window size = {window_size}, X_m_segment: {X_m_segment}")

        # Add smooth filter to the DFT result
        L = 10
        smooth = [0] * window_size
        for i in range(0, window_size):
            if abs(i) < window_size / 2:
                smooth[i] = (L - abs(i)) / ( L ^ 2)
            else:
                smooth[i] = 0

        Xs = abs(X_m_segment) * smooth

        # Store the Points satisifying the three conditions after the smooth filter
        maxIndex = np.argmax(Xs)
        fund = fs 
        Fup = fs / 2
        Flow = 80
        print(f"len of M = {len(Xs)}")
        for i in range(0, len(Xs)):
            if i == 0 and Xs[i] > Xs[i+1] and Xs[i] >= Xs[maxIndex]/5:
                print(f"*, i = {i}, freq = {i * fs / window_size}")
                if (i * fs / window_size) <= Fup and (i * fs / window_size) >= Flow:
                    print(f"fundamental frequency = {i * fs / window_size}")
                    if i * fs / window_size < fund:
                        fund = i * fs / window_size
            else:
                if Xs[i] > Xs[i-1] and Xs[i] > Xs[i+1] and Xs[i] >= Xs[maxIndex]/5:
                    print(f"*, i = {i}, freq = {i * fs / window_size}")
                    if i * fs / window_size <= Fup and i * fs / window_size >= Flow:
                        print(f"fundamental frequency = {i * fs / window_size}")
                        if i * fs / window_size < fund:
                            fund = i * fs / window_size
        
        print(f"get fundamental frequency = {fund} between each onset")

        if o == 0:
            OnsetMag.append(Xs[0])
            FundFreq.append(0)
        else:
            if Xs[0] > OnsetMag[-1]:
                OnsetMag.append(Xs[0])
                FundFreq.append(o * fs / len(Xs))
            else:
                OnsetMag.append(0)
                FundFreq.append(0)

        magnitude = np.abs(X_m_segment)
        indices = np.arange(OnsetIndex[o], OnsetIndex[o+1])
        Indices.extend(indices)
        DFTmagnitudes.extend(magnitude)
        OriginalData.extend(segment)
        SmoothData.extend(Xs)

    
    print(f"Onset magnitude = {OnsetMag} of all onset")
    print(f"Fundamental frequency = {FundFreq} of all onset")
    FundaFreq = min([value for value in FundFreq if value != 0 and value > 80 and value < 4000])
    print(f"Fundamental frequency = {FundaFreq}")


    fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

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
    plt.show()