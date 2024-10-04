import wave
import numpy as np
import matplotlib.pyplot as plt

with wave.open('hummingdata/10/lugo_朋友.wav', 'rb') as audio:
    
    # Step 0: Read the audio file and extract the amplitude data

    params = audio.getparams()
    num_channels, sample_width, frame_rate, num_frames = params[:4]
    
    frames = audio.readframes(num_frames)
    audio_data = np.frombuffer(frames, dtype=np.int16) 
    amplitude_data = abs(audio_data)

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
    C = np.zeros(len(B))
    for i in range(0, len(B)):
        for j in range(0, 11):
            C[i] += B[i - j] * f[j]
        # print(f"C{i}: {C[i]}")
            
    # Step 6: Find the onsets of C and > threshold value
    
    threshold = 15
    t = 0
    Predicted_onsets = []
    Predicted_onsets_values = []
    for i in range(1, len(C) // 3 ):
        if (C[3*i] > threshold) & (C[3*i] > C[3*i - 1]) & (C[3*i] > C[3*i - 2]):
            Predicted_onsets.append(3*i)
            Predicted_onsets_values.append(C[3*i])
            t += 1
    print(f"Number of onsets: {t}")

    # Draw the graph with onsets
    onset_times = np.array(Predicted_onsets) * n0 / frame_rate

    plt.figure(figsize=(10, 4))
    plt.plot(time, audio_data, label='Amplitude')

    for onset_time in onset_times:
        plt.axvline(x=onset_time, color='red', linestyle='-', label='Onset' if onset_time == onset_times[0] else "")

    plt.title('Amplitude vs. Time with Onsets')
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')
    plt.grid()
    plt.legend()
    plt.show()

    # Delete small intervals between onsets