import wave
import numpy as np

with wave.open('hummingdata/10/lugo_朋友.wav', 'rb') as audio:
    
    # Step 0: Read the audio file and extract the amplitude data

    params = audio.getparams()
    num_channels, sample_width, frame_rate, num_frames = params[:4]
    
    frames = audio.readframes(num_frames)
    audio_data = np.frombuffer(frames, dtype=np.int16) 
    amplitude_data = abs(audio_data)
    
    # Step 1: Select the max-amplitude data in groups of n0 = 80

    n0 = 80

    for i in range(0, len(amplitude_data), n0):
        group = amplitude_data[i:i + n0]
        A = max(group)
    
        print(f"Group {i // n0 }, Max Amplitude: {A}")