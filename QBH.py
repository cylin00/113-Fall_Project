import wave
import numpy as np

with wave.open('hummingdata/10/lugo_朋友.wav', 'rb') as audio:
    
    params = audio.getparams()
    num_channels, sample_width, frame_rate, num_frames = params[:4]
    
    print(f"num_channels = {num_channels}")
    print(f"sample_width = {sample_width}")
    print(f"frame_rate = {frame_rate}")
    print(f"num_frames = {num_frames}")


    frames = audio.readframes(num_frames)

    audio_data = np.frombuffer(frames, dtype=np.int16)  # dtype depends on sample width
    
    print(f"Audio Data: {audio_data[:10]}")
    print(f"Number of Channels: {num_channels}")
    print(f"Frame Rate: {frame_rate}")
