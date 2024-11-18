import wave
import os
import numpy as np

def GetAudioData(duration, audio):

    audio_string = os.path.join('hummingdata', str(duration), audio)

    with wave.open(audio_string, 'rb') as audio:

        params = audio.getparams()
        frame_rate, num_frames = params[2:4]
        
        frames = audio.readframes(num_frames)
        audio_data = np.frombuffer(frames, dtype=np.int16) 
        amplitude_data = abs(audio_data)
        time = np.linspace(0, len(audio_data) / frame_rate, num=len(audio_data))

    return (amplitude_data, time, frame_rate, audio_data)