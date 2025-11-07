# standardizing and preprocssing the data
# sample rate of 16 KHz
# clipped to duration of five seconds

import librosa
from scipy.signal import butter, filtfilt
import numpy as np
import os
import matplotlib.pyplot as plt


# convert audio data into model's expected input
def convert_to_input(y, target_length):
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
        return y


# resample audio function
def resample_audio(full_path, target_sr=16000):
    y, sr = librosa.load(full_path, sr=target_sr)
    return y, sr

# low pass filter function


def low_pass_filter(data, cutoff_freq, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    print(f"Filtered audio shape: {filtered_data.shape}")
    return filtered_data


master_path = "data/raw"

for root, dirs, files in os.walk(master_path):
    for file in files:
        if file.endswith((".wav")):
            full_path = os.path.join(root, file)
            y, sr = resample_audio(full_path)
            print(f"{file} >>> resampled at {sr}Hz")

filtered_audio = low_pass_filter(y, cutoff_freq=4000, sample_rate=sr)

model_input = convert_to_input(filtered_audio, target_length=16000)
print(f"Model input shape: {model_input.shape}")
