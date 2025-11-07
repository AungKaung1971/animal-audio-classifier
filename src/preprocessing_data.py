# standardizing and preprocssing the data
# sample rate of 16 KHz
# clipped to duration of five seconds

import librosa
from scipy.signal import butter, filtfilt
import numpy as np
import os
import matplotlib.pyplot as plt


# resample audio function
def resample_audio(full_path, target_sr=16000):
    y, sr = librosa.load(full_path, sr=target_sr)
    return y, sr


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
            print(f"{file} >>> resampled at {sr}")

filtered_audio = low_pass_filter(y, cutoff_freq=4000, sample_rate=sr)
