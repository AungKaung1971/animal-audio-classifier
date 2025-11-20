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

# # low pass filter function


# def low_pass_filter(data, cutoff_freq, sample_rate, order=4):
#     nyquist = 0.5 * sample_rate
#     normal_cutoff = cutoff_freq / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     filtered_data = filtfilt(b, a, data)
#     print(f"Filtered audio shape: {filtered_data.shape}")
#     return filtered_data


def normalize_audio(y):
    y = y/np.max(np.abs(y))
    return y


# def extract_mfcc(y, sr, n_mfcc=40):
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     return mfcc


def extract_mel_spectrogram(y, sr, n_mels=128, n_fft=1024, hop_length=128):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


master_path = "data/raw"
save_root = "data/processed"

os.makedirs(save_root, exist_ok=True)

for root, dirs, files in os.walk(master_path):
    for file in files:
        if file.endswith(".wav"):
            full_path = os.path.join(root, file)

            class_name = os.path.basename(root)
            class_out = os.path.join(save_root, class_name)
            os.makedirs(class_out, exist_ok=True)

            y, sr = resample_audio(full_path)
            print(f"{file} >>> resampled at {sr}Hz")

            y = normalize_audio(y)

            y = convert_to_input(y, target_length=sr * 5)

            # mfcc = extract_mfcc(y, sr)

            mel = extract_mel_spectrogram(y, sr)

            out_path = os.path.join(class_out, file.replace(".wav", ".npy"))
            # np.save(out_path, mfcc) *if you want mfcc
            np.save(out_path, mel)

            # print(f"{file} saved → {out_path} | MFCC shape {mfcc.shape}")
            print(f"{file} saved -> {out_path} | mel shape {mel.shape}")
