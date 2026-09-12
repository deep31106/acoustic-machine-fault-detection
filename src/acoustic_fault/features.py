import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr = None)

    features = {
        'RMS':librosa.feature.rms(y=y).mean(),
        'STD':np.std(y),
        'Variance':np.var(y),
        'ZCR':librosa.feature.zero_crossing_rate(y).mean(),
        'Peak':np.max(np.abs(y))
    }

    return features

def extract_frequency_features(audio_path):
    y, sr = librosa.load(audio_path, sr = None)

    stft = librosa.stft(
        y,
        n_fft = 1024,
        hop_length=256
    )

    magnitude = np.abs(stft)

    centroid = librosa.feature.spectral_centroid(
        S=magnitude,
        sr=sr
    ).mean()

    bandwidth = librosa.feature.spectral_bandwidth(
        S = magnitude,
        sr=sr
    ).mean()

    frequencies = librosa.fft_frequencies(
        sr=sr,
        n_fft = 1024
    )

    low_freq_mask = frequencies<=500

    low_freq_energy = np.sum(magnitude[low_freq_mask]**2)
    total_energy = np.sum(magnitude**2)
    low_freq_ratio = low_freq_energy/total_energy

    return {'spectral centroid':centroid, 'spectral bandwidth':bandwidth, 'low freq energy ratio':low_freq_ratio}