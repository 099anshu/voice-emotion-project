import librosa
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler


def extract_features(path, n_mfcc=40, max_len=174, scaler=None):
    """
    Extract comprehensive audio features including MFCC, spectral features, and temporal features.
    Output shape: (max_len, n_features)
    """
    # Load audio with consistent sample rate
    # Standardize to 3 seconds
    audio, sr = librosa.load(path, sr=22050, duration=3.0)

    # Remove silence from beginning and end
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Normalize audio
    audio = librosa.util.normalize(audio)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=512, n_fft=2048)

    # Extract additional features for better emotion recognition
    # Spectral centroid (brightness)
    spectral_centroids = librosa.feature.spectral_centroid(
        y=audio, sr=sr, hop_length=512)

    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, hop_length=512)

    # Zero crossing rate (voice/unvoiced)
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=512)

    # Root Mean Square Energy
    rmse = librosa.feature.rms(y=audio, hop_length=512)

    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr, hop_length=512)

    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=512)

    # Combine all features
    features = np.concatenate([
        mfcc,
        spectral_centroids,
        spectral_rolloff,
        zcr,
        rmse,
        spectral_bandwidth,
        chroma
    ], axis=0)

    # Transpose to get (time_steps, n_features)
    features = features.T

    # Pad or truncate to max_len
    if len(features) < max_len:
        pad_width = max_len - len(features)
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
    else:
        features = features[:max_len, :]

    # Return features with or without normalization
    if scaler is None:
        # Training mode: return raw features (will be normalized globally)
        return features
    else:
        # Inference mode: use existing scaler
        features_reshaped = features.reshape(-1, features.shape[-1])
        features = scaler.transform(features_reshaped).reshape(features.shape)
        return features


def augment_mfcc(features):
    """Add small Gaussian noise for data augmentation."""
    noise = np.random.normal(0, 0.01, features.shape)
    return features + noise
