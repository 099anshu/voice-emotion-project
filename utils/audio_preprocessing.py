import librosa
import numpy as np

def extract_features(path, n_mfcc=40, max_len=174):
    """
    Extract MFCC features from an audio file and pad/truncate to max_len.
    Output shape: (max_len, n_mfcc)
    """
    audio, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T  # shape: (time_steps, n_mfcc)

    # Pad or truncate
    if len(mfcc) < max_len:
        pad_width = max_len - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len, :]

    return mfcc

def augment_mfcc(features):
    """Add small Gaussian noise for data augmentation."""
    noise = np.random.normal(0, 0.01, features.shape)
    return features + noise
