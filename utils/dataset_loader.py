import os
from utils.audio_preprocessing import extract_features
import numpy as np

EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "surprise"]

def load_dataset(dataset_path="data"):
    X = []
    y = []
    
    for emotion in EMOTIONS:
        folder = os.path.join(dataset_path, emotion)
        if not os.path.exists(folder):
            continue
        
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                file_path = os.path.join(folder, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(EMOTIONS.index(emotion))
    
    return np.array(X), np.array(y)
