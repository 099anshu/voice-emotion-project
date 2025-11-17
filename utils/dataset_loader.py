import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from .audio_preprocessing import extract_features

def load_dataset(data_dir="data", n_mfcc=40, max_len=174):
    X, y = [], []
    label_map = {"angry":0, "happy":1, "sad":2, "fear":3, "neutral":4, "surprise":5}

    for emotion in os.listdir(data_dir):
        emotion_path = os.path.join(data_dir, emotion)
        if not os.path.isdir(emotion_path) or emotion not in label_map:
            continue

        for file in os.listdir(emotion_path):
            if file.endswith(".wav"):
                file_path = os.path.join(emotion_path, file)

                # RAVDESS format
                if "-" in file:
                    parts = file.split("-")
                    emotion_code = parts[2]
                    ravdess_map = {
                        "01": "neutral",
                        "03": "happy",
                        "04": "sad",
                        "05": "angry",
                        "06": "fear",
                        "08": "surprise"
                    }
                    if emotion_code in ravdess_map:
                        label = label_map[ravdess_map[emotion_code]]
                    else:
                        continue

                # CREMA-D format
                elif "_" in file:
                    parts = file.split("_")
                    emotion_code = parts[2].upper()
                    crema_map = {
                        "NE": "neutral",
                        "HI": "happy",
                        "SA": "sad",
                        "AN": "angry",
                        "FE": "fear",
                        "SU": "surprise"
                    }
                    if emotion_code in crema_map:
                        label = label_map[crema_map[emotion_code]]
                    else:
                        continue
                else:
                    continue

                features = extract_features(file_path, n_mfcc=n_mfcc, max_len=max_len)
                X.append(features)
                y.append(label)

    X = np.array(X)
    y = np.array(y)
    y = to_categorical(y, num_classes=6)

    return X, y
