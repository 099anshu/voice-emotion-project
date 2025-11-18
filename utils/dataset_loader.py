import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from .audio_preprocessing import extract_features


def load_dataset(data_dir="data", n_mfcc=40, max_len=174):
    X, y = [], []
    # Fixed consistent emotion mapping (6 emotions)
    emotion_labels = ["neutral", "happy", "sad", "angry", "fear", "surprise"]
    label_map = {emotion: idx for idx, emotion in enumerate(emotion_labels)}

    print("Extracting features from audio files...")
    all_features = []

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

                # Extract features without normalization (with error handling for corrupted files)
                try:
                    # Skip very small files that are likely corrupted
                    file_size = os.path.getsize(file_path)
                    if file_size < 2000:  # Skip files smaller than 2KB
                        continue

                    features = extract_features(
                        file_path, n_mfcc=n_mfcc, max_len=max_len, scaler=None)
                    if isinstance(features, tuple):
                        # Get features only, ignore scaler
                        features = features[0]

                    # Ensure features have the expected shape
                    if features is not None and features.shape[0] > 0:
                        all_features.append(features)
                        y.append(label)
                except Exception as e:
                    print(f"Warning: Skipping corrupted file {file_path}: {e}")
                    continue

    # Convert to arrays
    X = np.array(all_features)
    y = np.array(y)

    # Now normalize all features together using a global scaler
    print("Normalizing features globally...")
    from sklearn.preprocessing import StandardScaler
    import joblib

    # Reshape for scaler
    original_shape = X.shape
    X_reshaped = X.reshape(-1, X.shape[-1])

    # Fit scaler on all data
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_reshaped)

    # Save scaler for inference
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/feature_scaler.pkl")
    print("💾 Saved feature scaler to models/feature_scaler.pkl")

    # Reshape back
    X = X_normalized.reshape(original_shape)
    y = to_categorical(y, num_classes=len(label_map))

    # Define emotion labels in order
    emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise']

    return X, y, emotion_labels
