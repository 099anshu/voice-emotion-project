import os
import shutil

# SOURCE folder: where you unzipped CREMA-D
SOURCE = "AudioWAV"  # Change to your CREMA-D folder path
DEST = "data"

# CREMA-D emotion codes
emotion_map = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fear",
    "SU": "surprise",
    "DIS": "disgust"  # optional if you want to skip, can ignore
}

# Iterate through all files in CREMA-D
for file in os.listdir(SOURCE):
    if file.endswith(".wav"):
        parts = file.split("_")  # CREMA-D filenames use underscores
        emotion_code = parts[2]  # third part is emotion code

        if emotion_code in emotion_map:
            emotion_folder = os.path.join(DEST, emotion_map[emotion_code])
            os.makedirs(emotion_folder, exist_ok=True)

            src = os.path.join(SOURCE, file)
            dst = os.path.join(emotion_folder, file)
            shutil.copy(src, dst)

print("DONE! CREMA-D files are organized into /data/")
