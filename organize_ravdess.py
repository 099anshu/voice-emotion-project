import os
import shutil

SOURCE = "Audio_Speech_Actors_01-24"
DEST = "data"

emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

for actor in os.listdir(SOURCE):
    actor_folder = os.path.join(SOURCE, actor)
    
    if not os.path.isdir(actor_folder):
        continue
    
    for file in os.listdir(actor_folder):
        if file.endswith(".wav"):
            parts = file.split("-")
            emotion_code = parts[2]
            
            if emotion_code in emotion_map:
                emotion_folder = os.path.join(DEST, emotion_map[emotion_code])
                os.makedirs(emotion_folder, exist_ok=True)
                
                src = os.path.join(actor_folder, file)
                dst = os.path.join(emotion_folder, file)
                
                shutil.copy(src, dst)

print("DONE! Files are organized into /data/")
