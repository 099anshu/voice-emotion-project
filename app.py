# app.py - Real-time + File Prediction (100% consistent with training)
import sounddevice as sd
from scipy.io.wavfile import write
import scipy.io.wavfile as wav
import numpy as np
import tensorflow as tf
import librosa
import os
import sys
import joblib
import tempfile
from datetime import datetime
import subprocess
import platform
import speech_recognition as sr

# Add utils to path
sys.path.append('utils')

# Load model
model = tf.keras.models.load_model("models/best_emotion_model.keras")

EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise']

# Same feature extraction as training


def extract_features(audio_path_or_array):
    """Extract features using the same method as training"""
    from utils.audio_preprocessing import extract_features as extract_training_features

    # Load the scaler used during training
    scaler = joblib.load("models/feature_scaler.pkl")

    if isinstance(audio_path_or_array, str):
        # Use the training feature extraction
        features = extract_training_features(
            audio_path_or_array, scaler=scaler)
        return features
    else:
        # For real-time audio array, save temporarily and process
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            wav.write(tmp.name, 22050,
                      (audio_path_or_array * 32767).astype(np.int16))
            features = extract_training_features(tmp.name, scaler=scaler)
            os.unlink(tmp.name)
        return features


def record_audio(duration=3, fs=22050):
    print("Recording... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs,
                       channels=1, dtype='float32')
    sd.wait()

    # Save the recording with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("recordings", exist_ok=True)
    filename = f"recordings/recording_{timestamp}.wav"
    wav.write(filename, fs, (recording * 32767).astype(np.int16))
    print(f"📁 Recording saved to: {filename}")

    return np.squeeze(recording), filename


def predict_emotion(features):
    features = features[np.newaxis, ...]  # (1, 174, 120)
    pred = model.predict(features, verbose=0)
    idx = np.argmax(pred)
    confidence = pred[0][idx]
    return EMOTIONS[idx], confidence


def transcribe_audio(filename):
    """Transcribe audio file to text using speech recognition"""
    try:
        # Initialize recognizer
        r = sr.Recognizer()
        
        # Load audio file
        with sr.AudioFile(filename) as source:
            # Adjust for ambient noise and record
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = r.record(source)
            
        # Transcribe using Google Speech Recognition
        try:
            text = r.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Speech recognition error: {e}"
            
    except Exception as e:
        return f"Transcription failed: {e}"


def list_recordings():
    """List all recordings with their file sizes and timestamps"""
    recordings_dir = "recordings"
    if not os.path.exists(recordings_dir):
        print("❌ No recordings directory found")
        return []
    
    recordings = []
    for file in os.listdir(recordings_dir):
        if file.endswith('.wav'):
            filepath = os.path.join(recordings_dir, file)
            file_size = os.path.getsize(filepath) / 1024  # KB
            # Extract timestamp from filename
            timestamp_str = file.replace('recording_', '').replace('.wav', '')
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            except:
                formatted_time = timestamp_str
            
            recordings.append({
                'filename': file,
                'filepath': filepath,
                'size': file_size,
                'timestamp': formatted_time
            })
    
    # Sort by timestamp (newest first)
    recordings.sort(key=lambda x: x['filename'], reverse=True)
    
    if recordings:
        print(f"\n📁 Found {len(recordings)} recordings:")
        print("-" * 70)
        for i, rec in enumerate(recordings, 1):
            print(f"{i:2}. {rec['filename']}")
            print(f"    📅 {rec['timestamp']} | 📊 {rec['size']:.1f} KB")
    else:
        print("❌ No recordings found")
    
    return recordings


def delete_recording(filepath):
    """Delete a specific recording file"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            filename = os.path.basename(filepath)
            print(f"✅ Deleted: {filename}")
            return True
        else:
            print("❌ Recording not found")
            return False
    except Exception as e:
        print(f"❌ Error deleting recording: {e}")
        return False


def clear_all_recordings():
    """Delete all recordings after confirmation"""
    recordings_dir = "recordings"
    if not os.path.exists(recordings_dir):
        print("❌ No recordings directory found")
        return
    
    recordings = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')]
    if not recordings:
        print("❌ No recordings to delete")
        return
    
    print(f"⚠️  This will delete ALL {len(recordings)} recordings permanently!")
    confirm = input("Are you sure? Type 'YES' to confirm: ").strip()
    
    if confirm == 'YES':
        deleted_count = 0
        for file in recordings:
            try:
                os.remove(os.path.join(recordings_dir, file))
                deleted_count += 1
            except Exception as e:
                print(f"❌ Error deleting {file}: {e}")
        
        print(f"✅ Deleted {deleted_count} recordings")
    else:
        print("❌ Operation cancelled")


def play_audio(filename):
    """Play audio file using system default player"""
    try:
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["afplay", filename], check=True)
        elif platform.system() == "Windows":
            subprocess.run(["start", filename], shell=True, check=True)
        else:  # Linux
            subprocess.run(["aplay", filename], check=True)
        print("🔊 Audio playback finished")
    except Exception as e:
        print(f"❌ Could not play audio: {e}")


def show_sample_sentences():
    """Display sample sentences for testing each emotion"""
    samples = {
        "😐 neutral": [
            "The weather is okay today.",
            "I need to go to the store.",
            "This is a simple statement."
        ],
        "😊 happy": [
            "I'm so excited about this amazing opportunity!",
            "This is the best day ever!",
            "I love spending time with my friends!"
        ],
        "😢 sad": [
            "I'm really disappointed about what happened.",
            "This makes me feel quite down.",
            "I wish things were different."
        ],
        "😠 angry": [
            "This is absolutely unacceptable!",
            "I can't believe this happened again!",
            "This really makes me furious!"
        ],
        "😨 fear": [
            "I'm really worried about this situation.",
            "This makes me quite nervous.",
            "I'm afraid something bad might happen."
        ],
        "😲 surprise": [
            "Oh wow, I can't believe this!",
            "This is completely unexpected!",
            "What a shocking turn of events!"
        ]
    }

    print("\n" + "="*60)
    print("🎭 SAMPLE SENTENCES FOR EMOTION TESTING")
    print("="*60)
    for emotion, sentences in samples.items():
        print(f"\n{emotion}:")
        for i, sentence in enumerate(sentences, 1):
            print(f"  {i}. \"{sentence}\"")
    print("="*60)


# ------------------- MAIN LOOP -------------------
print("🎤 Voice Emotion Recognition Ready!")
print("Commands:")
print("  • Press Enter - Record 3 seconds using your microphone")
print("  • 'file path/to/audio.wav' - Test any audio file")
print("  • 'samples' - Show sample sentences for testing")
print("  • 'replay' - Play the last recording again")
print("  • 'list' - Show all recordings")
print("  • 'delete [number]' - Delete a specific recording (use 'list' first)")
print("  • 'clear' - Delete all recordings")
print("  • 'quit' - Exit the program")

last_recording = None

while True:
    user_input = input("\n> ").strip().lower()

    if user_input == "":
        # Record using microphone
        audio, filename = record_audio()
        last_recording = filename
        print("Processing...")
        
        # Transcribe the audio
        print("🎤 Transcribing...")
        transcription = transcribe_audio(filename)
        
        features = extract_features(audio)

    elif user_input == "samples":
        show_sample_sentences()
        continue

    elif user_input == "replay":
        if last_recording and os.path.exists(last_recording):
            print(f"🔊 Replaying: {last_recording}")
            play_audio(last_recording)
        else:
            print("❌ No recording to replay")
        continue

    elif user_input == "list":
        list_recordings()
        continue

    elif user_input.startswith("delete "):
        try:
            # Extract number from command like "delete 3"
            parts = user_input.split()
            if len(parts) == 2 and parts[1].isdigit():
                recording_num = int(parts[1])
                recordings = list_recordings()
                if recordings and 1 <= recording_num <= len(recordings):
                    selected_recording = recordings[recording_num - 1]
                    delete_recording(selected_recording['filepath'])
                else:
                    print("❌ Invalid recording number. Use 'list' to see available recordings.")
            else:
                print("❌ Usage: delete [number]. Example: delete 3")
        except Exception as e:
            print(f"❌ Error: {e}")
        continue

    elif user_input == "clear":
        clear_all_recordings()
        continue

    elif user_input == "quit":
        print("👋 Goodbye!")
        break

    elif os.path.exists(user_input):
        print(f"Loading {user_input}...")
        audio, _ = librosa.load(user_input, sr=22050)
        features = extract_features(audio)

    else:
        print("❌ File not found or invalid command. Type 'samples' for help.")
        continue

    # Predict emotion
    emotion, conf = predict_emotion(features)

    # Show results with emoji
    emoji_map = {
        "neutral": "😐", "happy": "😊", "sad": "😢",
        "angry": "😠", "fear": "😨", "surprise": "😲"
    }

    # Show transcription if available (from recording)
    if 'transcription' in locals():
        print(f"\n💬 You said: \"{transcription}\"")
    
    print(
        f"✅ Predicted: {emoji_map.get(emotion, '🤔')} {emotion.upper()} (Confidence: {conf:.3f})")

    if conf < 0.6:
        print("⚠️  Low confidence - try speaking more clearly or with stronger emotion")
    elif conf > 0.8:
        print("🎯 High confidence prediction!")

    print("-" * 60)
