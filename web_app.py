from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import numpy as np
import tensorflow as tf
import librosa
import joblib
import tempfile
import speech_recognition as sr
from werkzeug.utils import secure_filename
from utils.audio_preprocessing import extract_features
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and scaler
print("🔄 Loading model...")
try:
    model = tf.keras.models.load_model("models/improved_cnn_lstm_emotion_model.keras")
    print("✅ Loaded improved CNN-LSTM model")
except:
    try:
        model = tf.keras.models.load_model("models/best_emotion_model.keras")
        print("✅ Loaded best emotion model")
    except:
        model = tf.keras.models.load_model("models/lstm_emotion_model.keras")
        print("✅ Loaded basic LSTM model")

try:
    scaler = joblib.load("models/feature_scaler.pkl")
    print("✅ Loaded feature scaler")
except:
    print("⚠️ Warning: Could not load feature scaler")
    scaler = None

# Emotion mapping (6 emotions only - no disgust)
idx_to_emotion = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "fear",
    5: "surprise"
}

emotion_emojis = {
    "neutral": "😐",
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😲"
}

# History storage
HISTORY_FILE = 'recording_history.json'

def load_history():
    """Load recording history from file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    """Save recording history to file"""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")

def add_to_history(filename, emotion, confidence, transcription, audio_url):
    """Add a new recording to history"""
    history = load_history()
    history.insert(0, {
        "id": len(history) + 1,
        "filename": filename,
        "emotion": emotion,
        "confidence": confidence,
        "transcription": transcription,
        "audio_url": audio_url,
        "timestamp": datetime.now().isoformat()
    })
    # Keep only last 100 recordings
    history = history[:100]
    save_history(history)
    return history

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_emotion(audio_path):
    """Predict emotion from audio file"""
    try:
        # Extract features
        features = extract_features(audio_path, scaler=scaler)
        features = features.reshape(1, features.shape[0], features.shape[1])
        
        # Predict
        prediction = model.predict(features, verbose=0)
        
        # Get results
        confidence_scores = prediction[0]
        predicted_class = int(np.argmax(prediction))
        
        # Get emotion from mapping
        if predicted_class in idx_to_emotion:
            emotion = idx_to_emotion[predicted_class]
        else:
            emotion = "unknown"
            print(f"⚠️ Warning: Predicted class {predicted_class} not in emotion mapping")
        
        confidence = float(np.max(confidence_scores))
        
        # Get all emotion scores (only 6 emotions, filter out disgust if present)
        all_scores = {}
        for idx, emo in idx_to_emotion.items():
            if idx < len(confidence_scores) and emo != "disgust":
                all_scores[emo] = float(confidence_scores[idx])
            elif emo != "disgust":
                all_scores[emo] = 0.0
        
        # If emotion is disgust (which shouldn't happen), map to the next highest
        if emotion == "disgust":
            # Find the highest non-disgust emotion
            filtered_scores = {k: v for k, v in all_scores.items() if k != "disgust"}
            if filtered_scores:
                emotion = max(filtered_scores, key=filtered_scores.get)
                confidence = filtered_scores[emotion]
                predicted_class = [idx for idx, emo in idx_to_emotion.items() if emo == emotion][0] if emotion in idx_to_emotion.values() else predicted_class
        
        # Debug output
        print(f"🎯 Predicted Emotion: {emotion} (class: {predicted_class}, confidence: {confidence:.4f})")
        print(f"📊 All scores (6 emotions): {all_scores}")
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "predicted_class": predicted_class,
            "all_scores": all_scores,
            "emoji": emotion_emojis.get(emotion, "🎭")
        }
    except Exception as e:
        print(f"❌ Error in predict_emotion: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def transcribe_audio(audio_path):
    """Transcribe audio to text"""
    try:
        r = sr.Recognizer()
        temp_wav_path = None
        
        # Convert to WAV if needed
        if not audio_path.endswith('.wav'):
            # Convert using librosa
            try:
                y, sr_rate = librosa.load(audio_path, sr=16000)
                import soundfile as sf
                temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_wav_path = temp_wav.name
                sf.write(temp_wav_path, y, sr_rate)
                audio_path = temp_wav_path
            except Exception as e:
                return {"text": f"Audio conversion error: {str(e)}", "success": False}
        
        try:
            with sr.AudioFile(audio_path) as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = r.record(source)
            
            try:
                text = r.recognize_google(audio_data)
                # Clean up temp file if created
                if temp_wav_path and os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)
                return {"text": text, "success": True}
            except sr.UnknownValueError:
                if temp_wav_path and os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)
                return {"text": "Could not understand audio. Please speak more clearly.", "success": False}
            except sr.RequestError as e:
                if temp_wav_path and os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)
                return {"text": f"Speech recognition service error. Please check your internet connection.", "success": False}
        except Exception as e:
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
            return {"text": f"Audio processing error: {str(e)}", "success": False}
    except Exception as e:
        return {"text": f"Transcription error: {str(e)}", "success": False}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/history')
def get_history():
    """Get recording history"""
    try:
        history = load_history()
        return jsonify({"history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predict emotion
            result = predict_emotion(filepath)
            if "error" in result:
                print(f"❌ Prediction error: {result['error']}")
                return jsonify(result), 500
            
            # Transcribe audio
            transcription = transcribe_audio(filepath)
            
            audio_url = f"/api/audio/{filename}"
            
            # Add to history
            add_to_history(
                filename=filename,
                emotion=result["emotion"],
                confidence=result["confidence"],
                transcription=transcription["text"],
                audio_url=audio_url
            )
            
            # Filter out disgust from all_scores if present
            filtered_scores = {k: v for k, v in result["all_scores"].items() if k != "disgust"}
            
            # Debug: Print what we're sending
            print(f"✅ Sending response: emotion={result['emotion']}, confidence={result['confidence']:.4f}")
            
            # Return results with file path for replay
            response = {
                "emotion": result["emotion"],
                "confidence": result["confidence"],
                "predicted_class": result.get("predicted_class", -1),
                "all_scores": filtered_scores,
                "emoji": result["emoji"],
                "transcription": transcription["text"],
                "transcription_success": transcription["success"],
                "audio_url": audio_url
            }
            
            print(f"📤 Response: {response}")
            return jsonify(response)
        else:
            return jsonify({"error": "Invalid file type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/audio/<filename>')
def serve_audio(filename):
    """Serve audio file for replay"""
    try:
        # Secure filename
        filename = secure_filename(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404
        
        # Determine MIME type based on file extension
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'wav'
        mime_types = {
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'm4a': 'audio/mp4',
            'flac': 'audio/flac',
            'ogg': 'audio/ogg',
            'webm': 'audio/webm'
        }
        mimetype = mime_types.get(ext, 'audio/wav')
        
        return send_file(filepath, mimetype=mimetype, as_attachment=False)
    except Exception as e:
        print(f"❌ Error serving audio: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Voice Emotion Recognition Web App...")
    print("📱 Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)

