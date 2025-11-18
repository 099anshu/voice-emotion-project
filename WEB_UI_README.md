# 🎭 Voice Emotion Recognition - Web UI

A beautiful, aesthetic web interface for voice emotion recognition with a cream and brown theme.

## ✨ Features

- 🎨 **Aesthetic Design**: Beautiful cream and brown color scheme with smooth animations
- 🎤 **Audio Recording**: Record audio directly from your browser
- 📁 **File Upload**: Upload audio files (WAV, MP3, M4A, FLAC, OGG)
- 🔊 **Audio Replay**: Play back your recordings with a built-in audio player
- 📝 **Speech Transcription**: Automatic transcription of spoken audio
- 📊 **Emotion Analysis**: Detailed emotion prediction with confidence percentages
- 🎯 **Visual Results**: Cute emojis and animated progress bars for each emotion
- 📱 **Responsive Design**: Works beautifully on desktop and mobile devices

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Web Server**:
   ```bash
   python web_app.py
   ```

3. **Open in Browser**:
   Navigate to `http://localhost:5000`

## 📋 Usage

### Recording Audio
1. Click the **"Start Recording"** button
2. Grant microphone permissions if prompted
3. Speak into your microphone
4. Click **"Stop Recording"** when done
5. The system will automatically analyze your voice

### Uploading Audio
1. **Drag and Drop**: Drag an audio file onto the upload area
2. **Click to Browse**: Click the upload area to select a file
3. **File Browser**: Use the "Choose File" button

### Viewing Results
- **Transcription**: See what was said in the audio
- **Predicted Emotion**: Main emotion with confidence percentage
- **All Emotions**: Breakdown of all emotions with their confidence scores
- **Replay Audio**: Click the replay button or use the audio player controls

## 🎨 Design Features

- **Cream & Brown Theme**: Elegant color palette
- **Smooth Animations**: Fade-in, bounce, pulse, and scale effects
- **Interactive Elements**: Hover effects and visual feedback
- **Cute Emojis**: Emoji representations for each emotion
- **Progress Bars**: Animated confidence bars
- **Responsive Grid**: Adapts to different screen sizes

## 🔧 Technical Details

- **Backend**: Flask with RESTful API
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Audio Processing**: librosa, soundfile
- **Speech Recognition**: Google Speech Recognition API
- **Model**: Trained CNN-LSTM emotion recognition model

## 📁 Project Structure

```
voice-emotion-project/
├── web_app.py           # Flask backend server
├── templates/
│   └── index.html      # Main HTML template
├── static/
│   ├── style.css       # Stylesheet
│   ├── script.js       # Frontend JavaScript
│   └── uploads/        # Uploaded audio files
└── models/             # Trained model files
```

## 🎯 Supported Emotions

- 😐 Neutral
- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fear
- 😲 Surprise
- 🤢 Disgust

## 🔐 Notes

- **Internet Required**: Speech transcription uses Google Speech Recognition API (requires internet)
- **Microphone Permissions**: Browser will request microphone access for recording
- **File Size Limit**: Maximum 16MB file size
- **Audio Formats**: WAV, MP3, M4A, FLAC, OGG

## 🐛 Troubleshooting

- **Model Not Loading**: Ensure models are in the `models/` directory
- **Transcription Fails**: Check internet connection (Google Speech Recognition requires internet)
- **Audio Won't Play**: Ensure browser supports HTML5 audio
- **Recording Fails**: Check microphone permissions in browser settings

## 💡 Tips

- Speak clearly for better transcription accuracy
- Use quiet environments for better emotion detection
- Record at least 1-2 seconds of audio for best results
- Test with different emotions to see the confidence scores

Enjoy your beautiful voice emotion recognition system! 🎉

