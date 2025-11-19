# 🎭 Voice Emotion Recognition System

**Transform audio into emotional insights with AI-powered deep learning**

A production ready voice emotion recognition system that detects six distinct emotions from audio using advanced CNN-LSTM neural networks. Built with TensorFlow and trained on industry-standard RAVDESS and CREMA-D datasets.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-2.x-green)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌟 What is Voice Emotion Recognition?

Voice Emotion Recognition leverages cutting-edge deep learning to analyze audio recordings and identify the speaker's emotional state. By combining **CNN** layers for feature extraction with **LSTM** networks for temporal pattern recognition, the system achieves robust emotion detection across diverse speaking styles and environments.

**Core Concept:**
- **Input:** Audio recording (3 seconds) or uploaded file
- **Process:** Feature extraction → CNN-LSTM model → Multi-head attention
- **Output:** Emotion prediction with confidence scores + transcription

---

## ✨ Key Features

### 🎯 Intelligent Emotion Detection
Detects **six core emotions** with high accuracy: Neutral, Happy, Sad, Angry, Fear, and Surprise.

### 🌐 Dual Interface Options
- **Web UI:** Beautiful, responsive interface with real-time visualization
- **CLI Tool:** Command-line interface for quick testing and automation

### 🎤 Flexible Audio Input
- Record directly from microphone (3-second clips)
- Upload audio files (WAV, MP3, M4A, FLAC, OGG)
- Built-in audio playback and history tracking

### 📝 Speech Transcription
Automatic transcription using Google Speech Recognition API to see what was spoken alongside emotion detection.

### 📊 Comprehensive Analytics
- Confidence scores for all emotions
- Visual progress bars and emoji representations
- Recording history with timestamps

---

## 🏗️ Architecture Overview

### Model Stack
| Component | Technology |
|-----------|-----------|
| Feature Extraction | MFCC (40 coefficients) + Spectral Features |
| Neural Network | CNN-LSTM with Multi-Head Attention |
| Optimizer | Adam (adaptive learning rate) |
| Regularization | Dropout + Batch Normalization + L2 |
| Augmentation | Noise, Time Shifting, Masking |

### Application Stack
| Layer | Technology |
|-------|-----------|
| Web Backend | Flask + CORS |
| Frontend | HTML5 + CSS3 + Vanilla JS |
| CLI | Python + SoundDevice |
| Audio Processing | Librosa + SoundFile |
| Model Format | Keras (.keras) |

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8+
- Microphone (for recording)
- Internet connection (for transcription)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/voice-emotion-recognition.git
cd voice-emotion-recognition

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Option 1: Web Interface

```bash
# Start the web server
python web_app.py

# Open browser and navigate to
http://localhost:5000
```

### Option 2: CLI Application

```bash
# Run the command-line interface
python app.py

# Commands:
# - Press Enter: Record 3 seconds
# - 'file path/to/audio.wav': Test audio file
# - 'samples': View sample sentences
# - 'list': Show recording history
# - 'quit': Exit
```

---

## 📁 Project Structure

```
voice-emotion-recognition/
├── 🎯 Core Applications
│   ├── app.py                    # CLI interface
│   ├── web_app.py                # Flask web server
│   └── train.py                  # Model training script
│
├── 🎨 Web Interface
│   ├── templates/
│   │   └── index.html            # Main web UI
│   └── static/
│       ├── style.css             # Styling
│       ├── script.js             # Frontend logic
│       └── uploads/              # User recordings
│
├── 🧠 Model & Data
│   ├── models/                   # Trained models
│   │   ├── best_emotion_model.keras
│   │   ├── feature_scaler.pkl
│   │   ├── confusion_matrix.png
│   │   └── training_history.png
│   └── data/                     # Organized datasets
│       ├── angry/
│       ├── happy/
│       ├── sad/
│       ├── fear/
│       ├── neutral/
│       └── surprise/
│
├── ⚙️ Utilities
│   └── utils/
│       ├── audio_preprocessing.py
│       └── dataset_loader.py
│
└── 🔧 Configuration
    ├── requirements.txt
    ├── .gitignore
    └── README.md
```

---

## 📊 Dataset Setup

### Step 1: Download Datasets

**RAVDESS Dataset** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Download Link:** [https://zenodo.org/records/1188976](https://zenodo.org/records/1188976)
- **Extract to:** `Audio_Speech_Actors_01-24/`
- **Size:** ~24 professional actors, 1,440 audio files
- **Format:** 16-bit WAV files at 48kHz

**CREMA-D Dataset** (Crowd-sourced Emotional Multimodal Actors Dataset)
- **Download Link:** [https://github.com/CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- **Extract to:** `AudioWAV/`
- **Size:** 91 actors, 7,442 audio files
- **Format:** WAV files with varying sample rates

### Step 2: Organize Files

```bash
# Organize RAVDESS dataset into emotion folders
python organize_ravdess.py

# Organize CREMA-D dataset into emotion folders
python organize_crema_d.py
```

### Step 3: Verify Structure

Your `data/` folder should contain:
```
data/
├── angry/       (~1,200 files)
├── happy/       (~1,200 files)
├── sad/         (~1,200 files)
├── fear/        (~1,200 files)
├── neutral/     (~1,200 files)
└── surprise/    (~1,200 files)
```

---

## 🏋️ Training the Model

```bash
# Start training (100 epochs with early stopping)
python train.py
```

**Training Process:**
1. ✅ Load and organize ~7,000+ audio samples
2. ✅ Extract 120 audio features per sample
3. ✅ Apply data augmentation (2x factor)
4. ✅ Train CNN-LSTM with attention mechanism
5. ✅ Save best model based on validation accuracy
6. ✅ Generate confusion matrix and training plots

**Output Files:**
- `models/best_emotion_model.keras` - Best performing model
- `models/feature_scaler.pkl` - Feature normalization scaler
- `models/confusion_matrix.png` - Evaluation metrics
- `models/training_history.png` - Loss/accuracy curves

---

## 🎯 Use Cases

### For Research & Education
- Study emotion detection algorithms
- Experiment with different architectures
- Analyze feature importance in emotion recognition

### For Product Development
- Voice assistants with emotion awareness
- Customer service sentiment analysis
- Mental health monitoring applications

### For Content Creators
- Analyze emotional tone in podcasts
- Evaluate voice-over performances
- Quality control for emotional delivery

---

## 🔧 Technical Highlights

### Advanced Feature Extraction (120 features)
- **40 MFCC coefficients** - Voice characteristics
- **Spectral features** - Frequency domain analysis
- **Temporal features** - Energy and rhythm patterns
- **Chroma features** - Pitch class profiles

### Model Architecture
```
Input (174 timesteps, 120 features)
    ↓
CNN Blocks (64→128→256 filters)
    ↓
Bidirectional LSTM (128→64 units)
    ↓
Multi-Head Attention (4 heads)
    ↓
Dense Layers (256→128)
    ↓
Output (6 emotions, softmax)
```

### Training Optimizations
- **Class balancing** with computed weights
- **Learning rate scheduling** (reduce on plateau)
- **Data augmentation** (noise, shift, masking)
- **Early stopping** (patience: 15 epochs)

---

## 🌐 Web Interface Features

- **🎨 Aesthetic Design:** Cream and brown theme with smooth animations
- **🎙️ Direct Recording:** In-browser audio capture
- **📁 Drag & Drop:** Easy file uploads
- **📝 Transcription:** See what was spoken
- **📊 Visual Analytics:** Progress bars and emoji feedback
- **📜 History Tracking:** Review past recordings
- **🔄 Audio Replay:** Built-in player for saved recordings

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Microphone not working | Check browser/system permissions |
| Model not found | Run `python train.py` first |
| Import errors | Reinstall: `pip install -r requirements.txt` |
| Transcription fails | Verify internet connection |
| Low accuracy | Ensure dataset is properly organized |

---

## 📈 Model Performance

- **Architecture:** CNN-LSTM with Multi-Head Attention
- **Training Samples:** 7,000+ augmented to 14,000+
- **Features per Sample:** 120 (MFCC + spectral + temporal)
- **Input Shape:** (174 timesteps, 120 features)
- **Output Classes:** 6 emotions
- **Optimizer:** Adam with adaptive learning rate
- **Regularization:** Dropout (0.3-0.5) + L2 + Batch Norm

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/Enhancement`)
3. 💾 Commit changes (`git commit -m 'Add enhancement'`)
4. 📤 Push to branch (`git push origin feature/Enhancement`)
5. 🔄 Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

**Datasets:**
- **RAVDESS:** Livingstone SR, Russo FA (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). [https://zenodo.org/records/1188976](https://zenodo.org/records/1188976)
- **CREMA-D:** Cao H, Cooper DG, Keutmann MK, Gur RC, Nenkova A, Verma R (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset. [https://github.com/CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)

**Technologies:**
- TensorFlow/Keras - Deep learning framework
- Librosa - Audio feature extraction
- Flask - Web application framework

---

## 🔮 Roadmap

- [ ] Real-time streaming emotion detection
- [ ] Multi-language support
- [ ] Speaker identification
- [ ] Emotion intensity levels
- [ ] REST API for integration
- [ ] Docker containerization
- [ ] Mobile app (iOS/Android)

---

## 👨‍💻 Author

Built with ❤️ for the AI and machine learning community

**⭐ Star this repo if you find it helpful!**

---

**Voice Emotion Recognition - Where Audio Meets Emotional Intelligence**
