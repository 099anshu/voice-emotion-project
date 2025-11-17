# 🎤 Voice Emotion Recognition System

A real-time voice emotion recognition system using LSTM neural networks to detect six distinct emotions from audio recordings. Built with TensorFlow/Keras and trained on RAVDESS and CREMA-D datasets.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Features

- **Real-time emotion detection** from voice recordings
- **Six emotion classes**: Angry, Happy, Sad, Fear, Neutral, Surprise
- **LSTM-based deep learning model** with 128→64 architecture
- **MFCC feature extraction** (40 coefficients)
- **Data augmentation** for improved model robustness
- **Multi-dataset support**: RAVDESS and CREMA-D
- **Simple CLI interface** for recording and prediction

## 📊 Model Performance

- **Architecture**: Bidirectional LSTM with Dropout & Batch Normalization
- **Training Data**: ~7,000+ audio samples
- **Feature Extraction**: 40 MFCC coefficients
- **Sequence Length**: 174 time steps (padded/truncated)
- **Optimizer**: Adam (learning rate: 0.0003)
- **Regularization**: L2 (0.001) + Dropout (0.3-0.4)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- macOS, Linux, or Windows
- Microphone for recording

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/voice-emotion-recognition.git
cd voice-emotion-recognition

# 2. Create a virtual environment
python3 -m venv venv

# 3. Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the emotion recognition system
python app.py

# Press Enter when ready to record
# Speak for 3 seconds
# View the predicted emotion
```

## 📁 Project Structure

```
voice-emotion-recognition/
├── app.py                      # Main application (record & predict)
├── train.py                    # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
│
├── data/                       # Organized audio dataset
│   ├── angry/
│   ├── happy/
│   ├── sad/
│   ├── fear/
│   ├── neutral/
│   └── surprise/
│
├── models/                     # Saved model files
│   └── lstm_emotion_model.keras
│
├── utils/                      # Utility functions
│   ├── audio_preprocessing.py  # MFCC extraction & augmentation
│   └── dataset_loader.py       # Dataset loading & preprocessing
│
└── organize_scripts/           # Dataset organization scripts
    ├── organize_ravdess.py
    └── organize_crema_d.py
```

## 🎓 Dataset Setup

This project uses two popular speech emotion datasets:

### 1. RAVDESS Dataset

**Download**: [Ryerson Audio-Visual Database of Emotional Speech and Song](https://zenodo.org/record/1188976)

#### Understanding RAVDESS Filenames

Example: `03-01-05-01-02-01-12.wav`

Position breakdown:
- Position 1: Modality (03 = audio-only)
- Position 2: Vocal channel (01 = speech)
- **Position 3: Emotion** ⭐
- Position 4: Intensity (01/02)
- Position 5: Statement (01/02)
- Position 6: Repetition (01/02)
- Position 7: Actor ID (01-24)

#### Emotion Codes (Position 3)

| Code | Emotion  | Used in Project |
|------|----------|-----------------|
| 01   | Neutral  | ✅ Yes          |
| 02   | Calm     | ❌ Skipped      |
| 03   | Happy    | ✅ Yes          |
| 04   | Sad      | ✅ Yes          |
| 05   | Angry    | ✅ Yes          |
| 06   | Fear     | ✅ Yes          |
| 07   | Disgust  | ❌ Skipped      |
| 08   | Surprise | ✅ Yes          |

#### Organizing RAVDESS

```bash
# 1. Download and extract RAVDESS dataset
# 2. Place in project folder as 'Audio_Speech_Actors_01-24'
# 3. Run organization script
python organize_ravdess.py
```

### 2. CREMA-D Dataset

**Download**: [Crowd-sourced Emotional Multimodal Actors Dataset](https://github.com/CheyneyComputerScience/CREMA-D)

#### Understanding CREMA-D Filenames

Example: `1001_DFA_ANG_XX.wav`

Parts breakdown:
- Part 1: Actor ID (1001-1091)
- Part 2: Sentence ID (DFA, IEO, etc.)
- **Part 3: Emotion code** ⭐
- Part 4: Intensity level (XX, LO, MD, HI)

#### Emotion Codes (Part 3)

| Code | Emotion  | Used in Project |
|------|----------|-----------------|
| NEU  | Neutral  | ✅ Yes          |
| HAP  | Happy    | ✅ Yes          |
| SAD  | Sad      | ✅ Yes          |
| ANG  | Angry    | ✅ Yes          |
| FEA  | Fear     | ✅ Yes          |
| SU   | Surprise | ✅ Yes          |
| DIS  | Disgust  | ❌ Skipped      |

#### Organizing CREMA-D

```bash
# 1. Download and extract CREMA-D dataset
# 2. Place audio files in 'AudioWAV' folder
# 3. Run organization script
python organize_crema_d.py
```

### Final Data Structure

After running both organization scripts, your `data/` folder should look like:

```
data/
├── angry/       # ~1,200 files
├── happy/       # ~1,200 files
├── sad/         # ~1,200 files
├── fear/        # ~1,200 files
├── neutral/     # ~1,200 files
└── surprise/    # ~1,200 files
```

## 🏋️ Training the Model

### Step 1: Prepare the Dataset

Ensure you've organized both RAVDESS and CREMA-D datasets as described above.

### Step 2: Train the Model

```bash
python train.py
```

The training process includes:
1. **Loading dataset** from `data/` folder
2. **Data augmentation** (Gaussian noise)
3. **Train/test split** (80/20)
4. **Model compilation** with Adam optimizer
5. **Training** with early stopping (patience: 15 epochs)
6. **Model saving** to `models/lstm_emotion_model.keras`

### Training Parameters

```python
# Adjustable in train.py
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
TEST_SPLIT = 0.2
EARLY_STOP_PATIENCE = 15
```

### Expected Output

```
📥 Loading dataset...
✂️ Splitting dataset...
🏗️ Building LSTM model...
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
lstm (LSTM)                 (None, 174, 128)          86528     
dropout (Dropout)           (None, 174, 128)          0         
lstm_1 (LSTM)               (None, 64)                49408     
dropout_1 (Dropout)         (None, 64)                0         
batch_normalization         (None, 64)                256       
dense (Dense)               (None, 64)                4160      
dropout_2 (Dropout)         (None, 64)                0         
dense_1 (Dense)             (None, 6)                 390       
=================================================================
Total params: 140,742

🏃 Training model...
Epoch 1/50
180/180 [==============================] - 15s 75ms/step
...
💾 Saving model...
✅ Training complete!
```

## 🎙️ Using the Application

### Recording & Prediction

```bash
python app.py
```

The application will:
1. Load the trained model
2. Wait for you to press Enter
3. Record 3 seconds of audio
4. Save as `record.wav`
5. Extract MFCC features
6. Predict emotion
7. Display result

### Example Session

```
Press Enter to record...
🎤 Recording...
Saved as record.wav
Predicted Emotion: happy
----------------------------------------
Press Enter to record...
```

## 🔧 Technical Details

### Audio Preprocessing

```python
# Feature extraction parameters
SAMPLE_RATE = 22050      # Hz
N_MFCC = 40              # MFCC coefficients
MAX_LENGTH = 174         # Time steps (frames)
RECORDING_DURATION = 3   # seconds
```

### MFCC Feature Extraction

Mel-Frequency Cepstral Coefficients (MFCCs) are used to represent the audio signal:

1. **Load audio** at 22,050 Hz
2. **Extract 40 MFCC coefficients**
3. **Transpose** to shape (time_steps, n_mfcc)
4. **Pad or truncate** to 174 frames
5. **Output shape**: (174, 40)

### Data Augmentation

To improve model generalization:

```python
def augment_mfcc(features):
    """Add small Gaussian noise"""
    noise = np.random.normal(0, 0.01, features.shape)
    return features + noise
```

### Model Architecture

```python
Sequential([
    LSTM(128, return_sequences=True)  # First LSTM layer
    Dropout(0.4)                      # Regularization
    LSTM(64)                          # Second LSTM layer
    Dropout(0.4)                      # Regularization
    BatchNormalization()              # Normalize activations
    Dense(64, activation='relu', l2=0.001)
    Dropout(0.3)                      # Final dropout
    Dense(6, activation='softmax')    # Output layer
])
```

## 📚 Dependencies

### Core Libraries

- **tensorflow**: Deep learning framework
- **librosa**: Audio processing and feature extraction
- **numpy**: Numerical computing
- **scikit-learn**: ML utilities (train_test_split, etc.)
- **sounddevice**: Audio recording
- **scipy**: Scientific computing (audio I/O)
- **pandas**: Data manipulation
- **matplotlib**: Visualization (optional)

### Installation

```bash
pip install tensorflow librosa numpy scikit-learn sounddevice scipy pandas matplotlib
```

Or use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Microphone Not Working

**macOS**: Grant microphone permissions in System Preferences → Security & Privacy

**Linux**: Install PortAudio
```bash
sudo apt-get install portaudio19-dev
```

**Windows**: Ensure correct audio input device is selected

#### 2. Model File Not Found

```bash
# Train the model first
python train.py
```

#### 3. ImportError: No module named 'librosa'

```bash
# Reinstall dependencies
pip install -r requirements.txt
```

#### 4. TensorFlow GPU Issues

For CPU-only installation:
```bash
pip install tensorflow-cpu
```

#### 5. Dataset Not Organized

Ensure you've run:
```bash
python organize_ravdess.py
python organize_crema_d.py
```

## 🎨 Customization

### Adding New Emotions

1. Add emotion to label mapping in `utils/dataset_loader.py`
2. Update model output layer to new class count
3. Retrain the model

### Adjusting Recording Duration

```python
# In app.py
def record_audio(filename="record.wav", duration=5, fs=22050):
    # Change duration parameter
```

### Changing Model Architecture

Edit `train.py` to modify layers, units, or hyperparameters:

```python
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(256, return_sequences=True),  # Increase units
    # ... add more layers
])
```

## 📈 Performance Optimization

### Tips for Better Accuracy

1. **More training data**: Add additional datasets
2. **Longer training**: Increase epochs if not overfitting
3. **Hyperparameter tuning**: Adjust learning rate, batch size
4. **Feature engineering**: Add more audio features (spectral features, pitch, etc.)
5. **Ensemble models**: Combine multiple models
6. **Cross-validation**: Use k-fold cross-validation

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **RAVDESS Dataset**: Livingstone SR, Russo FA (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
- **CREMA-D Dataset**: Cao H, Cooper DG, Keutmann MK, Gur RC, Nenkova A, Verma R (2014)
- **TensorFlow/Keras**: Deep learning framework
- **Librosa**: Audio processing library


## 🔮 Future Enhancements

- [ ] Web interface with real-time visualization
- [ ] Mobile app (iOS/Android)
- [ ] Multi-language support
- [ ] Speaker identification
- [ ] Emotion intensity detection
- [ ] Real-time streaming support
- [ ] REST API for integration
- [ ] Docker containerization

⭐ Star this repo if you find it helpful!
