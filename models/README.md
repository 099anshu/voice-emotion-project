# Models Directory

This directory contains the trained emotion recognition models and related files.

## Files

### Available in Repository
- `feature_scaler.pkl` - StandardScaler for feature normalization (1.9KB)
- `confusion_matrix.png` - Model performance visualization (131KB)
- `training_history.png` - Training metrics over epochs (516KB)

### Download Required (Large Files)
Due to GitHub file size limits, the following model files need to be downloaded separately:

- `best_emotion_model.keras` (10MB) - **Main trained model**
- `emotion_model.h5` (1.6MB) - Alternative model format
- `lstm_emotion_model.keras` (1.7MB) - LSTM variant

## Download Instructions

### Option 1: Google Drive
Download the models from: [Google Drive Link](https://drive.google.com/your-link)

### Option 2: Hugging Face Hub
Download from: [Hugging Face Model](https://huggingface.co/your-username/voice-emotion-model)

### Option 3: Train Your Own
```bash
python train.py
```
This will generate `best_emotion_model.keras` in the models directory.

## Model Performance

- **Test Accuracy**: 84.4%
- **Test Precision**: 84.9%
- **Test Recall**: 83.9%

## Usage

The main model (`best_emotion_model.keras`) is automatically loaded by `app.py`. Ensure it's in this directory before running the application.
