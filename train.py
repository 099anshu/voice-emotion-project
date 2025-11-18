#!/usr/bin/env python3
"""
Improved Voice Emotion Recognition Training Script
==================================================
Features:
- Advanced CNN-LSTM with MultiHead Attention
- Comprehensive audio features (MFCC + spectral + temporal)
- Data augmentation with noise, time shifting, and masking
- Proper feature normalization using global scaler
- Class balancing and learning rate scheduling
- Comprehensive evaluation with confusion matrices
"""

from utils.dataset_loader import load_dataset
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, BatchNormalization,
                                     Dropout, LSTM, Bidirectional, Dense,
                                     GlobalAveragePooling1D, MultiHeadAttention,
                                     LayerNormalization, Add)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules

# -------------------------
# CONFIGURATION
# -------------------------
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 15
RANDOM_STATE = 42

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("🚀 Starting Improved Voice Emotion Recognition Training")
print("=" * 60)

# -------------------------
# DATA AUGMENTATION FUNCTIONS
# -------------------------


def add_noise(features, noise_factor=0.005):
    """Add Gaussian noise to features"""
    noise = np.random.normal(0, noise_factor, features.shape)
    return features + noise


def time_shift(features, shift_max=10):
    """Randomly shift features in time"""
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(features, shift, axis=0)


def frequency_mask(features, freq_mask_param=15):
    """Mask random frequency channels"""
    masked = features.copy()
    num_mel_channels = features.shape[1]
    f = np.random.randint(0, freq_mask_param)
    f0 = np.random.randint(0, num_mel_channels - f)
    masked[:, f0:f0+f] = 0
    return masked


def time_mask(features, time_mask_param=20):
    """Mask random time steps"""
    masked = features.copy()
    len_spectro = features.shape[0]
    t = np.random.randint(0, time_mask_param)
    t0 = np.random.randint(0, len_spectro - t)
    masked[t0:t0+t, :] = 0
    return masked


def data_augmentation(X, y, augmentation_factor=2):
    """
    Apply comprehensive data augmentation
    """
    print(f"Applying data augmentation (factor: {augmentation_factor})...")

    augmented_X = []
    augmented_y = []

    for i in range(len(X)):
        # Original sample
        augmented_X.append(X[i])
        augmented_y.append(y[i])

        # Generate augmented samples
        for _ in range(augmentation_factor - 1):
            aug_sample = X[i].copy()

            # Apply random augmentations
            if np.random.random() > 0.5:
                aug_sample = add_noise(aug_sample)
            if np.random.random() > 0.5:
                aug_sample = time_shift(aug_sample)
            if np.random.random() > 0.7:
                aug_sample = frequency_mask(aug_sample)
            if np.random.random() > 0.7:
                aug_sample = time_mask(aug_sample)

            augmented_X.append(aug_sample)
            augmented_y.append(y[i])

    return np.array(augmented_X), np.array(augmented_y)

# -------------------------
# MODEL ARCHITECTURE
# -------------------------


def build_advanced_model(input_shape, num_classes):
    """
    Build advanced CNN-LSTM model with MultiHead Attention
    """
    print("🏗️ Building improved CNN-LSTM model with attention...")

    inputs = Input(shape=input_shape)

    # CNN blocks for feature extraction
    x = Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(64, 3, padding='same', activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.25)(x)

    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.25)(x)

    x = Conv1D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.25)(x)

    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    # Multi-head attention mechanism
    attention_output = MultiHeadAttention(
        num_heads=4, key_dim=32
    )(x, x)

    # Residual connection + normalization
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)

    # Global pooling and dense layers
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


# -------------------------
# LOAD AND PREPARE DATA
# -------------------------
print("📥 Loading dataset...")
X, y, emotion_labels = load_dataset("data")

print(f"Original dataset shape: X={X.shape}, y={y.shape}")
print(f"Feature dimensions: {X.shape[2]} features per time step")
print(f"Detected emotions: {emotion_labels}")

# Check number of classes
num_classes = y.shape[1]
emotion_counts = np.bincount(np.argmax(y, axis=1))
print(f"Detected {num_classes} emotions:")
for i, (emotion, count) in enumerate(zip(emotion_labels, emotion_counts)):
    print(f"  {emotion}: {count} samples")

# Apply data augmentation
print("🔄 Applying data augmentation...")
X, y = data_augmentation(X, y, augmentation_factor=2)
print(f"Augmented dataset shape: X={X.shape}, y={y.shape}")

# Compute class weights for imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(
    np.argmax(y, axis=1)), y=np.argmax(y, axis=1))
class_weights_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weights_dict}")

# -------------------------
# TRAIN/VALIDATION/TEST SPLIT
# -------------------------
print("✂️ Splitting dataset...")

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=np.argmax(y, axis=1)
)

# Second split: training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE,
    stratify=np.argmax(y_temp, axis=1)
)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")

# -------------------------
# BUILD AND COMPILE MODEL
# -------------------------
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_advanced_model(input_shape, num_classes)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

model.summary()

# -------------------------
# CALLBACKS
# -------------------------
os.makedirs("models", exist_ok=True)

callbacks = [
    ModelCheckpoint(
        "models/best_emotion_model.keras",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    )
]

# -------------------------
# TRAIN MODEL
# -------------------------
print("🏃 Training model...")
print("Starting training...")

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    class_weight=class_weights_dict,
    verbose=1
)

# -------------------------
# EVALUATE MODEL
# -------------------------
print("🔍 Evaluating model...")

# Load best model
print("✅ Loaded best model from checkpoint")
best_model = tf.keras.models.load_model("models/best_emotion_model.keras")

# Evaluate on test set
test_loss, test_accuracy, test_precision, test_recall = best_model.evaluate(
    X_test, y_test, verbose=0
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Detailed classification report
y_pred = best_model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_test_classes, y_pred_classes,
                            target_names=emotion_labels))

# Confusion Matrix
print("\n📈 Confusion Matrix:")
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Confusion matrix saved to models/confusion_matrix.png")

# -------------------------
# PLOT TRAINING HISTORY
# -------------------------
print("\n📉 Plotting training history...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0, 0].set_title('Model Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Loss
axes[0, 1].plot(history.history['loss'], label='Training Loss')
axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
axes[0, 1].set_title('Model Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Precision
axes[1, 0].plot(history.history['precision'], label='Training Precision')
axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
axes[1, 0].set_title('Model Precision')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Recall
axes[1, 1].plot(history.history['recall'], label='Training Recall')
axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
axes[1, 1].set_title('Model Recall')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
plt.close()
print("Training history saved to models/training_history.png")

# -------------------------
# SAVE FINAL MODEL
# -------------------------
print("\n💾 Saving final model...")
best_model.save("models/improved_cnn_lstm_emotion_model.keras")

# -------------------------
# TEST INFERENCE
# -------------------------
print("\n🎯 Testing inference...")
try:
    # Test with a sample from the dataset
    test_sample = X_test[0:1]  # Take first test sample
    prediction = best_model.predict(test_sample, verbose=0)
    predicted_emotion = emotion_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    print(
        f"Predicted emotion: {predicted_emotion} (confidence: {confidence:.3f})")
    print("All confidence scores:")
    for i, score in enumerate(prediction[0]):
        print(f"  {emotion_labels[i]}: {score:.3f}")

    print("✅ Successfully predicted emotion for test sample")

except Exception as e:
    print(f"❌ Error during inference test: {e}")

print("\n✅ Training and evaluation complete!")
print(f"📈 Final test accuracy: {test_accuracy:.1%}")
print("🎯 Model saved and ready for use!")
