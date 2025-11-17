import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from utils.dataset_loader import load_dataset
from utils.audio_preprocessing import augment_mfcc

# -------------------------
# 1. LOAD DATASET
# -------------------------
print("📥 Loading dataset...")
X, y = load_dataset("data")  # (samples, time_steps, n_mfcc)

# Data augmentation
X_aug = np.array([augment_mfcc(x) for x in X])
X = np.concatenate([X, X_aug])
y = np.concatenate([y, y])

# -------------------------
# 2. TRAIN/TEST SPLIT
# -------------------------
print("✂️ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# -------------------------
# 3. RESHAPE FOR LSTM
# -------------------------
# LSTM expects (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# -------------------------
# 4. BUILD LSTM MODEL
# -------------------------
print("🏗️ Building LSTM model...")
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------
# 5. TRAIN
# -------------------------
print("🏃 Training model...")
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# -------------------------
# 6. SAVE MODEL
# -------------------------
print("💾 Saving model...")
os.makedirs("models", exist_ok=True)
model.save("models/lstm_emotion_model.keras")

print("✅ Training complete!")
