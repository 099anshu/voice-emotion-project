from utils.dataset_loader import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import os

print("Loading dataset...")
X, y = load_dataset("data")

print("Normalizing...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Splitting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Building model...")
model = Sequential([
    Dense(128, activation="relu", input_shape=(40,)),
    Dense(64, activation="relu"),
    Dense(6, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Training...")
model.fit(X_train, y_train, epochs=25)

print("Saving...")
os.makedirs("models", exist_ok=True)
model.save("models/emotion_model.h5")

print("Done!")
