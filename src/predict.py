import cv2
import numpy as np
import librosa
import joblib

from tensorflow.keras.models import load_model

# =========================
# LOAD MODELS
# =========================

image_model = load_model("models/image_emotion_model.h5")
audio_model = load_model("models/audio_emotion_model.h5")

scaler = joblib.load("models/audio_scaler.pkl")
encoder = joblib.load("models/audio_label_encoder.pkl")

print("Models loaded successfully!")

# =========================
# IMAGE PREDICTION
# =========================

image_path = "test_inputs/test.jpg"

img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Cannot find image: {image_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (48, 48))

gray = gray.astype("float32") / 255.0
gray = np.expand_dims(gray, axis=-1)
gray = np.expand_dims(gray, axis=0)

image_prediction = image_model.predict(gray)

image_emotions = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

image_emotion = image_emotions[np.argmax(image_prediction)]

print("\nImage Emotion:", image_emotion)

# =========================
# AUDIO PREDICTION
# =========================

audio_path = "test_inputs/test.wav"

signal, sr = librosa.load(audio_path, sr=22050)

mfcc = librosa.feature.mfcc(
    y=signal,
    sr=sr,
    n_mfcc=40
)

mfcc = np.mean(mfcc.T, axis=0)

audio_features = np.array([mfcc])

audio_features = scaler.transform(audio_features)

audio_prediction = audio_model.predict(audio_features)

emotion_index = np.argmax(audio_prediction)

audio_emotion = encoder.inverse_transform([emotion_index])[0]

print("Audio Emotion:", audio_emotion)

# =========================
# FINAL DECISION
# =========================

if image_emotion == audio_emotion:
    final_emotion = image_emotion
else:
    image_confidence = np.max(image_prediction)
    audio_confidence = np.max(audio_prediction)

    if image_confidence > audio_confidence:
        final_emotion = image_emotion
    else:
        final_emotion = audio_emotion

print("\nFinal Emotion:", final_emotion)