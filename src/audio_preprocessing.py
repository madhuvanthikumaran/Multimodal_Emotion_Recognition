import os
import numpy as np
import librosa

DATASET_PATH = "datasets/RAVDESS"

X = []
y = []

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

for actor in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if file.endswith(".wav"):

            file_path = os.path.join(actor_path, file)

            # Load audio
            signal, sr = librosa.load(file_path, sr=22050)

            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=signal,
                sr=sr,
                n_mfcc=40
            )

            # Take mean across time axis
            mfcc = np.mean(mfcc.T, axis=0)

            X.append(mfcc)

            # Extract emotion code from filename
            emotion_code = file.split("-")[2]

            y.append(emotion_map[emotion_code])

X = np.array(X)
y = np.array(y)

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

np.save("models/audio_features.npy", X)
np.save("models/audio_labels.npy", y)

print("Audio preprocessing successful!")