import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load data
X = np.load("models/audio_features.npy")
y = np.load("models/audio_labels.npy")

print("Features Shape:", X.shape)
print("Labels Shape:", y.shape)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

joblib.dump(encoder, "models/audio_label_encoder.pkl")

y_encoded = to_categorical(y_encoded)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, "models/audio_scaler.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Model
model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.4),

    Dense(256, activation='relu'),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(64, activation='relu'),

    Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16
)

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

model.save("models/audio_emotion_model.h5")

print("Audio model saved successfully!")