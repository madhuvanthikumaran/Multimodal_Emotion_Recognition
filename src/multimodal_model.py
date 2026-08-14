from tensorflow.keras.models import load_model

image_model = load_model("models/image_emotion_model.h5")
audio_model = load_model("models/audio_emotion_model.h5")

print("Image model loaded")
print("Audio model loaded")

print("Multimodal model ready")