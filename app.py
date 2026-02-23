import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------- Load model & tokenizer (cached for speed) ----------
@st.cache_resource
def load_resources():
    model = load_model("emotion_cnn_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()

# ---------- Settings ----------
MAX_LEN = 100  # same value used during training
emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']

# ---------- Prediction function ----------
def predict_emotion(text):
    if not text.strip():
        return None
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = model.predict(padded)
    return emotion_labels[np.argmax(prediction)]

# ---------- UI ----------
st.title("Emotion Classification using CNN")

user_input = st.text_area("Enter text:")

if st.button("Classify Emotion"):
    result = predict_emotion(user_input)

    if result is None:
        st.warning("Please enter text")
    else:
        st.success(f"Predicted Emotion: {result}")
