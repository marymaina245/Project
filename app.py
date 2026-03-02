# 1️⃣ imports
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# 2️⃣ load model + tokenizer (top of file)
model = load_model("emotion_cnn_model.keras")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

max_len = 100   # same value used in training


# 3️⃣ prediction function (IMPORTANT — define before UI)
def predict(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len)

    probs = model.predict(pad)[0]
    idx = probs.argmax()

    confidence = float(probs[idx])

    return emotion, confidence


# 4️⃣ 👉 YOUR UI CODE GOES HERE (the code you pasted)
st.title("Patient Emotion Monitoring System")
st.caption("Detect emotional state from patient or clinical text")

text = st.text_area("Enter patient statement / nurse note")

if st.button("Analyze Emotional State"):

    emotion, confidence = predict(text)

    st.subheader("Detected Emotion")
    st.success(emotion)

    st.write(f"Confidence: {confidence:.2f}")

    clinical_map = {
        "fear": "Anxiety / Pre-procedure fear",
        "sadness": "Emotional distress",
        "anger": "Frustration or pain distress",
        "joy": "Positive recovery indicator",
        "love": "Gratitude / positive staff interaction",
        "surprise": "Unexpected reaction — review context"
    }

    st.subheader("Clinical Interpretation")
    st.info(clinical_map.get(emotion))

    if emotion in ["fear", "sadness", "anger"] and confidence > 0.75:
        st.error("⚠️ Possible emotional distress — consider follow-up")

st.warning("This AI supports clinicians and does not replace professional judgement.")





