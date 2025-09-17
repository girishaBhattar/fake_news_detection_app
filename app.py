import streamlit as st
import pickle
import numpy as np

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article below or upload a file to check if it's **Fake** or **Real**.")

# Input
user_input = st.text_area("📝 Enter News Article Text:", height=200)

if st.button("🔍 Check"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        confidence = round(np.max(model.predict_proba(input_vector)) * 100, 2)

        if prediction == 1:
            st.success("✅ This news article is likely **REAL**.")
        else:
            st.error("❌ This news article is likely **FAKE**.")

        st.markdown(f"### 📊 Confidence Score: {confidence}%")
        st.progress(int(confidence))
