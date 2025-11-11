# ---------------------------------------------------------
# 🎙 Accent-Aware Cuisine Recommendation System (Streamlit)
# ---------------------------------------------------------

import streamlit as st
import numpy as np
import librosa
import torch
import joblib
import os
from transformers import AutoFeatureExtractor, HubertModel

# ---------------------------------------------------------
# Load Model and Assets
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    base_path = os.path.join(os.getcwd(), "models")

    model = joblib.load(os.path.join(base_path, "hubert_model.pkl"))
    states = joblib.load(os.path.join(base_path, "hubert_states.pkl"))  # list of folder names

    extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    hubert.eval()

    return model, states, extractor, hubert


model, states, extractor, hubert = load_assets()

# ---------------------------------------------------------
# Accent → Cuisine Mapping
# ---------------------------------------------------------
accent_to_cuisine = {
    "Andhra Pradesh": "Pulihora, Pesarattu",
    "Gujarat": "Dhokla, Thepla",
    "Jharkhand": "Litti Chokha",
    "Karnataka": "Bisi Bele Bath, Neer Dosa",
    "Kerala": "Appam, Avial, Puttu",
    "Tamil Nadu": "Pongal, Idli, Dosa",
}

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Accent-Aware Cuisine", page_icon="🎧")
st.title("🎙 Accent-Aware Cuisine Recommendation")
st.write("Upload an English speech sample. I’ll detect the accent and suggest cuisines from that region 🍛")

uploaded = st.file_uploader("📂 Upload .wav or .mp3 file", type=["wav", "mp3"])

if uploaded is not None:
    st.audio(uploaded, format="audio/wav")
    with st.spinner("🎧 Extracting HuBERT features and predicting..."):
        try:
            # Load and normalize audio
            y, sr = librosa.load(uploaded, sr=16000)
            y = librosa.util.normalize(y)

            # Limit to 5 seconds
            if len(y) > 16000 * 5:
                y = y[:16000 * 5]

            # Check for silent input
            if np.mean(np.abs(y)) < 0.01:
                st.warning("⚠️ Audio seems too quiet or silent. Please upload a clearer file.")
                st.stop()

            # HuBERT feature extraction
            inputs = extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = hubert(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

            # Predict accent
            pred = model.predict([emb])[0]
            raw_state = states[pred]  # e.g., 'kerala'
            accent = raw_state.replace("_", " ").title()  # 'Kerala'
            cuisine = accent_to_cuisine.get(accent, "Cuisine recommendation not available 🍽️")

            # Display results
            st.success(f"🗣️ Detected Accent: **{accent}**")
            st.info(f"🍲 Recommended Cuisine: **{cuisine}**")

            # Optional: show confidence
            if hasattr(model, "predict_proba"):
                conf = float(np.max(model.predict_proba([emb])) * 100)
                st.caption(f"Model Confidence: {conf:.2f}%")

        except Exception as e:
            st.error(f"❌ Error while processing: {e}")
