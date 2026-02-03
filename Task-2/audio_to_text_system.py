import streamlit as st
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import tempfile

st.set_page_config(page_title="Speech to Text", layout="centered")

st.title("🎙️ Speech-to-Text System")
st.write("Using Pre-trained **Wav2Vec 2.0** Model")

@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    return processor, model

processor, model = load_model()

def speech_to_text(audio_file):
    speech, rate = librosa.load(audio_file, sr=16000)
    input_values = processor(
        speech,
        return_tensors="pt",
        sampling_rate=16000
    ).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription

uploaded_file = st.file_uploader(
    "Upload a WAV audio file",
    type=["wav"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_audio_path = tmp.name

    if st.button("🔍 Convert Speech to Text"):
        with st.spinner("Transcribing..."):
            text = speech_to_text(temp_audio_path)
        st.success("Transcription Complete!")
        st.text_area("📝 Transcribed Text", text, height=150)
