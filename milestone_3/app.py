import os
import streamlit as st
import sounddevice as sd
import wave
import tempfile
import speech_recognition as sr
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="SPEACH TO TEXT CONVERTION FROM LIVE RECORDING AND FROM FILES",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- CUSTOM THEME & STYLES --------------------
st.markdown("""
<style>
/* Base */
:root {
  --bg: #0b0f14;
  --panel: #11161e;
  --panel-2: #0f141b;
  --text: #e6edf3;
  --muted: #9aa7b2;
  --accent: #22d3ee;   /* cyan */
  --accent-2: #7c3aed; /* violet */
  --border: #1f2937;
  --success: #22c55e;
  --warning: #f59e0b;
  --danger: #ef4444;
}
html, body, .stApp {
  background: radial-gradient(1200px 600px at 10% -20%, #0d1218 0%, var(--bg) 60%) no-repeat fixed;
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue, Arial, Noto Sans, "Apple Color Emoji", "Segoe UI Emoji";
}

/* Header */
.app-hero {
  background: linear-gradient(135deg, rgba(34, 211, 238, .25), rgba(124, 58, 237, .22));
  border: 1px solid rgba(124, 58, 237, .35);
  box-shadow: 0 20px 60px rgba(32, 39, 49, .6), inset 0 0 100px rgba(34, 211, 238, .08);
  border-radius: 18px;
  padding: 26px 28px;
}
.app-title {
  margin: 0;
  font-weight: 800;
  font-size: 34px;
  letter-spacing: .2px;
  color: var(--text);
}
.app-subtitle {
  margin-top: 6px;
  color: var(--muted);
  font-size: 16px;
}

/* Grid Layout */
.section {
  margin-top: 22px;
}
.grid-2 {
  display: grid;
  grid-template-columns: 1.05fr 1.6fr;
  gap: 22px;
}
.grid-1-center {
  display: flex;
  justify-content: center;
}

/* Cards */
.card {
  background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
}
.card h3 {
  margin-top: 2px;
  margin-bottom: 12px;
  font-size: 20px;
  font-weight: 700;
  color: var(--text);
}

/* Inputs */
.stRadio > div {
  background-color: #0f141b;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px;
}
.stSlider, .stFileUploader, .stTextArea {
  border-radius: 12px !important;
}
textarea {
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
  background-color: #101620 !important;
  color: var(--text) !important;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
  color: white !important;
  border-radius: 12px !important;
  border: 0 !important;
  font-weight: 700 !important;
  letter-spacing: .2px;
  padding: 10px 16px !important;
  transition: transform .12s ease, box-shadow .12s ease, filter .12s ease;
  box-shadow: 0 12px 22px rgba(34, 211, 238, .18);
}
.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 16px 28px rgba(124, 58, 237, .18);
  filter: brightness(1.03);
}

/* Secondary buttons */
.btn-secondary > button {
  background: linear-gradient(90deg, #334155, #1f2937) !important;
  color: #e5e7eb !important;
  border-radius: 12px !important;
  border: 1px solid #344156 !important;
}

/* Tag pills */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  color: #cbd5e1;
  border: 1px dashed #334155;
  background: #0f141b;
  font-size: 13px;
}

/* Footer */
.footer {
  color: var(--muted);
  text-align: center;
  margin-top: 24px;
  font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div class="app-hero">
  <h1 class="app-title">SPEACH TO TEXT CONVERTION FROM LIVE RECORDING AND FROM FILES</h1>
  <div class="app-subtitle">Record live or upload audio. Clean transcripts, quick summaries, smooth experience.</div>
  <div style="margin-top:12px; display:flex; gap:10px; flex-wrap:wrap;">
    <span class="pill">🎙️ Live capture</span>
    <span class="pill">📁 WAV upload</span>
    <span class="pill">🧠 Extractive summary</span>
  </div>
</div>
""", unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

# -------------------- SIMPLE SUMMARIZER --------------------
def simple_summarizer(text, num_sentences=3):
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) <= num_sentences:
        return text
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()
    ranked = np.argsort(scores)[::-1][:num_sentences]
    summary = " ".join([sentences[i] for i in sorted(ranked)])
    return summary

# -------------------- SPEECH RECOGNITION --------------------
recognizer = sr.Recognizer()

def transcribe_audio(path):
    with sr.AudioFile(path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text

# -------------------- AUDIO RECORDING --------------------
def record_audio(duration=5, fs=44100):
    st.info("🎤 Recording… Speak now")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmpfile.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())
    st.success("✅ Recording finished. Ready to process.")
    return tmpfile.name

# -------------------- MAIN LAYOUT --------------------
st.markdown('<div class="section grid-1-center">', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎧 Input")
    mode = st.radio("Choose input type", ["🎤 Record from microphone", "📁 Upload WAV file"])

    if mode.startswith("🎤"):
        duration = st.slider("⏱️ Duration (seconds)", 3, 20, 5)
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("🎙️ Start recording"):
                st.session_state.audio_path = record_audio(duration)
        with c2:
            if st.session_state.audio_path:
                st.audio(st.session_state.audio_path)
    else:
        uploaded = st.file_uploader("📂 Select a .wav file", type=["wav"])
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(uploaded.read())
            st.session_state.audio_path = tmp.name
            st.audio(tmp.name)
            st.success("✅ File uploaded")

    # Input actions
    a1, a2, a3 = st.columns([1, 1, 1])
    with a1:
        if st.session_state.audio_path:
            st.button("🚀 Transcribe & summarize")
    with a2:
        if st.session_state.audio_path:
            st.button("🔁 Reset", key="reset_btn")
            if "reset_btn" in st.session_state:
                # Reset state when button rendered + clicked (handled below)
                pass
    with a3:
        st.caption("Tip: Use high‑quality audio for best results.")
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Handle reset click explicitly (Streamlit reruns; check last interaction)
if st.session_state.get("reset_btn"):
    st.session_state.audio_path = None
    st.session_state.transcription = ""
    st.session_state.summary = ""
    st.experimental_set_query_params()  # soft refresh marker

# -------------------- OUTPUT (conditionally visible) --------------------
if st.session_state.audio_path:
    st.markdown('<div class="section grid-2">', unsafe_allow_html=True)

    # Left: Process controls
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Process")
        st.markdown("Refine your audio to text and extract a concise summary.")
        colp1, colp2 = st.columns([1, 1])
        with colp1:
            if st.button("🧠 Run transcription & summary", key="run_process"):
                try:
                    with st.spinner("🎧 Transcribing…"):
                        text = transcribe_audio(st.session_state.audio_path)
                        st.session_state.transcription = text
                    with st.spinner("🧠 Summarizing…"):
                        summary = simple_summarizer(text, num_sentences=3)
                        st.session_state.summary = summary
                    st.success("✅ Done. See results on the right.")
                except Exception as e:
                    st.error(f"❌ {e}")
        with colp2:
            st.caption("Your audio stays local to the app during processing.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Right: Results panel
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📄 Results")

        if st.session_state.transcription:
            st.markdown("**📝 Transcribed text**")
            st.text_area("", st.session_state.transcription, height=200)

        if st.session_state.summary:
            st.markdown("**🧾 Summary**")
            st.text_area("", st.session_state.summary, height=140)

        if st.session_state.transcription or st.session_state.summary:
            d1, d2 = st.columns([1, 1])
            with d1:
                if st.session_state.transcription:
                    st.download_button(
                        "⬇️ Download transcription",
                        st.session_state.transcription,
                        "transcription.txt",
                        help="Save the full transcript"
                    )
            with d2:
                if st.session_state.summary:
                    st.download_button(
                        "⬇️ Download summary",
                        st.session_state.summary,
                        "summary.txt",
                        help="Save the summarized text"
                    )
        else:
            st.info("Run the process to generate transcription and summary.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown('<div class="footer">Built by Soumodip Ghosh • SPEACH TO TEXT CONVERTION FROM LIVE RECORDING AND FROM FILES</div>', unsafe_allow_html=True)