import streamlit as st
import tempfile
import torch
import whisperx
import os
import time
from pyannote.audio import Pipeline

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Speech Diarization & Transcription",
    page_icon="🎙️",
    layout="wide"
)

# ------------------- TITLE -------------------
st.title("🎧 Speech Diarization & Transcription Dashboard")
st.markdown("Upload an audio file below and get **speaker-separated transcription** using WhisperX + Pyannote.")

# ------------------- UPLOAD SECTION -------------------
uploaded_file = st.file_uploader("Upload your audio file (mp3, wav, m4a, etc.)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name

    st.success(f"✅ File uploaded successfully: `{uploaded_file.name}`")

    # ------------------- MODEL SETUP -------------------
    st.markdown("### ⚙️ Model Loading in Progress...")
    start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    model = whisperx.load_model("small", device, compute_type=compute_type)
    st.write("🔹 WhisperX model loaded.")

    # ------------------- TRANSCRIPTION -------------------
    st.markdown("### 📝 Transcribing Audio...")
    transcribe_start = time.time()
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio)
    st.write("✅ Transcription complete in {:.2f} sec".format(time.time() - transcribe_start))

    # ------------------- ALIGNMENT -------------------
    st.markdown("### 🎯 Aligning Words...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)
    st.write("✅ Alignment complete.")

    # ------------------- DIARIZATION -------------------
    st.markdown("### 🧠 Speaker Diarization...")
    try:
        diarize_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=None)
        diarize_segments = diarize_pipeline(audio_path)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        st.write("✅ Speaker diarization complete.")
    except Exception as e:
        st.warning("⚠️ Diarization failed. Proceeding without speaker labels.")
        st.error(str(e))

    end_time = time.time()
    st.info(f"Total processing time: {end_time - start_time:.2f} seconds")

    # ------------------- DISPLAY OUTPUT -------------------
    st.markdown("### 🎙️ Final Speaker-Labelled Transcript")

    if "segments" in result and len(result["segments"]) > 0:
        for seg in result["segments"]:
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", "")
            st.markdown(f"**{speaker}:** {text}")
    else:
        st.error("No transcription segments found.")

    # ------------------- DOWNLOAD SECTION -------------------
    st.markdown("### 💾 Download Transcription")
    transcript_text = "\n".join(
        [f"{seg.get('speaker', 'Unknown')}: {seg.get('text', '')}" for seg in result.get("segments", [])]
    )

    transcript_file = "transcript_with_speakers.txt"
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    with open(transcript_file, "rb") as f:
        st.download_button(
            label="⬇️ Download Transcript",
            data=f,
            file_name=transcript_file,
            mime="text/plain"
        )

else:
    st.info("Please upload an audio file to begin.")
