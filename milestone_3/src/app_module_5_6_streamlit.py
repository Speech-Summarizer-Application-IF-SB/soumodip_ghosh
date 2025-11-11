import streamlit as st
import tempfile
import torch
import whisperx
import os
import time

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Speech Transcription (WhisperX)",
    page_icon="🎙️",
    layout="wide"
)

# ------------------- TITLE -------------------
st.title("🎧 WhisperX Speech-to-Text Dashboard")
st.markdown("Upload an audio file below to get **transcription with word-level alignment** (Speaker diarization disabled for CPU mode).")

# ------------------- UPLOAD SECTION -------------------
uploaded_file = st.file_uploader("📁 Upload your audio file (mp3, wav, m4a, etc.)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name

    st.success(f"✅ File uploaded successfully: `{uploaded_file.name}`")

    # ------------------- MODEL SETUP -------------------
    st.markdown("### ⚙️ Loading WhisperX model...")
    start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float32"  # Force float32 for CPU compatibility

    model = whisperx.load_model("small", device=device, compute_type=compute_type)
    st.info(f"🔹 Model loaded on **{device.upper()}** (compute_type={compute_type}).")

    # ------------------- TRANSCRIPTION -------------------
    st.markdown("### 📝 Transcribing Audio...")
    transcribe_start = time.time()
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio)
    st.success("✅ Transcription complete in {:.2f} seconds.".format(time.time() - transcribe_start))

    # ------------------- ALIGNMENT -------------------
    st.markdown("### 🎯 Aligning Words...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)
    st.success("✅ Word alignment complete.")

    # ------------------- DIARIZATION (DISABLED) -------------------
    st.markdown("### 🧠 Speaker Diarization (Disabled for CPU Mode)")
    st.warning("⏭️ Skipped speaker diarization due to incompatible pyannote.audio version on CPU. Use GPU + pyannote.audio==2.1.1 for full diarization.")

    end_time = time.time()
    st.info(f"⏱️ Total processing time: {end_time - start_time:.2f} seconds")

    # ------------------- DISPLAY OUTPUT -------------------
    st.markdown("### 🎙️ Final Transcript")
    if "segments" in result and len(result["segments"]) > 0:
        for seg in result["segments"]:
            text = seg.get("text", "")
            st.markdown(f"🗒️ **Text:** {text}")
    else:
        st.error("❌ No transcription segments found.")

    # ------------------- DOWNLOAD SECTION -------------------
    st.markdown("### 💾 Download Transcription")
    transcript_text = "\n".join(
        [f"{seg.get('text', '')}" for seg in result.get("segments", [])]
    )

    transcript_file = "final_transcript.txt"
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
    st.info("📥 Please upload an audio file to begin.")
