import os, re, wave, socket, tempfile, traceback
from datetime import datetime
import streamlit as st
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional dependency for PDF
try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="🎙️ SPEACH TO TEXT CONVERTION FROM LIVE RECORDING AND FROM FILES", page_icon="🎤", layout="wide")

# -------------------- STYLE --------------------
st.markdown("""
<style>
:root {
  --bg: #0b0f14;
  --panel: #11161e;
  --panel-2: #0f141b;
  --text: #e6edf3;
  --muted: #9aa7b2;
  --accent: #22d3ee;
  --accent-2: #7c3aed;
  --border: #1f2937;
}
html, body, .stApp {
  background: radial-gradient(1200px 600px at 10% -20%, #0d1218 0%, var(--bg) 60%) no-repeat fixed;
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue, Arial;
}
.app-hero {
  background: linear-gradient(135deg, rgba(34, 211, 238, .25), rgba(124, 58, 237, .22));
  border: 1px solid rgba(124, 58, 237, .35);
  border-radius: 18px;
  padding: 26px 28px;
  text-align: center;
}
.app-title {
  margin: 0;
  font-weight: 800;
  font-size: 34px;
  color: var(--text);
}
.app-subtitle {
  margin-top: 6px;
  color: var(--muted);
  font-size: 16px;
}
.card {
  background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
  margin-top: 20px;
}
.stButton > button {
  background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
  color: white !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
  padding: 10px 16px !important;
  transition: transform .12s ease, box-shadow .12s ease;
}
.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 16px 28px rgba(124, 58, 237, .18);
}
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
  <h1 class="app-title">🎙️SPEACH TO TEXT CONVERTION FROM LIVE RECORDING AND FROM FILES</h1>
  <div class="app-subtitle">“Audio Input → Automatic Transcription → Concise Summarization → Optimized Subtitles Export” / Email</div>
</div>
""", unsafe_allow_html=True)

# -------------------- SESSION DEFAULTS --------------------
if "audio_path" not in st.session_state: st.session_state.audio_path = None
if "transcription" not in st.session_state: st.session_state.transcription = ""
if "summary" not in st.session_state: st.session_state.summary = ""
if "meta" not in st.session_state: st.session_state.meta = {}
if "email_cfg" not in st.session_state:
    st.session_state.email_cfg = {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": "465",
        "email_user": "",
        "email_pass": "",
        "email_to": "",
        "subject": ""
    }

recognizer = sr.Recognizer()

# -------------------- HELPERS --------------------
def summarize_tfidf(text, num_sentences=3):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= num_sentences: return text
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()
    idx = np.argsort(scores)[::-1][:num_sentences]
    return " ".join([sentences[i] for i in sorted(idx)])

def transcribe_google(path):
    with sr.AudioFile(path) as src:
        audio = recognizer.record(src)
    return recognizer.recognize_google(audio)

def record_audio(duration=5, fs=44100):
    st.info(f"🎤 Recording for {duration} seconds... Speak now!")
    rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp.name, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(fs)
        wf.writeframes(rec.tobytes())
    st.success("✅ Recording complete! Click 'Process Audio'.")
    return tmp.name

def build_markdown(title, date, transcript, summary, speakers=""):
    return f"""# {title or 'Meeting Summary'}
Date: {date}{f"  |  Speakers: {speakers}" if speakers else ""}

---

## Transcription
{transcript or "(empty)"}

## Summary
{summary or "(empty)"}
"""

def md_to_pdf_bytes(md_text):
    from fpdf import FPDF
    import urllib.request
    font_path = os.path.join(tempfile.gettempdir(), "DejaVuSans.ttf")
    if not os.path.exists(font_path):
        urllib.request.urlretrieve(
            "https://github.com/dejavu-fonts/dejavu-fonts/raw/version_2_37/ttf/DejaVuSans.ttf",
            font_path
        )
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", size=12)
    width = pdf.w - pdf.l_margin - pdf.r_margin
    safe_text = md_text.encode("utf-8", "ignore").decode("utf-8")
    for line in safe_text.split("\n"):
        pdf.multi_cell(width, 6, line)
    return pdf.output(dest="S").encode("latin1", "ignore")

def send_email(smtp_host, smtp_port, user, pwd, to, subject, body, attachments):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    from email.header import Header
    from email.utils import formataddr
    msg = MIMEMultipart()
    msg["From"] = formataddr((str(Header("Speech-to-Text App", "utf-8")), user))
    msg["To"] = to
    msg["Subject"] = Header(subject or "Meeting Summary", "utf-8")
    msg.attach(MIMEText(body, "plain", "utf-8"))
    for fname, data, main, sub in attachments:
        part = MIMEBase(main, sub)
        part.set_payload(data)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{fname}"')
        msg.attach(part)
    with smtplib.SMTP_SSL(smtp_host, int(smtp_port)) as server:
        server.login(user, pwd)
        server.sendmail(user, [to], msg.as_string())

# -------------------- SIDEBAR --------------------
st.sidebar.header("🧾 Session Details")
title = st.sidebar.text_input("Title", "Meeting Summary")
date_str = st.sidebar.text_input("Date", datetime.now().strftime("%Y-%m-%d"))
speakers = st.sidebar.text_input("Speakers (optional)", "")

st.sidebar.markdown("---")
st.sidebar.subheader("📧 Email / Export")
cfg = st.session_state.email_cfg
cfg["smtp_host"] = st.sidebar.text_input("SMTP Host", cfg["smtp_host"])
cfg["smtp_port"] = st.sidebar.text_input("SMTP Port", cfg["smtp_port"])
cfg["email_user"] = st.sidebar.text_input("From Email", cfg["email_user"])
cfg["email_pass"] = st.sidebar.text_input("App Password", type="password", value=cfg["email_pass"])
cfg["email_to"] = st.sidebar.text_input("To Email", cfg["email_to"])
cfg["subject"] = st.sidebar.text_input
# -------------------- INPUT SECTION --------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🎧 Input Options")
st.markdown("<p style='color:#9aa7b2;'>Choose how you want to provide audio — record live or upload a file.</p>", unsafe_allow_html=True)

mode = st.radio("Select Input Method:", ["🎙️ Record from Microphone", "📂 Upload WAV File"], horizontal=True)

if mode.startswith("🎙️"):
    duration = st.slider("⏱️ Recording Duration (seconds)", 5, 60, 20)
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🎤 Start Recording"):
            st.session_state.audio_path = record_audio(duration)
    with c2:
        if st.session_state.audio_path:
            st.markdown("**🔊 Preview Recorded Audio**")
            st.audio(st.session_state.audio_path)
            st.success(f"✅ Recorded {duration} seconds of audio")

elif mode.startswith("📂"):
    uploaded = st.file_uploader("📂 Upload a .wav file", type=["wav"])
    if uploaded is not None:
        tmp_path = os.path.join(tempfile.gettempdir(), "uploaded_audio.wav")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())
        st.session_state.audio_path = tmp_path
        st.markdown("**🔊 Preview Uploaded Audio**")
        st.audio(st.session_state.audio_path, format="audio/wav")
        st.success("✅ File uploaded successfully!")

st.markdown('</div>', unsafe_allow_html=True)
# -------------------- OUTPUT SECTION --------------------
if st.session_state.audio_path:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Output")

    if st.button("🚀 Process Audio"):
        try:
            with st.spinner("🎧 Transcribing..."):
                text = transcribe_google(st.session_state.audio_path)
            with st.spinner("🧠 Summarizing..."):
                summary = summarize_tfidf(text)
            st.session_state.transcription = text
            st.session_state.summary = summary
            st.success("✅ Done! See below.")
        except Exception as e:
            st.error(f"❌ {e}")
            st.code(traceback.format_exc())

    if st.session_state.transcription:
        st.markdown("**📝 Transcription**")
        st.text_area("", st.session_state.transcription, height=200)

    if st.session_state.summary:
        st.markdown("**🧾 Summary**")
        st.text_area("", st.session_state.summary, height=150)

        md = build_markdown(title, date_str, st.session_state.transcription or "", st.session_state.summary or "", speakers)
        st.download_button("⬇️ Download Markdown (.md)", data=md.encode("utf-8"), file_name=f"{title or 'summary'}.md", mime="text/markdown")
        if HAS_FPDF:
            pdf_bytes = md_to_pdf_bytes(md)
            st.download_button("📄 Download PDF (.pdf)", data=pdf_bytes, file_name=f"{title or 'summary'}.pdf", mime="application/pdf")

        # Replay & Clear
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🎧 Replay Last Recording") and st.session_state.audio_path:
                st.audio(st.session_state.audio_path)
        with c2:
            if st.button("🗑️ Clear All"):
                for k in ["audio_path", "transcription", "summary"]:
                    st.session_state[k] = None if k == "audio_path" else ""
                st.experimental_rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    # -------------------- FOOTER --------------------
st.markdown(
    "<hr><p style='text-align:center;color:#94A3B8;'>✨ Built with Streamlit by <strong>Soumodip Ghosh</strong></p>",
    unsafe_allow_html=True
)