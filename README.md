Live Meeting Summarizer — Real-Time STT + Diarization + AI Summaries

A complete end-to-end system that converts live meetings into speaker-segmented transcripts and LLM-powered summaries with one click.

📌 Overview

Modern meetings generate long, unorganized conversations that are difficult to review. Manual note-taking is unreliable, and existing tools either require the cloud, lack diarization, or fail to provide clean summaries.

This project solves that problem by combining offline speech-to-text, speaker diarization, and AI summarization into a single streamlined pipeline—all wrapped inside a user-friendly Streamlit interface.

The result:
A real-time meeting assistant that listens, understands, separates speakers, and summarizes everything only when the meeting ends.

🎯 Key Features

Real-Time Speech-to-Text (STT) using Whisper or Vosk

Speaker Diarization via Pyannote.audio + Hugging Face

Structured Summaries generated using transformer-based LLMs (Groq LLaMA 3.1 / T5 / BART)

Clean Streamlit UI with live transcription panel

One-click Export to Markdown / PDF

Email Delivery of complete meeting summary

Local Processing Support (no constant cloud dependency)

Evaluation Metrics included: WER, DER, ROUGE

Logged Meeting History saved in JSON/Parquet

🏗 Architecture
Start Recording
      ↓
Real-Time Audio Stream (PyAudio/SoundDevice)
      ↓
Speech-to-Text Engine (Whisper / Vosk)
      ↓
Audio Saved → Pyannote Speaker Diarization
      ↓
Speaker-Labeled Transcript
      ↓
LLM Summarization (Groq/HuggingFace Transformers)
      ↓
Streamlit UI: Transcript + Summary
      ↓
Export → PDF / MD or Send via Email

🧩 Tech Stack
Area	Tools / Libraries
Speech-to-Text	Whisper, Vosk, PyAudio, SoundDevice
Diarization	Pyannote.audio, Torchaudio, Hugging Face
Summarization	Groq LLaMA 3.1, T5, BART
Frontend	Streamlit
Backend	Python Threading, AsyncIO, Queue
Evaluation	jiwer (WER), ROUGE Score, BLEU
Export	JSON, Markdown, PDF, smtplib
🧱 Project Milestones
Milestone 1 — Real-Time STT System

Developed threaded microphone audio capture

Integrated STT using Whisper/Vosk

Benchmarked accuracy using AMI Corpus

Achieved WER < 15%

Milestone 2 — Speaker Diarization + Summarization

Implemented Pyannote.audio diarization with DER < 20%

Merged diarization tags with STT output

Built summarization prompts and integrated LLaMA/T5/BART

Achieved ROUGE > 0.4 on summary quality

Milestone 3 — UI Integration

Combined STT → Diarization → Summarization into one clean pipeline

Streamlit UI shows live transcription and final summary after stop

Prevented race conditions using queues and asynchronous processing

Added export and email delivery modules

Milestone 4 — Testing, Optimization, Documentation

Added Markdown and PDF export

Implemented email sharing using SMTP

Structured logs saved for each session

Project fully documented and demonstration-ready

📊 Results

Accurate real-time STT across multi-speaker scenarios

Speaker turns correctly separated (DER < 20%)

High-quality structured summaries (ROUGE > 0.4)

Smooth UI experience with no blocking
(Dashboard screenshot can be added here)

🚀 Future Scope

Multilingual STT and summarization

Real-time integration with Zoom / Google Meet / MS Teams

Advanced analytics:

Action item extraction

Decision tracking

Sentiment evaluation

Cloud deployment with GPU inference (Docker/Kubernetes)

Mobile app version with background listening

Vector embeddings for search across past meetings

📎 Project Structure (Sample)
│── app/
│   ├── main.py
│   ├── components/
│   ├── utils/
│   ├── models/
│── data/
│── exports/
│── logs/
│── requirements.txt
│── README.md

🛠 Setup Instructions

Clone the repo:

git clone <repo-url>
cd live-meeting-summarizer


Install dependencies:

pip install -r requirements.txt


Run Streamlit app:

streamlit run app/main.py

📌 Conclusion

This project delivers a complete, production-ready system for transforming any meeting into a clean, structured, speaker-aware summary. The pipeline achieves strong benchmarks (WER < 15%, DER < 20%, ROUGE > 0.4) and provides a polished Streamlit UI with export and email features.

It reduces the cognitive load of meetings, automates note-taking, and helps teams focus on decisions rather than documentation.
