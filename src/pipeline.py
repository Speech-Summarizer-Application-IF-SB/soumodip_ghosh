import torch
import whisperx
from pathlib import Path

# ------------------ CONFIG ------------------
AUDIO_FILE = "../uploads/clean.wav"  # Path to your audio file
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # CPU or GPU
COMPUTE_TYPE = "float32"  # Use float32 as requested
WHISPER_MODEL = "small.en"  # English model

# ------------------ LOAD MODEL ------------------
print(f"[INFO] Device detected: {DEVICE}")
print(f"[INFO] Using compute_type: {COMPUTE_TYPE} and model: {WHISPER_MODEL}")

print("[STEP] Loading WhisperX ASR model...")
asr_model = whisperx.load_model(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)

# ------------------ TRANSCRIBE ------------------
print("[STEP] Transcribing audio...")
transcription = asr_model.transcribe(AUDIO_FILE)

# ------------------ LOAD ALIGNER ------------------
print("[STEP] Aligning words to timestamps...")
aligner = whisperx.load_align_model(language_code=transcription["language"], device=DEVICE)
result_aligned = whisperx.align(transcription["segments"], aligner, AUDIO_FILE, DEVICE)

# ------------------ SPEAKER DIARIZATION ------------------
try:
    print("[STEP] Running speaker diarization...")
    from pyannote.audio import Pipeline

    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=None)
    diarization = diarization_pipeline(AUDIO_FILE)
    # Assign speaker labels
    for segment in result_aligned["segments"]:
        segment["speaker"] = None
        for turn in diarization.itertracks(yield_label=True):
            segment_start, segment_end = segment["start"], segment["end"]
            turn_start, turn_end, speaker_label = turn[0].start, turn[0].end, turn[1]
            if segment_start >= turn_start and segment_end <= turn_end:
                segment["speaker"] = speaker_label
except Exception as e:
    print(f"[WARN] Diarization failed: {e}")
    print("[INFO] Proceeding without speaker labels...")

# ------------------ SAVE TRANSCRIPTION ------------------
output_file = Path("../transcription.txt")
print("[STEP] Building speaker-labeled transcription...")
with open(output_file, "w", encoding="utf-8") as f:
    for segment in result_aligned["segments"]:
        speaker = segment.get("speaker", "Unknown")
        text = segment["text"]
        start = segment["start"]
        end = segment["end"]
        f.write(f"[{speaker}] {start:.2f}-{end:.2f}: {text}\n")

print(f"[OK] Transcription written to: {output_file}")
print("[DONE] Pipeline finished successfully.")
