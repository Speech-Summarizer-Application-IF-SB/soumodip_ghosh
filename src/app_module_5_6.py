# -*- coding: utf-8 -*-
import os
import sys
from pipeline import main as run_pipeline
from summarizer import summarize_text

# ---- FIX 1: Force UTF-8 output to avoid 'charmap' errors on Windows ----
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def main():
    print("\n===== MODULE 5 & 6: TRANSCRIPTION + SUMMARIZATION =====\n")

    # Check if audio path is passed as argument
    if len(sys.argv) < 2:
        print("⚠️  Usage: python app_module_5_6.py <audio_file_path>")
        sys.exit(1)

    audio_path = sys.argv[1]

    # Validate file path
    if not os.path.exists(audio_path):
        print(f"❌ Error: File not found — {audio_path}")
        sys.exit(1)

    print(f"🎧 Processing audio file: {audio_path}\n")

    # ---------------------------
    # STEP 1: TRANSCRIPTION
    # ---------------------------
    print("🔹 Step 1: Transcribing audio using WhisperX...\n")

    try:
        run_pipeline()  # pipeline.py handles transcription + diarization
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        sys.exit(1)

    # Read generated transcript file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transcript_file = os.path.join(base_dir, "transcription.txt")

    if not os.path.exists(transcript_file):
        print("❌ transcription.txt not found. Ensure pipeline.py generated it.")
        sys.exit(1)

    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    print("\n--- TRANSCRIPT ---")
    print(transcript)
    print("\n")

    # ---------------------------
    # STEP 2: SUMMARIZATION
    # ---------------------------
    print("🔹 Step 2: Generating summary...\n")

    try:
        summary = summarize_text(transcript)
        print("\n--- SUMMARY ---")
        print(summary)
    except Exception as e:
        print(f"❌ Summarization failed: {e}")
        sys.exit(1)

    # ---------------------------
    # SAVE OUTPUT
    # ---------------------------
    summary_file = os.path.join(base_dir, "summary.txt")

    try:
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"\n✅ Summary saved to: {summary_file}")
    except Exception as e:
        print(f"⚠️ Could not save summary: {e}")


if __name__ == "__main__":
    main()
