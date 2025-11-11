import whisperx
import torch
import gc
import os

DEVICE = "cpu"
MODEL_SIZE = "small"
AUDIO_FILE = r"C:\Users\SOUMODIP\OneDrive\Desktop\speach_to_text_NLP\milestone_3\uploads\clean.wav"

def main():
    print("\n=== WhisperX Speech-to-Text Pipeline (CPU MODE - float32 enforced, no diarization) ===")

    if not os.path.exists(AUDIO_FILE):
        print(f"❌ Audio file not found: {AUDIO_FILE}")
        return

    # 1️⃣ Load model
    print("\n[1/4] Loading WhisperX model...")
    model = whisperx.load_model(MODEL_SIZE, device=DEVICE, compute_type="float32")

    # 2️⃣ Transcribe
    print("\n[2/4] Transcribing audio...")
    result = model.transcribe(AUDIO_FILE)
    print("✅ Transcription complete!")

    # 3️⃣ Alignment
    print("\n[3/4] Aligning timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, AUDIO_FILE, DEVICE)
    print("✅ Alignment complete!")

    # 4️⃣ Save output
    output_dir = os.path.dirname(AUDIO_FILE)
    output_file = os.path.join(output_dir, "final_transcription.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        for seg in result_aligned["segments"]:
            start, end = round(seg["start"], 2), round(seg["end"], 2)
            f.write(f"[{start:.2f} - {end:.2f}] {seg['text'].strip()}\n")

    print(f"\n✅ Transcription saved successfully:\n{output_file}")

    del model, model_a
    gc.collect()
    torch.cuda.empty_cache()
    print("\n🎯 Completed successfully on CPU (no diarization).\n")

if _name_ == "_main_":
    main()