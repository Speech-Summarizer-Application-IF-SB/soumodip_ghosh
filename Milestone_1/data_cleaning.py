import librosa
import numpy as np
import soundfile as sf
from scipy.signal import medfilt
file_path = r"C:\Users\SOUMODIP\Downloads\harvard.wav\harvard.wav"
# Load the audio
y, sr = librosa.load(file_path, sr=None)
# Compute magnitude spectrogram
s_full, phase = librosa.magphase(librosa.stft(y))
# Compute average noise power safely
frames_0_1s = min(int(0.1 * s_full.shape[1]), s_full.shape[1])
noise_power = np.mean(s_full[:, :frames_0_1s], axis=1)
# Example: apply a simple median filter for cleaning (placeholder)
y_clean = medfilt(y, kernel_size=3)
# Save cleaned audio properly
sf.write(file='clean.wav', data=y_clean, samplerate=sr)
print("Cleaned audio saved as 'clean.wav'")
