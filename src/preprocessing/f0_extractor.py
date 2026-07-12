"""F0 extraction at 50fps to match ContentVec frame rate.

Supports RMVPE (recommended for singing) and WORLD DIO as fallback.
"""

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf


class F0Extractor:
    SAMPLE_RATE = 16000
    HOP_SIZE = 320  # 50fps to match ContentVec

    def __init__(self, method: str = "rmvpe", f0_min: float = 50, f0_max: float = 1100,
                 device: str = "cuda"):
        self.method = method
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.device = device

        if method == "rmvpe":
            self._init_rmvpe()

    def _init_rmvpe(self):
        try:
            from rmvpe import RMVPE
            self.rmvpe = RMVPE("pretrained/rmvpe.pt", device=self.device)
        except ImportError:
            print("RMVPE not available, falling back to WORLD DIO")
            self.method = "dio"

    def extract(self, audio_path: str, n_frames: int = None) -> np.ndarray:
        """Extract F0 from audio file.

        Returns:
            f0: [T] numpy array in Hz, 0 for unvoiced frames. At 50fps.
        """
        audio = self._load_audio(audio_path)

        if self.method == "rmvpe":
            f0 = self._extract_rmvpe(audio)
        else:
            f0 = self._extract_dio(audio)

        if n_frames is not None:
            f0 = self._match_length(f0, n_frames)

        return f0

    def _extract_rmvpe(self, audio: np.ndarray) -> np.ndarray:
        f0 = self.rmvpe.infer_from_audio(audio, sample_rate=self.SAMPLE_RATE, thred=0.03)
        target_len = len(audio) // self.HOP_SIZE
        f0 = self._match_length(f0, target_len)
        return f0

    def _extract_dio(self, audio: np.ndarray) -> np.ndarray:
        import pyworld as pw
        audio_f64 = audio.astype(np.float64)
        f0, t = pw.dio(audio_f64, self.SAMPLE_RATE, f0_floor=self.f0_min, f0_ceil=self.f0_max,
                       frame_period=self.HOP_SIZE / self.SAMPLE_RATE * 1000)
        f0 = pw.stonemask(audio_f64, f0, t, self.SAMPLE_RATE)
        return f0.astype(np.float32)

    def _match_length(self, f0: np.ndarray, target_len: int) -> np.ndarray:
        """Interpolate F0 to exact target frame count. NEVER truncate."""
        if len(f0) == target_len:
            return f0
        f0_t = torch.from_numpy(f0).float().unsqueeze(0).unsqueeze(0)
        f0_resampled = F.interpolate(f0_t, size=target_len, mode="linear", align_corners=False)
        return f0_resampled.squeeze().numpy()

    def _load_audio(self, path: str) -> np.ndarray:
        try:
            audio, sr = sf.read(path, dtype="float32")
        except Exception:
            audio, sr = librosa.load(path, sr=None, mono=True)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLE_RATE)
        return audio
