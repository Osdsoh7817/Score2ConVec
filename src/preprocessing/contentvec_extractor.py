"""Extract ContentVec features from audio files.

Uses HuggingFace transformers to load ContentVec (HuBERT-Base fine-tuned for
speaker-invariant content). Output: [T, 768] at 50fps (hop=320 @16kHz).
"""

import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from transformers import HubertModel


class ContentVecExtractor:
    SAMPLE_RATE = 16000
    HOP_SIZE = 320  # → 50fps

    def __init__(self, model_path: str = "lengyue233/content-vec-best",
                 device: str = "cuda", layer: int = 12):
        self.device = device
        self.layer = layer

        self.model = HubertModel.from_pretrained(
            model_path, local_files_only=Path(model_path).exists()
        )
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def extract(self, audio_path: str) -> np.ndarray:
        """Extract ContentVec features from an audio file.

        Returns:
            features: [T, 768] numpy array at 50fps
        """
        audio = self._load_audio(audio_path)
        wav = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        outputs = self.model(wav, output_hidden_states=True)
        features = outputs.hidden_states[self.layer].squeeze(0).cpu().numpy()
        return features

    def extract_to_file(self, audio_path: str, output_path: str, max_frames: int = None):
        """Extract and save as .npy file."""
        features = self.extract(audio_path)
        if max_frames is not None:
            features = features[:max_frames]
        np.save(output_path, features)
        return features.shape

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
