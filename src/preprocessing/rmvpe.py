"""RMVPE pitch estimator — minimal inference-only implementation.

Architecture: DeepUNet + BiGRU → 360 pitch bins (20-cent resolution).
Cleaned up from RVC's infer/lib/rmvpe.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel


# ── Model components ──────────────────────────────────────────────────


class ConvBlockRes(nn.Module):
    def __init__(self, in_ch, out_ch, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=momentum),
            nn.ReLU(),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        sc = self.shortcut(x) if hasattr(self, "shortcut") else x
        return self.conv(x) + sc


class ResEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.conv = nn.ModuleList(
            [ConvBlockRes(in_ch if i == 0 else out_ch, out_ch, momentum)
             for i in range(n_blocks)]
        )
        self.pool = nn.AvgPool2d(kernel_size) if kernel_size else None

    def forward(self, x):
        for c in self.conv:
            x = c(x)
        return (x, self.pool(x)) if self.pool else x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_pad = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride, 1, out_pad, bias=False),
            nn.BatchNorm2d(out_ch, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList(
            [ConvBlockRes(out_ch * 2 if i == 0 else out_ch, out_ch, momentum)
             for i in range(n_blocks)]
        )

    def forward(self, x, skip):
        x = self.conv1(x)
        x = torch.cat((x, skip), dim=1)
        for c in self.conv2:
            x = c(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch, in_size, n_layers, kernel_size, n_blocks, out_ch=16,
                 momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                ResEncoderBlock(in_ch, out_ch, kernel_size, n_blocks, momentum))
            in_ch = out_ch
            out_ch *= 2
            in_size //= 2
        self.out_channel = out_ch
        self.out_size = in_size

    def forward(self, x):
        skips = []
        x = self.bn(x)
        for layer in self.layers:
            skip, x = layer(x)
            skips.append(skip)
        return x, skips


class Intermediate(nn.Module):
    def __init__(self, in_ch, out_ch, n_layers, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_ch, out_ch, None, n_blocks, momentum))
        for _ in range(n_layers - 1):
            self.layers.append(ResEncoderBlock(out_ch, out_ch, None, n_blocks, momentum))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_ch, n_layers, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            out_ch = in_ch // 2
            self.layers.append(ResDecoderBlock(in_ch, out_ch, stride, n_blocks, momentum))
            in_ch = out_ch

    def forward(self, x, skips):
        for i, layer in enumerate(self.layers):
            x = layer(x, skips[-1 - i])
        return x


class E2E(nn.Module):
    def __init__(self, n_blocks, n_gru, kernel_size,
                 en_de_layers=5, inter_layers=4, in_ch=1, en_out_ch=16):
        super().__init__()
        self.unet = nn.Module()
        self.unet.encoder = Encoder(in_ch, 128, en_de_layers, kernel_size,
                                    n_blocks, en_out_ch)
        self.unet.intermediate = Intermediate(
            self.unet.encoder.out_channel // 2,
            self.unet.encoder.out_channel, inter_layers, n_blocks)
        self.unet.decoder = Decoder(
            self.unet.encoder.out_channel, en_de_layers, kernel_size, n_blocks)
        self.cnn = nn.Conv2d(en_out_ch, 3, 3, padding=1)
        self.fc = nn.Sequential(
            nn.Module(),  # placeholder, replaced below
            nn.Linear(512, 360),
            nn.Dropout(0.25),
            nn.Sigmoid(),
        )
        self.fc[0] = BiGRU(3 * 128, 256, n_gru)

    def forward(self, mel):
        x = mel.transpose(-1, -2).unsqueeze(1)
        x, skips = self.unet.encoder(x)
        x = self.unet.intermediate(x)
        x = self.unet.decoder(x, skips)
        x = self.cnn(x).transpose(1, 2).flatten(-2)
        return self.fc(x)


class BiGRU(nn.Module):
    def __init__(self, in_features, hidden, num_layers):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden, num_layers=num_layers,
                          batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


# ── Mel spectrogram ───────────────────────────────────────────────────


class MelExtractor(nn.Module):
    def __init__(self, sr=16000, n_fft=1024, hop=160, n_mels=128, fmin=30, fmax=8000):
        super().__init__()
        mel_basis = librosa_mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                fmin=fmin, fmax=fmax, htk=True)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.n_fft = n_fft
        self.hop = hop
        self.win = n_fft
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, audio):
        spec = torch.stft(audio, self.n_fft, self.hop, self.win,
                          self.window, center=True, return_complex=True)
        mag = spec.abs()
        mel = self.mel_basis @ mag
        return torch.log(torch.clamp(mel, min=1e-5))


# ── RMVPE wrapper ─────────────────────────────────────────────────────


CENTS_MAPPING = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))


class RMVPE:
    def __init__(self, model_path: str, device="cuda"):
        self.device = torch.device(device)
        self.mel = MelExtractor().to(self.device)
        model = E2E(4, 1, (2, 2))
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval().to(self.device)
        self.model = model

    @torch.no_grad()
    def __call__(self, audio: np.ndarray, thred=0.03) -> np.ndarray:
        """audio: 1-D float32 numpy array at 16 kHz. Returns f0 in Hz at 100fps."""
        wav = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        mel = self.mel(wav)

        n_frames = mel.shape[-1]
        pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if pad > 0:
            mel = F.pad(mel, (0, pad))

        hidden = self.model(mel)[:, :n_frames].squeeze(0).cpu().numpy()

        center = np.argmax(hidden, axis=1)
        salience = np.pad(hidden, ((0, 0), (4, 4)))
        center += 4

        todo_s = np.array([salience[i, center[i]-4:center[i]+5] for i in range(len(center))])
        todo_c = np.array([CENTS_MAPPING[center[i]-4:center[i]+5] for i in range(len(center))])

        cents = np.sum(todo_s * todo_c, axis=1) / np.sum(todo_s, axis=1)
        cents[np.max(salience, axis=1) <= thred] = 0

        f0 = 10 * (2 ** (cents / 1200))
        f0[f0 == 10] = 0
        return f0
