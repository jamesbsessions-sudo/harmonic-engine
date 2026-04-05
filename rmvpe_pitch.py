"""
RMVPE pitch extractor for the harmonic engine.

Drop-in replacement for the torchcrepe melody extraction step.
Extracts vocal f0 directly from polyphonic audio (full mix) — no need
to run Demucs stem separation first for melody.

Usage in app.py:
    from rmvpe_pitch import RMVPEPitchExtractor
    
    pitch_extractor = RMVPEPitchExtractor(
        model_path="models/rmvpe.pt",  # downloaded from HuggingFace
        device="cpu",                   # or "cuda" on Railway if GPU available
    )
    
    # Extract from full mix (the whole point — bypasses Demucs for melody)
    f0, voiced_flag, voiced_probs = pitch_extractor.extract(audio, sr)
    
    # Or from vocal stem if you still want to compare
    f0, voiced_flag, voiced_probs = pitch_extractor.extract(vocal_audio, sr)

Download weights:
    huggingface-cli download lj1995/VoiceConversionWebUI rmvpe.pt --local-dir models/
    
    Or manually from:
    https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import torch

from rmvpe_model import E2E, MelSpectrogram

logger = logging.getLogger(__name__)

# RMVPE operates at 16kHz with hop_length=160, giving 10ms frames (100 fps)
RMVPE_SR = 16000
RMVPE_HOP = 160
RMVPE_FRAME_RATE = RMVPE_SR / RMVPE_HOP  # 100 fps

# 360 bins covering C1 (32.7 Hz) to B6 (1975.5 Hz) in 20-cent intervals
# cents_mapping[i] = 1997.3794... + 20*i  (this is the cent value for bin i)
CENTS_MAPPING = 20 * np.arange(360) + 1997.3794084376191


class RMVPEPitchExtractor:
    """Vocal pitch extractor using RMVPE.
    
    Extracts f0 from polyphonic audio at 100 fps (10ms hop).
    Returns arrays compatible with the harmonic engine's melody analysis.
    """
    
    def __init__(
        self,
        model_path: str = "models/rmvpe.pt",
        device: Optional[str] = None,
        is_half: bool = False,
        voicing_threshold: float = 0.03,
    ):
        """
        Args:
            model_path: Path to rmvpe.pt weights file.
            device: Torch device. Auto-detects if None.
            is_half: Use float16 inference (faster on GPU, skip on CPU).
            voicing_threshold: Confidence threshold for voiced/unvoiced decision.
                Lower = more sensitive (catches quiet vocals), higher = stricter.
                0.03 is the RVC default; try 0.015-0.05 for your tracks.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.is_half = is_half and device != "cpu"
        self.voicing_threshold = voicing_threshold
        
        # Padded cents mapping for local average decoding
        self.cents_mapping = np.pad(CENTS_MAPPING, (4, 4))  # -> 368 bins
        
        # Load model
        logger.info(f"Loading RMVPE model from {model_path}")
        self.model = self._load_model(model_path)
        
        # Mel extractor matching RMVPE training config:
        # 128 mel bins, 16kHz SR, 1024 win, 160 hop, 30-8000 Hz range
        self.mel_extractor = MelSpectrogram(
            is_half=self.is_half,
            n_mel_channels=128,
            sampling_rate=RMVPE_SR,
            win_length=1024,
            hop_length=RMVPE_HOP,
            n_fft=None,  # defaults to win_length
            mel_fmin=30,
            mel_fmax=8000,
        ).to(self.device)
    
    def _load_model(self, model_path: str) -> E2E:
        """Load RMVPE weights into the E2E architecture."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RMVPE weights not found at {model_path}. "
                f"Download from: https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt\n"
                f"  huggingface-cli download lj1995/VoiceConversionWebUI rmvpe.pt --local-dir {Path(model_path).parent}"
            )
        
        model = E2E(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt)
        model.eval()
        
        if self.is_half:
            model = model.half()
        
        model = model.to(self.device)
        logger.info(f"RMVPE loaded on {self.device} (half={self.is_half})")
        return model
    
    def extract(
        self,
        audio: np.ndarray,
        sr: int,
        f0_min: float = 50.0,
        f0_max: float = 1500.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract vocal f0 from audio.
        
        Args:
            audio: Audio signal, mono, float32. Can be full polyphonic mix.
            sr: Sample rate of the input audio.
            f0_min: Minimum expected f0 in Hz (for post-filtering).
            f0_max: Maximum expected f0 in Hz (for post-filtering).
        
        Returns:
            f0: Array of f0 values in Hz, shape (n_frames,). 0 = unvoiced.
            voiced_flag: Boolean array, True where vocal pitch detected.
            voiced_probs: Confidence values per frame (max salience), shape (n_frames,).
        """
        # Ensure mono
        if audio.ndim > 1:
            audio = librosa.to_mono(audio.T)
        
        # Resample to 16kHz if needed
        if sr != RMVPE_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=RMVPE_SR)
        
        audio = audio.astype(np.float32)
        
        # Compute mel spectrogram
        audio_tensor = torch.from_numpy(audio).float().to(self.device).unsqueeze(0)
        mel = self.mel_extractor(audio_tensor, center=True)
        
        # Run model inference
        with torch.no_grad():
            n_frames = mel.shape[-1]
            # Pad to multiple of 32 frames (model requirement)
            pad_amount = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            mel_padded = F.pad(mel, (0, pad_amount), mode="constant")
            hidden = self.model(mel_padded)
            hidden = hidden[:, :n_frames]  # trim padding
        
        # Decode to f0
        salience = hidden.squeeze(0).cpu().numpy()  # (n_frames, 360)
        f0, voiced_probs = self._decode(salience, self.voicing_threshold)
        
        # Post-filter: zero out f0 outside expected vocal range
        out_of_range = (f0 > 0) & ((f0 < f0_min) | (f0 > f0_max))
        f0[out_of_range] = 0.0
        
        voiced_flag = f0 > 0
        
        return f0, voiced_flag, voiced_probs
    
    def extract_with_times(
        self,
        audio: np.ndarray,
        sr: int,
        f0_min: float = 50.0,
        f0_max: float = 1500.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract f0 with corresponding timestamps.
        
        Same as extract() but also returns a time array.
        
        Returns:
            times: Timestamps in seconds for each frame.
            f0: f0 values in Hz.
            voiced_flag: Boolean voiced/unvoiced mask.
            voiced_probs: Confidence per frame.
        """
        f0, voiced_flag, voiced_probs = self.extract(audio, sr, f0_min, f0_max)
        times = np.arange(len(f0)) / RMVPE_FRAME_RATE
        return times, f0, voiced_flag, voiced_probs
    
    def _decode(
        self, salience: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode salience matrix to f0 values using local weighted average.
        
        Args:
            salience: (n_frames, 360) sigmoid activations from the model.
            threshold: Voicing confidence threshold.
        
        Returns:
            f0: (n_frames,) array of f0 in Hz, 0 for unvoiced frames.
            max_salience: (n_frames,) max confidence per frame.
        """
        center = np.argmax(salience, axis=1)
        max_salience = np.max(salience, axis=1)
        
        # Pad salience for windowed averaging
        salience_padded = np.pad(salience, ((0, 0), (4, 4)))
        center += 4  # offset for padding
        
        # Local weighted average over 9 bins centered on peak
        todo_salience = []
        todo_cents = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience_padded[idx, starts[idx]:ends[idx]])
            todo_cents.append(self.cents_mapping[starts[idx]:ends[idx]])
        
        todo_salience = np.array(todo_salience)  # (n_frames, 9)
        todo_cents = np.array(todo_cents)          # (n_frames, 9)
        
        product_sum = np.sum(todo_salience * todo_cents, axis=1)
        weight_sum = np.sum(todo_salience, axis=1)
        cents_pred = product_sum / (weight_sum + 1e-10)
        
        # Convert cents to Hz: f = 10 * 2^(cents/1200)
        f0 = 10.0 * (2.0 ** (cents_pred / 1200.0))
        
        # Apply voicing threshold
        f0[max_salience <= threshold] = 0.0
        
        # Clean up: the base frequency (10 Hz) means unvoiced
        f0[f0 == 10.0] = 0.0
        
        return f0, max_salience
    
    def f0_to_midi(self, f0: np.ndarray) -> np.ndarray:
        """Convert f0 array to MIDI note numbers. Unvoiced frames → NaN."""
        midi = np.full_like(f0, np.nan)
        voiced = f0 > 0
        midi[voiced] = 12.0 * np.log2(f0[voiced] / 440.0) + 69.0
        return midi
    
    def f0_to_note_name(self, f0_hz: float) -> str:
        """Convert a single f0 value to note name (e.g. 'A4', 'C#5')."""
        if f0_hz <= 0:
            return "—"
        midi = 12.0 * np.log2(f0_hz / 440.0) + 69.0
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        note_idx = int(round(midi)) % 12
        octave = int(round(midi)) // 12 - 1
        cents_off = round((midi - round(midi)) * 100)
        name = f"{note_names[note_idx]}{octave}"
        if cents_off != 0:
            name += f" ({cents_off:+d}¢)"
        return name


# Need this import for F.pad in extract()
import torch.nn.functional as F
