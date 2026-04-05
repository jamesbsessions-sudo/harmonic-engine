"""
Harmonic Taste Profiling Engine — MVP Backend v8
Full feature set: Demucs stems, bass+chroma+Chordino chord detection,
RMVPE melody extraction (polyphonic, replaces torchcrepe),
time signature, half-time detection,
key centre, harmonic rhythm, spectral features, rhythm analysis.
"""

import os
import uuid
import tempfile
import subprocess
import shutil
from pathlib import Path
from collections import Counter

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

import librosa
import numpy as np
import torch
from chord_extractor.extractors import Chordino
from rmvpe_pitch import RMVPEPitchExtractor, RMVPE_FRAME_RATE

app = FastAPI(title="Harmonic Taste Profiling Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORK_DIR = Path(tempfile.gettempdir()) / "harmonic-engine"
WORK_DIR.mkdir(exist_ok=True)

chordino = Chordino(roll_on=1)

# ── RMVPE pitch extractor (replaces torchcrepe) ─────────────────
# Downloads weights on first run if not present.
# Extracts vocal pitch directly from polyphonic audio — no stem separation needed for melody.
RMVPE_MODEL_PATH = os.environ.get("RMVPE_MODEL_PATH", "models/rmvpe.pt")
rmvpe_extractor = None  # Lazy-loaded on first use to avoid blocking startup


# ── Note / interval utilities ─────────────────────────────────────
NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

NOTE_TO_NUM = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'Fb': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7,
    'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11,
}

ENHARMONIC_MAP = {
    'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb',
}
FLAT_KEYS = {'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb',
             'Dm', 'Gm', 'Cm', 'Fm', 'Bbm', 'Ebm'}

INTERVAL_NAMES = {
    0: 'unison', 1: 'minor 2nd', 2: 'major 2nd', 3: 'minor 3rd',
    4: 'major 3rd', 5: 'perfect 4th', 6: 'tritone', 7: 'perfect 5th',
    8: 'minor 6th', 9: 'major 6th', 10: 'minor 7th', 11: 'major 7th',
    12: 'octave',
}


def normalise_enharmonic(name: str, key: str) -> str:
    key_root = key.split()[0] if ' ' in key else key
    use_flats = key_root in FLAT_KEYS or 'b' in key_root
    if not use_flats:
        return name
    for sharp, flat in ENHARMONIC_MAP.items():
        if name.startswith(sharp):
            return flat + name[len(sharp):]
    return name


def hz_to_note_name(hz):
    """Convert Hz to note name, returning None for invalid values."""
    if hz <= 0 or np.isnan(hz):
        return None
    return librosa.hz_to_note(hz)


def hz_to_midi(hz):
    """Convert Hz to MIDI note number."""
    if hz <= 0 or np.isnan(hz):
        return None
    return 12 * np.log2(hz / 440.0) + 69


# ── Chord quality detection from combined evidence ────────────────
def identify_chord_quality(root_name: str, combined_chroma: np.ndarray,
                           base_threshold: float = 0.3) -> str:
    root_num = NOTE_TO_NUM.get(root_name, 0)

    if combined_chroma.max() > 0:
        chroma_norm = combined_chroma / combined_chroma.max()
    else:
        return root_name

    def has_interval(semitones, threshold=None):
        if threshold is None:
            threshold = base_threshold
        idx = (root_num + semitones) % 12
        return chroma_norm[idx] >= threshold

    has_minor_3rd = has_interval(3)
    has_major_3rd = has_interval(4)
    has_perfect_4th = has_interval(5)
    has_dim_5th = has_interval(6)
    has_perfect_5th = has_interval(7)
    has_minor_6th = has_interval(8)
    has_major_6th = has_interval(9)
    has_minor_7th = has_interval(10)
    has_major_7th = has_interval(11)
    has_9th = has_interval(2, threshold=0.55)

    if has_minor_3rd and has_dim_5th and not has_perfect_5th:
        if has_minor_7th:
            return root_name + "m7b5"
        return root_name + "dim"

    if has_major_3rd and has_minor_6th and not has_perfect_5th:
        return root_name + "aug"

    if has_perfect_4th and not has_major_3rd and not has_minor_3rd:
        if has_minor_7th:
            return root_name + "7sus4"
        return root_name + "sus4"

    if has_major_3rd and has_major_7th:
        if has_9th:
            return root_name + "maj9"
        return root_name + "maj7"

    if has_major_3rd and has_minor_7th:
        if has_9th:
            return root_name + "9"
        return root_name + "7"

    if has_minor_3rd and has_minor_7th:
        if has_9th:
            return root_name + "m9"
        return root_name + "m7"

    if has_minor_3rd and has_major_7th:
        return root_name + "mMaj7"

    if has_major_3rd and has_major_6th:
        return root_name + "6"

    if has_minor_3rd and has_major_6th:
        return root_name + "m6"

    if has_minor_3rd:
        return root_name + "m"

    if has_major_3rd:
        return root_name + "maj"

    if has_perfect_5th:
        return root_name + "5"

    return root_name


# ── Chordino note evidence helpers ────────────────────────────────
def chord_label_to_notes(label: str) -> set:
    if label == 'N' or not label:
        return set()

    if ':' in label:
        root_str, quality = label.split(':', 1)
    elif len(label) > 1 and label[1] in ('#', 'b'):
        root_str = label[:2]
        quality = label[2:]
    else:
        root_str = label[0]
        quality = label[1:]

    root = NOTE_TO_NUM.get(root_str, None)
    if root is None:
        return set()

    notes = {root}
    quality_lower = quality.lower()

    if 'm' in quality_lower and 'maj' not in quality_lower:
        notes.add((root + 3) % 12)
    else:
        notes.add((root + 4) % 12)

    if 'dim' in quality_lower:
        notes.add((root + 6) % 12)
    elif 'aug' in quality_lower:
        notes.add((root + 8) % 12)
    else:
        notes.add((root + 7) % 12)

    if 'maj7' in quality_lower or 'maj9' in quality_lower:
        notes.add((root + 11) % 12)
    elif '7' in quality_lower or '9' in quality_lower:
        notes.add((root + 10) % 12)

    if '9' in quality_lower:
        notes.add((root + 2) % 12)
    if '6' in quality_lower:
        notes.add((root + 9) % 12)
    if 'sus4' in quality_lower or 'sus' in quality_lower:
        notes.discard((root + 4) % 12)
        notes.discard((root + 3) % 12)
        notes.add((root + 5) % 12)

    return notes


# ── Demucs stem separation ────────────────────────────────────────
def separate_full_stems(wav_path: str, output_dir: str) -> dict:
    """
    Ensemble stem separation: run htdemucs_ft and mdx_extra sequentially,
    then average their outputs for more consistent, accurate stems.
    Falls back to single htdemucs if ensemble fails.
    """
    import soundfile as sf

    song_name = Path(wav_path).stem
    stem_names = ["bass", "drums", "vocals", "other"]

    # ── Run model 1: htdemucs_ft (fine-tuned, best overall) ────
    cmd1 = [
        "python", "-m", "demucs",
        "-n", "htdemucs_ft",
        "--shifts", "1",
        "--out", output_dir,
        wav_path,
    ]
    result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=600)
    model1_ok = result1.returncode == 0
    model1_dir = Path(output_dir) / "htdemucs_ft" / song_name

    # ── Run model 2: mdx_extra (different architecture) ────────
    cmd2 = [
        "python", "-m", "demucs",
        "-n", "mdx_extra",
        "--shifts", "1",
        "--out", output_dir,
        wav_path,
    ]
    result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)
    model2_ok = result2.returncode == 0
    model2_dir = Path(output_dir) / "mdx_extra" / song_name

    # ── Average the stems if both models succeeded ─────────────
    if model1_ok and model2_ok and model1_dir.exists() and model2_dir.exists():
        ensemble_dir = Path(output_dir) / "ensemble" / song_name
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        stems = {}
        for stem_name in stem_names:
            path1 = model1_dir / f"{stem_name}.wav"
            path2 = model2_dir / f"{stem_name}.wav"

            if path1.exists() and path2.exists():
                # Load both stems and average
                y1, sr1 = sf.read(str(path1))
                y2, sr2 = sf.read(str(path2))

                # Match lengths (trim to shorter)
                min_len = min(len(y1), len(y2))
                y1 = y1[:min_len]
                y2 = y2[:min_len]

                # Average
                y_avg = (y1 + y2) / 2.0

                # Save averaged stem
                avg_path = ensemble_dir / f"{stem_name}.wav"
                sf.write(str(avg_path), y_avg, sr1)
                stems[stem_name] = str(avg_path)

            elif path1.exists():
                stems[stem_name] = str(path1)
            elif path2.exists():
                stems[stem_name] = str(path2)

        if stems:
            return stems

    # ── Fallback: use whichever model succeeded ────────────────
    if model1_ok and model1_dir.exists():
        stems = {}
        for stem_name in stem_names:
            stem_path = model1_dir / f"{stem_name}.wav"
            if stem_path.exists():
                stems[stem_name] = str(stem_path)
        if stems:
            return stems

    if model2_ok and model2_dir.exists():
        stems = {}
        for stem_name in stem_names:
            stem_path = model2_dir / f"{stem_name}.wav"
            if stem_path.exists():
                stems[stem_name] = str(stem_path)
        if stems:
            return stems

    # ── Last resort: basic htdemucs ────────────────────────────
    cmd3 = [
        "python", "-m", "demucs",
        "-n", "htdemucs",
        "--shifts", "1",
        "--out", output_dir,
        wav_path,
    ]
    result3 = subprocess.run(cmd3, capture_output=True, text=True, timeout=300)
    if result3.returncode != 0:
        raise Exception(f"All Demucs models failed. Last error: {result3.stderr[:300]}")

    fallback_dir = Path(output_dir) / "htdemucs" / song_name
    stems = {}
    for stem_name in stem_names:
        stem_path = fallback_dir / f"{stem_name}.wav"
        if stem_path.exists():
            stems[stem_name] = str(stem_path)

    if not stems:
        raise Exception(f"No stems found after fallback")
    return stems


# ── Bass root detection ───────────────────────────────────────────
def detect_bass_roots(bass_path: str, beat_frames, sr=44100):
    y_bass, sr = librosa.load(bass_path, sr=sr, mono=True)

    bass_chroma = librosa.feature.chroma_cqt(
        y=y_bass, sr=sr,
        fmin=librosa.note_to_hz('C1'),
        n_octaves=4,
    )

    if len(beat_frames) > 1:
        beat_chroma = librosa.util.sync(bass_chroma, beat_frames, aggregate=np.median)
    else:
        beat_chroma = bass_chroma

    bass_notes = []
    for i in range(beat_chroma.shape[1]):
        frame = beat_chroma[:, i]
        if frame.sum() > 0:
            bass_notes.append(NOTE_NAMES[frame.argmax()])
        else:
            bass_notes.append(None)

    return bass_notes, y_bass


def check_bass_energy(y_bass, sr=44100):
    """
    Check whether the bass stem has meaningful energy.
    Returns a per-segment energy profile and an overall confidence score.
    Low energy = no bass instrument present, should fall back to low-pass.
    """
    rms = librosa.feature.rms(y=y_bass, frame_length=2048, hop_length=512)
    avg_rms = float(np.mean(rms))
    max_rms = float(np.max(rms))

    # Threshold: if average RMS is very low, bass stem is essentially empty
    # These thresholds are tuned for normalised audio
    if avg_rms < 0.005:
        return 0.0, "empty"
    elif avg_rms < 0.02:
        return 0.3, "weak"
    elif avg_rms < 0.05:
        return 0.6, "moderate"
    else:
        return 1.0, "strong"


def detect_bass_roots_lowpass(other_path: str, beat_frames, sr=44100):
    """
    Fallback bass root detection: low-pass filter the 'other' stem
    to extract bass frequencies when there's no dedicated bass instrument.
    """
    from scipy.signal import butter, sosfilt

    y_other, sr = librosa.load(other_path, sr=sr, mono=True)

    # Low-pass filter at 300Hz to isolate bass frequencies
    nyquist = sr / 2
    cutoff = 300
    sos = butter(5, cutoff / nyquist, btype='low', output='sos')
    y_bass_filtered = sosfilt(sos, y_other)

    bass_chroma = librosa.feature.chroma_cqt(
        y=y_bass_filtered, sr=sr,
        fmin=librosa.note_to_hz('C1'),
        n_octaves=3,
    )

    if len(beat_frames) > 1:
        beat_chroma = librosa.util.sync(bass_chroma, beat_frames, aggregate=np.median)
    else:
        beat_chroma = bass_chroma

    bass_notes = []
    for i in range(beat_chroma.shape[1]):
        frame = beat_chroma[:, i]
        if frame.sum() > 0:
            bass_notes.append(NOTE_NAMES[frame.argmax()])
        else:
            bass_notes.append(None)

    return bass_notes


# ── Harmonic chroma ───────────────────────────────────────────────
def get_harmonic_chroma(other_path: str, beat_frames, sr=44100):
    y_other, sr = librosa.load(other_path, sr=sr, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y_other, sr=sr)
    if len(beat_frames) > 1:
        return librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    return chroma


# ── Chordino note evidence ────────────────────────────────────────
def get_chordino_note_evidence(other_path: str, beat_times):
    try:
        chord_changes = chordino.extract(other_path)
    except Exception:
        return None

    if not chord_changes:
        return None

    n_beats = len(beat_times)
    evidence = np.zeros((12, n_beats))

    for beat_idx, bt in enumerate(beat_times):
        active_chord = None
        for j, change in enumerate(chord_changes):
            if change.timestamp <= bt:
                active_chord = change.chord
            else:
                break

        if active_chord and active_chord != 'N':
            notes = chord_label_to_notes(active_chord)
            for note in notes:
                evidence[note % 12, beat_idx] = 1.0

    return evidence


# ── Build chord progression ───────────────────────────────────────
def build_chord_progression(bass_notes, harmonic_chroma, chordino_evidence,
                            beat_times, key):
    if not bass_notes:
        return [], []

    # ── Step 1: Group consecutive beats by bass root ───────────
    groups = []
    current_root = bass_notes[0]
    current_start = 0

    for i in range(1, len(bass_notes)):
        note = bass_notes[i]
        if note is not None and note != current_root:
            groups.append((current_root, current_start, i - 1))
            current_root = note
            current_start = i
    groups.append((current_root, current_start, len(bass_notes) - 1))

    # ── Step 2: Remove ghost groups (single-beat blips) ────────
    # If a group is 1 beat and the groups before and after have the same root,
    # ── Step 2: Calculate common roots (needed for ghost detection) ──
    root_beat_counts = {}
    for root, start, end in groups:
        if root:
            root_beat_counts[root] = root_beat_counts.get(root, 0) + (end - start + 1)

    total_beats = sum(root_beat_counts.values())
    common_roots = {r for r, c in root_beat_counts.items() if c >= total_beats * 0.05}

    # ── Step 3: Remove ghost groups (single-beat blips) ────────
    # A ghost is a single-beat group where:
    # a) The groups before and after have the same root (obvious misread), OR
    # b) The groups before and after both have common roots AND this root is uncommon
    #    (passing note between two real chords), OR
    # c) It's 1 beat and the next group is 2+ beats (approach/passing note before
    #    a real chord — a real chord change would hold for more than 1 beat), OR
    # d) It's 1 beat and the previous group is 2+ beats (departure note after a
    #    real chord)
    cleaned_groups = []
    for i, (root, start, end) in enumerate(groups):
        duration = end - start + 1

        if duration <= 1 and i > 0 and i < len(groups) - 1:
            prev_root = groups[i - 1][0]
            next_root = groups[i + 1][0]
            prev_duration = groups[i - 1][2] - groups[i - 1][1] + 1
            next_duration = groups[i + 1][2] - groups[i + 1][1] + 1

            # Case a: same root on both sides
            if prev_root == next_root:
                continue

            # Case b: both neighbours are common roots but this one isn't
            if prev_root in common_roots and next_root in common_roots and root not in common_roots:
                continue

            # Case c: single beat followed by a longer group — approach note
            if next_duration >= 2:
                continue

            # Case d: single beat preceded by a longer group — departure note
            if prev_duration >= 2:
                continue

        # Also check first position: single beat at start followed by longer group
        if duration <= 1 and i == 0 and len(groups) > 1:
            next_duration = groups[i + 1][2] - groups[i + 1][1] + 1
            if next_duration >= 2:
                continue

        cleaned_groups.append((root, start, end))

    # Re-merge consecutive groups with same root after ghost removal
    merged_groups = []
    for root, start, end in cleaned_groups:
        if merged_groups and merged_groups[-1][0] == root:
            prev_root, prev_start, prev_end = merged_groups[-1]
            merged_groups[-1] = (prev_root, prev_start, end)
        else:
            merged_groups.append((root, start, end))

    # ── Step 4: Detect intro artefacts ─────────────────────────
    chord_timestamps = []
    progression = []

    for group_idx, (root, start, end) in enumerate(merged_groups):
        if root is None:
            continue

        time = round(beat_times[start], 2) if start < len(beat_times) else 0.0

        # Skip intro artefacts: groups in the first 3 seconds whose root
        # doesn't appear meaningfully elsewhere in the song
        if time < 3.0 and root not in common_roots:
            continue

        chroma_frames = []
        for i in range(start, end + 1):
            if i < harmonic_chroma.shape[1]:
                chroma_frames.append(harmonic_chroma[:, i])

        avg_chroma = np.mean(chroma_frames, axis=0) if chroma_frames else np.zeros(12)

        if chordino_evidence is not None:
            chordino_frames = []
            for i in range(start, end + 1):
                if i < chordino_evidence.shape[1]:
                    chordino_frames.append(chordino_evidence[:, i])
            avg_chordino = np.mean(chordino_frames, axis=0) if chordino_frames else np.zeros(12)
        else:
            avg_chordino = np.zeros(12)

        combined = (avg_chroma * 0.7) + (avg_chordino * 0.3)

        chord_name = identify_chord_quality(root, combined, base_threshold=0.3)
        chord_name = normalise_enharmonic(chord_name, key)

        if not chord_timestamps or chord_timestamps[-1]["chord"] != chord_name:
            chord_timestamps.append({"time": time, "chord": chord_name})

        if not progression or progression[-1] != chord_name:
            progression.append(chord_name)

    # ── Step 5: Remove trailing artefacts ──────────────────────
    if len(chord_timestamps) > 2:
        last_root = chord_timestamps[-1]["chord"]
        if len(last_root) > 1 and last_root[1] in ('#', 'b'):
            last_root_name = last_root[:2]
        else:
            last_root_name = last_root[0]

        if last_root_name not in common_roots:
            chord_timestamps = chord_timestamps[:-1]
            progression = [e["chord"] for e in chord_timestamps]

    return chord_timestamps, progression


# ── Time signature detection ──────────────────────────────────────
def detect_time_signature(y, sr, tempo):
    """
    Detect whether the music is in 3-feel (3/4, 6/8) or 4-feel (4/4).
    Uses multiple heuristics:
    1. Beat strength patterns (strong beats every 3 vs 4)
    2. Tempo sanity check (if detected tempo is very high, likely doubled)
    3. Autocorrelation of onset envelope for periodicity
    
    Returns (time_sig_string, beats_per_bar, corrected_tempo)
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_val, beats = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)

    if len(beats) < 6:
        return "4/4", 4, tempo

    # Get onset strength at each beat
    beat_strengths = []
    for b in beats:
        if b < len(onset_env):
            beat_strengths.append(onset_env[b])
    
    if len(beat_strengths) < 6:
        return "4/4", 4, tempo

    beat_strengths = np.array(beat_strengths)

    # ── Method 1: Beat strength grouping ───────────────────────
    score_3 = 0
    score_4 = 0

    for i, strength in enumerate(beat_strengths):
        if i % 3 == 0:
            score_3 += strength
        else:
            score_3 -= strength * 0.2

        if i % 4 == 0:
            score_4 += strength
        elif i % 4 == 2:
            score_4 += strength * 0.5
        else:
            score_4 -= strength * 0.2

    n_groups_3 = len(beat_strengths) / 3
    n_groups_4 = len(beat_strengths) / 4

    if n_groups_3 > 0:
        score_3 /= n_groups_3
    if n_groups_4 > 0:
        score_4 /= n_groups_4

    # ── Method 2: Tempo range heuristic ────────────────────────
    # If tempo > 150, it's likely the beat tracker doubled the tempo
    # Common 6/8 songs have a "real" tempo of 60-120 but tracker reads 120-240
    tempo_suggests_double = tempo > 150

    # ── Method 3: Check if grouping in 3s produces more regular accents
    # Calculate variance of accent strengths when grouped by 3 vs 4
    if len(beat_strengths) >= 12:
        # Group by 3: take every 3rd beat's strength
        accents_3 = beat_strengths[::3]
        non_accents_3 = np.concatenate([beat_strengths[1::3], beat_strengths[2::3]])
        contrast_3 = np.mean(accents_3) - np.mean(non_accents_3) if len(non_accents_3) > 0 else 0

        # Group by 4: take every 4th beat's strength
        accents_4 = beat_strengths[::4]
        non_accents_4 = np.concatenate([beat_strengths[1::4], beat_strengths[2::4], beat_strengths[3::4]])
        contrast_4 = np.mean(accents_4) - np.mean(non_accents_4) if len(non_accents_4) > 0 else 0
    else:
        contrast_3 = 0
        contrast_4 = 0

    # ── Decision ───────────────────────────────────────────────
    three_feel_evidence = 0

    if score_3 > score_4:
        three_feel_evidence += 1
    if tempo_suggests_double:
        three_feel_evidence += 1
    if contrast_3 > contrast_4:
        three_feel_evidence += 1

    if three_feel_evidence >= 2:
        # It's a 3-feel — correct the tempo
        corrected_tempo = tempo / 2 if tempo > 150 else tempo
        if corrected_tempo > 100:
            return "6/8", 6, corrected_tempo
        else:
            return "3/4", 3, corrected_tempo
    else:
        # Check if tempo is suspiciously high even for 4/4
        if tempo > 160:
            # Might be doubled — check if half tempo makes more sense
            half = tempo / 2
            if 60 <= half <= 140:
                return "4/4", 4, half
        return "4/4", 4, tempo


# ── Half-time / double-time detection ─────────────────────────────
def detect_tempo_feel(y, sr, tempo, drum_path=None):
    """
    Detect whether the track feels half-time, normal, or double-time.
    Uses the drum stem if available (more reliable than full mix which
    picks up hi-hats and subdivisions as onsets).
    """
    if drum_path:
        try:
            y_drums, sr = librosa.load(drum_path, sr=sr, mono=True)
            onset_frames = librosa.onset.onset_detect(y=y_drums, sr=sr)
        except Exception:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    else:
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)

    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    if duration == 0:
        return "normal", tempo

    onsets_per_second = len(onset_times) / duration
    beats_per_second = tempo / 60.0

    if beats_per_second > 0:
        ratio = onsets_per_second / beats_per_second
    else:
        return "normal", tempo

    # Wider thresholds — only flag half/double time when it's very clear
    if ratio < 0.4:
        return "half-time", tempo / 2
    elif ratio > 3.5:
        return "double-time", tempo * 2
    else:
        return "normal", tempo


# ── Key centre detection ──────────────────────────────────────────
def detect_key_centre(chord_timestamps, key):
    """
    Determine the key centre — which chord the song gravitates toward
    most (weighted by duration). This may differ from the detected key.
    """
    if not chord_timestamps or len(chord_timestamps) < 2:
        return key.split()[0] if ' ' in key else key

    # Calculate duration each chord is active
    chord_durations = {}
    for i, entry in enumerate(chord_timestamps):
        chord = entry["chord"]
        if i + 1 < len(chord_timestamps):
            duration = chord_timestamps[i + 1]["time"] - entry["time"]
        else:
            duration = 2.0  # Assume last chord lasts ~2 seconds

        # Extract just the root for key centre purposes
        if len(chord) > 1 and chord[1] in ('#', 'b'):
            root = chord[:2]
        else:
            root = chord[0]

        chord_durations[root] = chord_durations.get(root, 0) + duration

    # Most prominent root by time
    if chord_durations:
        return max(chord_durations, key=chord_durations.get)
    return key.split()[0] if ' ' in key else key


# ── Harmonic rhythm (chord changes per bar) ───────────────────────
def calculate_harmonic_rhythm(chord_timestamps, tempo, beats_per_bar, duration):
    """
    Calculate how many chord changes happen per bar on average.
    """
    if not chord_timestamps or tempo <= 0:
        return 0.0

    seconds_per_bar = (60.0 / tempo) * beats_per_bar
    n_bars = duration / seconds_per_bar if seconds_per_bar > 0 else 1

    n_changes = len(chord_timestamps)
    changes_per_bar = n_changes / max(n_bars, 1)

    return round(changes_per_bar, 2)


# ── Melody extraction via RMVPE ──────────────────────────────────
def get_rmvpe():
    """Lazy-load RMVPE model on first use."""
    global rmvpe_extractor
    if rmvpe_extractor is None:
        # Auto-download weights if not present
        if not os.path.exists(RMVPE_MODEL_PATH):
            os.makedirs(os.path.dirname(RMVPE_MODEL_PATH) or "models", exist_ok=True)
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    "lj1995/VoiceConversionWebUI", "rmvpe.pt",
                    local_dir=os.path.dirname(RMVPE_MODEL_PATH) or "models",
                )
            except ImportError:
                raise FileNotFoundError(
                    f"RMVPE weights not found at {RMVPE_MODEL_PATH}. "
                    "Install huggingface-hub (`pip install huggingface-hub`) for auto-download, "
                    "or manually download from: "
                    "https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt"
                )
        rmvpe_extractor = RMVPEPitchExtractor(
            model_path=RMVPE_MODEL_PATH,
            device="cpu",
            is_half=False,
            voicing_threshold=0.03,
        )
    return rmvpe_extractor


def extract_melody(audio: np.ndarray, sr: int = 44100):
    """
    Use RMVPE to extract vocal melody directly from polyphonic audio.
    
    Key change from v7: this takes the FULL MIX (or any audio), not the vocal stem.
    RMVPE is trained to extract vocal f0 from polyphonic music directly,
    eliminating the instrument bleed that contaminated torchcrepe results.
    
    Returns detailed melody analysis matching the previous output format.
    """
    from scipy.signal import medfilt

    extractor = get_rmvpe()

    # ── Extract f0 from audio via RMVPE ───────────────────────
    f0, voiced_flag, voiced_probs = extractor.extract(
        audio, sr,
        f0_min=80.0,     # Low male voice
        f0_max=1200.0,   # High female voice / falsetto
    )

    voiced_pitches = f0[voiced_flag]

    if len(voiced_pitches) < 10:
        return {
            "melody_notes": [],
            "melody_range_low": None,
            "melody_range_high": None,
            "melody_range_semitones": 0,
            "interval_histogram": {},
            "most_common_intervals": [],
            "pentatonic_adherence": 0.0,
            "note_density": 0.0,
            "melody_contour": "Unknown",
        }

    # ── Pitch range ────────────────────────────────────────────
    min_hz = float(np.percentile(voiced_pitches, 5))
    max_hz = float(np.percentile(voiced_pitches, 95))
    range_semitones = int(12 * np.log2(max_hz / min_hz)) if min_hz > 0 else 0
    min_note = hz_to_note_name(min_hz)
    max_note = hz_to_note_name(max_hz)

    # ── Convert to MIDI and apply aggressive smoothing ──────
    raw_midi = np.array([hz_to_midi(hz) for hz in voiced_pitches])
    raw_midi = raw_midi[raw_midi != None].astype(float)

    # Wider median filter (11 frames = 110ms at 100fps)
    # This is aggressive enough to snap through chromatic slides
    # between two real notes — a slide of Eb→E→F→F#→G over 40-80ms
    # gets snapped to just the start or end pitch
    if len(raw_midi) > 11:
        smoothed_midi = medfilt(raw_midi, kernel_size=11)
    elif len(raw_midi) > 5:
        smoothed_midi = medfilt(raw_midi, kernel_size=5)
    else:
        smoothed_midi = raw_midi

    # Quantise to nearest semitone
    quantised = np.round(smoothed_midi).astype(int)

    # ── Build note sequence (group consecutive same notes) ─────
    # Track how many frames each note lasts so we can filter short ones
    note_runs = []  # list of (midi_note, frame_count)
    for note in quantised:
        if note_runs and note == note_runs[-1][0]:
            note_runs[-1] = (note, note_runs[-1][1] + 1)
        else:
            note_runs.append((int(note), 1))

    # ── Remove short notes (< 4 frames = 40ms) ────────────────
    # Real sung notes last at least 50-100ms. Anything shorter is
    # either a chromatic slide artifact or a detection glitch.
    # When removing, extend the previous note to fill the gap.
    min_frames = 4
    filtered_runs = []
    for note, count in note_runs:
        if count >= min_frames:
            filtered_runs.append((note, count))
        elif filtered_runs:
            # Extend previous note
            prev_note, prev_count = filtered_runs[-1]
            filtered_runs[-1] = (prev_note, prev_count + count)
        # else: drop leading short notes

    # ── Build melody sequence from filtered runs ───────────────
    melody_sequence = [note for note, count in filtered_runs]

    # ── Remove chromatic passing runs ──────────────────────────
    # If we see 3+ consecutive notes each 1 semitone apart in the
    # same direction, collapse them to just the final note.
    # e.g. Eb E F F# G → G  (the voice jumped to G, the rest is slide)
    if len(melody_sequence) > 3:
        cleaned = [melody_sequence[0]]
        i = 1
        while i < len(melody_sequence):
            # Look ahead for a chromatic run
            run_start = i - 1
            run_dir = melody_sequence[i] - melody_sequence[i - 1]

            if abs(run_dir) == 1:
                # We have a semitone step — check if it continues
                j = i + 1
                while j < len(melody_sequence):
                    step = melody_sequence[j] - melody_sequence[j - 1]
                    if step == run_dir:
                        j += 1
                    else:
                        break

                run_length = j - run_start
                if run_length >= 3:
                    # This is a chromatic slide — keep only the destination
                    cleaned.append(melody_sequence[j - 1])
                    i = j
                    continue

            cleaned.append(melody_sequence[i])
            i += 1

        melody_sequence = cleaned

    # ── Remove isolated single-frame notes (blips) ─────────────
    if len(melody_sequence) > 2:
        cleaned = [melody_sequence[0]]
        for i in range(1, len(melody_sequence) - 1):
            prev = melody_sequence[i - 1]
            curr = melody_sequence[i]
            nxt = melody_sequence[i + 1]
            if not (prev == nxt and prev != curr):
                cleaned.append(curr)
        cleaned.append(melody_sequence[-1])
        melody_sequence = cleaned

    # ── Interval analysis ──────────────────────────────────────
    intervals = []
    for i in range(1, len(melody_sequence)):
        interval = abs(melody_sequence[i] - melody_sequence[i - 1])
        if interval <= 12:
            intervals.append(interval)

    interval_counts = Counter(intervals)
    total_intervals = sum(interval_counts.values())

    interval_histogram = {}
    for interval, count in sorted(interval_counts.items()):
        name = INTERVAL_NAMES.get(interval, f"{interval} semitones")
        interval_histogram[name] = round(count / max(total_intervals, 1), 3)

    most_common = [
        INTERVAL_NAMES.get(iv, f"{iv} semitones")
        for iv, _ in interval_counts.most_common(3)
    ]

    # ── Pentatonic adherence ───────────────────────────────────
    if melody_sequence:
        note_classes = [n % 12 for n in melody_sequence]
        most_common_note = Counter(note_classes).most_common(1)[0][0]
        pentatonic_degrees = {
            (most_common_note + d) % 12 for d in [0, 2, 4, 7, 9]
        }
        penta_count = sum(1 for n in note_classes if n in pentatonic_degrees)
        pentatonic_adherence = round(penta_count / len(note_classes), 2)
    else:
        pentatonic_adherence = 0.0

    # ── Note density (notes per second) ────────────────────────
    duration = len(audio) / sr
    note_density = round(len(melody_sequence) / max(duration, 1), 2)

    # ── Melodic contour ────────────────────────────────────────
    if len(melody_sequence) >= 4:
        quarter = len(melody_sequence) // 4
        q1_avg = np.mean(melody_sequence[:quarter])
        q2_avg = np.mean(melody_sequence[quarter:2*quarter])
        q3_avg = np.mean(melody_sequence[2*quarter:3*quarter])
        q4_avg = np.mean(melody_sequence[3*quarter:])

        if q4_avg > q1_avg + 2:
            contour = "Ascending"
        elif q4_avg < q1_avg - 2:
            contour = "Descending"
        elif q2_avg > q1_avg and q2_avg > q4_avg:
            contour = "Arch (rise then fall)"
        elif q2_avg < q1_avg and q2_avg < q4_avg:
            contour = "Valley (fall then rise)"
        else:
            contour = "Stable"
    else:
        contour = "Unknown"

    # ── Build full melody notes list ───────────────────────────
    melody_notes = []
    for note in melody_sequence:
        try:
            note_name = librosa.midi_to_note(note)
            melody_notes.append(note_name)
        except Exception:
            pass

    # ── Build timestamped pitches for chord-relation analysis ──
    # RMVPE runs at 100fps (10ms hop), same as the old torchcrepe config
    voiced_indices = np.where(voiced_flag)[0]
    timestamped_pitches = list(zip(
        (voiced_indices / RMVPE_FRAME_RATE).tolist(),
        f0[voiced_flag].tolist(),
    ))

    return {
        "melody_notes": melody_notes,
        "melody_note_count": len(melody_notes),
        "melody_range_low": min_note,
        "melody_range_high": max_note,
        "melody_range_semitones": range_semitones,
        "interval_histogram": interval_histogram,
        "most_common_intervals": most_common,
        "pentatonic_adherence": pentatonic_adherence,
        "note_density": note_density,
        "melody_contour": contour,
        # Internal data for chord-relation analysis
        "_timestamped_pitches": timestamped_pitches,
        "_melody_sequence": melody_sequence,
    }


# ── Melody-to-chord relationship analysis ─────────────────────────
def analyse_melody_chord_relationship(melody_data, chord_timestamps, key):
    """
    For each melody note, determine its relationship to the chord
    playing underneath it: chord tone, scale tone, or chromatic tone.
    Also tracks which scale degrees the melody favours.
    """
    timestamped_pitches = melody_data.get("_timestamped_pitches", [])
    if not timestamped_pitches or not chord_timestamps:
        return {
            "chord_tone_pct": 0,
            "scale_tone_pct": 0,
            "chromatic_tone_pct": 0,
            "favourite_chord_tones": [],
            "favourite_scale_degrees": [],
            "tension_profile": "Unknown",
        }

    # Determine key root for scale degree calculation
    key_parts = key.split()
    key_root = key_parts[0] if key_parts else "C"
    key_mode = key_parts[1] if len(key_parts) > 1 else "Major"
    key_root_num = NOTE_TO_NUM.get(key_root, 0)

    # Scale degrees for major/minor
    if key_mode == "Minor":
        scale_intervals = {0, 2, 3, 5, 7, 8, 10}  # Natural minor
    else:
        scale_intervals = {0, 2, 4, 5, 7, 9, 11}  # Major

    scale_notes = {(key_root_num + iv) % 12 for iv in scale_intervals}

    # Scale degree names
    if key_mode == "Minor":
        degree_names = {0: '1', 2: '2', 3: 'b3', 5: '4', 7: '5', 8: 'b6', 10: 'b7'}
    else:
        degree_names = {0: '1', 2: '2', 4: '3', 5: '4', 7: '5', 9: '6', 11: '7'}

    chord_tone_count = 0
    scale_tone_count = 0
    chromatic_count = 0
    chord_tone_types = []  # 'root', '3rd', '5th', '7th' etc
    scale_degree_hits = []
    total_notes = 0

    for timestamp, hz in timestamped_pitches:
        if hz <= 0:
            continue

        midi = int(round(12 * np.log2(hz / 440.0) + 69))
        note_class = midi % 12

        # Find which chord is active at this timestamp
        active_chord = None
        for i, ct in enumerate(chord_timestamps):
            if ct["time"] <= timestamp:
                active_chord = ct["chord"]
            else:
                break

        if not active_chord:
            continue

        total_notes += 1

        # Parse chord root
        if len(active_chord) > 1 and active_chord[1] in ('#', 'b'):
            chord_root_str = active_chord[:2]
            chord_quality = active_chord[2:]
        else:
            chord_root_str = active_chord[0]
            chord_quality = active_chord[1:]

        chord_root = NOTE_TO_NUM.get(chord_root_str, 0)

        # Build chord tone set
        chord_tones = {chord_root}  # root
        quality_lower = chord_quality.lower()

        if 'm' in quality_lower and 'maj' not in quality_lower:
            chord_tones.add((chord_root + 3) % 12)  # minor 3rd
        else:
            chord_tones.add((chord_root + 4) % 12)  # major 3rd

        if 'dim' in quality_lower:
            chord_tones.add((chord_root + 6) % 12)
        elif 'aug' in quality_lower:
            chord_tones.add((chord_root + 8) % 12)
        else:
            chord_tones.add((chord_root + 7) % 12)  # 5th

        if 'maj7' in quality_lower or 'maj9' in quality_lower:
            chord_tones.add((chord_root + 11) % 12)
        elif '7' in quality_lower or '9' in quality_lower:
            chord_tones.add((chord_root + 10) % 12)

        if '9' in quality_lower:
            chord_tones.add((chord_root + 2) % 12)

        # Classify the melody note
        interval_from_root = (note_class - chord_root) % 12

        if note_class in chord_tones:
            chord_tone_count += 1
            # What kind of chord tone?
            tone_labels = {0: 'root', 3: 'b3', 4: '3', 7: '5', 10: 'b7', 11: '7', 2: '9'}
            label = tone_labels.get(interval_from_root, f'interval_{interval_from_root}')
            chord_tone_types.append(label)
        elif note_class in scale_notes:
            scale_tone_count += 1
        else:
            chromatic_count += 1

        # Track scale degree
        interval_from_key = (note_class - key_root_num) % 12
        if interval_from_key in degree_names:
            scale_degree_hits.append(degree_names[interval_from_key])

    # Calculate percentages
    if total_notes > 0:
        chord_tone_pct = round(chord_tone_count / total_notes * 100, 1)
        scale_tone_pct = round(scale_tone_count / total_notes * 100, 1)
        chromatic_pct = round(chromatic_count / total_notes * 100, 1)
    else:
        chord_tone_pct = scale_tone_pct = chromatic_pct = 0

    # Most common chord tones
    ct_counts = Counter(chord_tone_types)
    favourite_chord_tones = [
        {"tone": tone, "pct": round(count / max(total_notes, 1) * 100, 1)}
        for tone, count in ct_counts.most_common(5)
    ]

    # Most common scale degrees
    sd_counts = Counter(scale_degree_hits)
    favourite_scale_degrees = [
        {"degree": deg, "pct": round(count / max(total_notes, 1) * 100, 1)}
        for deg, count in sd_counts.most_common(5)
    ]

    # Tension profile
    if chord_tone_pct > 65:
        tension_profile = "Consonant (mostly chord tones)"
    elif chord_tone_pct > 45:
        tension_profile = "Balanced (mix of chord and scale tones)"
    elif chromatic_pct > 20:
        tension_profile = "Chromatic (high tension)"
    else:
        tension_profile = "Scalar (scale-based, moderate tension)"

    return {
        "chord_tone_pct": chord_tone_pct,
        "scale_tone_pct": scale_tone_pct,
        "chromatic_tone_pct": chromatic_pct,
        "favourite_chord_tones": favourite_chord_tones,
        "favourite_scale_degrees": favourite_scale_degrees,
        "tension_profile": tension_profile,
    }


# ── Voicing width estimation ──────────────────────────────────────
def estimate_voicing_character(other_path: str, chord_timestamps, sr=44100):
    """
    Estimate whether chord voicings are close (clustered) or open (spread)
    by analysing the spectral spread of the harmonic stem during each chord.
    Also detects likely inversions by comparing bass note to chord root.
    """
    y_other, sr_loaded = librosa.load(other_path, sr=sr, mono=True)

    voicing_data = []
    inversion_count = 0
    total_chords = 0

    for i, ct in enumerate(chord_timestamps):
        start_time = ct["time"]
        if i + 1 < len(chord_timestamps):
            end_time = chord_timestamps[i + 1]["time"]
        else:
            end_time = start_time + 2.0

        # Extract the audio segment for this chord
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y_other[start_sample:end_sample]

        if len(segment) < 1024:
            continue

        total_chords += 1

        # Spectral bandwidth = how spread out the frequencies are
        bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
        avg_bandwidth = float(np.mean(bandwidth))

        # Spectral flatness = how "noisy" vs "tonal" (lower = more tonal/focused)
        flatness = librosa.feature.spectral_flatness(y=segment)
        avg_flatness = float(np.mean(flatness))

        voicing_data.append({
            "chord": ct["chord"],
            "bandwidth_hz": round(avg_bandwidth, 1),
            "flatness": round(avg_flatness, 4),
        })

    # Overall voicing character
    if voicing_data:
        avg_bandwidth_all = np.mean([v["bandwidth_hz"] for v in voicing_data])

        if avg_bandwidth_all > 2500:
            voicing_label = "Open voicings (wide spread)"
        elif avg_bandwidth_all > 1500:
            voicing_label = "Mixed voicings"
        else:
            voicing_label = "Close voicings (clustered)"
    else:
        avg_bandwidth_all = 0
        voicing_label = "Unknown"

    # Detect inversions from bass note vs chord root
    inversion_chords = []
    for ct in chord_timestamps:
        chord = ct["chord"]
        # Parse chord root
        if len(chord) > 1 and chord[1] in ('#', 'b'):
            chord_root = chord[:2]
        else:
            chord_root = chord[0]

        # We don't have per-chord bass notes in chord_timestamps,
        # but if the chord name was built with a corrected root,
        # we can check if slash notation would apply
        # For now, just report the voicing width data

    return {
        "average_bandwidth_hz": round(avg_bandwidth_all, 1),
        "voicing_character": voicing_label,
        "per_chord_voicing": voicing_data[:12],  # Cap output
    }


# ── Roman numeral analysis ─────────────────────────────────────────
def parse_chord_root(chord_name: str):
    """Extract root string and quality from a chord name."""
    if len(chord_name) > 1 and chord_name[1] in ('#', 'b'):
        return chord_name[:2], chord_name[2:]
    return chord_name[0], chord_name[1:]


def chord_to_roman_numeral(chord_name: str, key: str) -> str:
    """Convert a chord to roman numeral notation relative to the key."""
    key_parts = key.split()
    key_root = key_parts[0] if key_parts else "C"
    key_mode = key_parts[1] if len(key_parts) > 1 else "Major"
    key_num = NOTE_TO_NUM.get(key_root, 0)

    root_str, quality = parse_chord_root(chord_name)
    chord_num = NOTE_TO_NUM.get(root_str, 0)
    interval = (chord_num - key_num) % 12

    # Roman numerals for each semitone distance
    roman_map = {
        0: 'I', 1: 'bII', 2: 'II', 3: 'bIII', 4: 'III', 5: 'IV',
        6: '#IV', 7: 'V', 8: 'bVI', 9: 'VI', 10: 'bVII', 11: 'VII',
    }

    numeral = roman_map.get(interval, '?')

    # Lowercase for minor chords
    quality_lower = quality.lower()
    is_minor = (
        quality_lower.startswith('m') and not quality_lower.startswith('maj')
    ) or (
        'm' in quality_lower 
        and 'maj' not in quality_lower 
        and 'dim' not in quality_lower
        and 'aug' not in quality_lower
    )

    if is_minor:
        numeral = numeral.lower()

    # Append quality suffix
    if 'maj9' in quality_lower:
        numeral += 'maj9'
    elif 'maj7' in quality_lower:
        numeral += 'maj7'
    elif 'm7b5' in quality_lower:
        numeral += 'm7b5'
    elif 'm9' in quality_lower:
        numeral += 'm9'
    elif 'm7' in quality_lower or (is_minor and '7' in quality_lower and 'maj' not in quality_lower):
        numeral += '7'
    elif 'm6' in quality_lower:
        numeral += '6'
    elif '7sus4' in quality_lower:
        numeral += '7sus4'
    elif 'sus4' in quality_lower:
        numeral += 'sus4'
    elif '9' in quality_lower and 'maj' not in quality_lower:
        numeral += '9'
    elif '7' in quality_lower and 'maj' not in quality_lower:
        numeral += '7'
    elif '6' in quality_lower:
        numeral += '6'
    elif 'dim' in quality_lower:
        numeral += 'dim'
    elif 'aug' in quality_lower:
        numeral += 'aug'
    elif '5' in quality_lower:
        numeral += '5'

    return numeral


def analyse_harmony_advanced(chord_timestamps, chords, key):
    """
    Advanced harmonic analysis: roman numerals, chord movements,
    cadences, chromatic borrowing.
    """
    if not chords or chords == ["Could not detect"]:
        return {
            "roman_numerals": [],
            "progression_as_numerals": "",
            "chord_movements": [],
            "most_common_movements": [],
            "cadences_detected": [],
            "chromatic_borrowing_pct": 0,
            "diatonic_chords": [],
            "borrowed_chords": [],
        }

    key_parts = key.split()
    key_root = key_parts[0] if key_parts else "C"
    key_mode = key_parts[1] if len(key_parts) > 1 else "Major"
    key_num = NOTE_TO_NUM.get(key_root, 0)

    # Diatonic intervals for major and minor keys
    if key_mode == "Minor":
        diatonic_intervals = {0, 2, 3, 5, 7, 8, 10}
    else:
        diatonic_intervals = {0, 2, 4, 5, 7, 9, 11}

    diatonic_roots = {(key_num + iv) % 12 for iv in diatonic_intervals}

    # ── Roman numeral conversion ───────────────────────────────
    roman_numerals = []
    for ct in chord_timestamps:
        chord = ct["chord"]
        numeral = chord_to_roman_numeral(chord, key)
        roman_numerals.append({
            "time": ct["time"],
            "chord": chord,
            "numeral": numeral,
        })

    # Progression as a string
    unique_numerals = []
    for rn in roman_numerals:
        if not unique_numerals or unique_numerals[-1] != rn["numeral"]:
            unique_numerals.append(rn["numeral"])
    progression_str = " → ".join(unique_numerals)

    # ── Chord-to-chord movements ───────────────────────────────
    movements = []
    movement_counts = Counter()

    for i in range(1, len(roman_numerals)):
        prev = roman_numerals[i - 1]["numeral"]
        curr = roman_numerals[i]["numeral"]
        if prev != curr:
            movement = f"{prev} → {curr}"
            movements.append(movement)
            movement_counts[movement] += 1

    most_common_movements = [
        {"movement": mv, "count": count, "pct": round(count / max(len(movements), 1) * 100, 1)}
        for mv, count in movement_counts.most_common(6)
    ]

    # ── Root movement intervals ────────────────────────────────
    root_intervals = Counter()
    for i in range(1, len(chord_timestamps)):
        prev_root_str, _ = parse_chord_root(chord_timestamps[i - 1]["chord"])
        curr_root_str, _ = parse_chord_root(chord_timestamps[i]["chord"])
        prev_num = NOTE_TO_NUM.get(prev_root_str, 0)
        curr_num = NOTE_TO_NUM.get(curr_root_str, 0)

        if prev_root_str == curr_root_str:
            continue

        interval_up = (curr_num - prev_num) % 12
        interval_down = (prev_num - curr_num) % 12

        # Use whichever direction is shorter
        if interval_up <= interval_down:
            label = INTERVAL_NAMES.get(interval_up, f"{interval_up}") + " up"
        else:
            label = INTERVAL_NAMES.get(interval_down, f"{interval_down}") + " down"

        root_intervals[label] += 1

    total_root_movements = sum(root_intervals.values())
    root_movement_profile = [
        {"interval": iv, "pct": round(count / max(total_root_movements, 1) * 100, 1)}
        for iv, count in root_intervals.most_common(5)
    ]

    # ── Cadence detection ──────────────────────────────────────
    # Look for common 2-3 chord patterns at phrase boundaries
    # A phrase boundary is approximated by: before a long chord, or before a repeat
    cadences = Counter()
    numeral_list = [rn["numeral"] for rn in roman_numerals]

    for i in range(2, len(numeral_list)):
        # 2-chord cadence
        cadence_2 = f"{numeral_list[i-1]} → {numeral_list[i]}"
        cadences[cadence_2] += 1

        # 3-chord cadence
        if i >= 3:
            cadence_3 = f"{numeral_list[i-2]} → {numeral_list[i-1]} → {numeral_list[i]}"
            cadences[cadence_3] += 1

    cadences_detected = [
        {"cadence": cad, "count": count}
        for cad, count in cadences.most_common(5)
    ]

    # ── Chromatic borrowing ────────────────────────────────────
    diatonic_count = 0
    borrowed_count = 0
    diatonic_chords = []
    borrowed_chords = []

    seen_chords = set()
    for ct in chord_timestamps:
        chord = ct["chord"]
        if chord in seen_chords:
            continue
        seen_chords.add(chord)

        root_str, _ = parse_chord_root(chord)
        root_num = NOTE_TO_NUM.get(root_str, 0)

        if root_num in diatonic_roots:
            diatonic_count += 1
            diatonic_chords.append(chord)
        else:
            borrowed_count += 1
            borrowed_chords.append(chord)

    total_unique = diatonic_count + borrowed_count
    borrowing_pct = round(borrowed_count / max(total_unique, 1) * 100, 1)

    return {
        "roman_numerals": roman_numerals,
        "progression_as_numerals": progression_str,
        "root_movement_profile": root_movement_profile,
        "chord_movements": most_common_movements,
        "cadences_detected": cadences_detected,
        "chromatic_borrowing_pct": borrowing_pct,
        "diatonic_chords": diatonic_chords,
        "borrowed_chords": borrowed_chords,
    }


# ── Drum pattern analysis ─────────────────────────────────────────
def analyse_drum_patterns(drum_path: str, beat_times, sr=44100):
    """
    Analyse drum stem to extract kick, snare, and hi-hat patterns.
    Uses frequency band filtering to isolate each element.
    """
    from scipy.signal import butter, sosfilt

    y_drums, sr = librosa.load(drum_path, sr=sr, mono=True)

    if len(y_drums) < 1024:
        return {"error": "Drum stem too short"}

    nyquist = sr / 2

    # ── Kick detection (20-150Hz) ──────────────────────────────
    sos_kick = butter(4, [20 / nyquist, 150 / nyquist], btype='band', output='sos')
    y_kick = sosfilt(sos_kick, y_drums)
    kick_onsets = librosa.onset.onset_detect(y=y_kick, sr=sr, units='time')

    # ── Snare detection (200-1000Hz band) ────────────────────────
    # Use onset strength to filter out ghost notes and bleed — only keep strong hits
    sos_snare = butter(4, [200 / nyquist, 1000 / nyquist], btype='band', output='sos')
    y_snare = sosfilt(sos_snare, y_drums)

    snare_onset_env = librosa.onset.onset_strength(y=y_snare, sr=sr)
    snare_onset_frames = librosa.onset.onset_detect(
        y=y_snare, sr=sr, onset_envelope=snare_onset_env
    )

    # Only keep onsets above the 60th percentile of onset strength
    # This filters ghost notes and kick bleed, keeping only main snare hits
    if len(snare_onset_frames) > 0 and len(snare_onset_env) > 0:
        snare_strengths = []
        for frame in snare_onset_frames:
            if frame < len(snare_onset_env):
                snare_strengths.append(snare_onset_env[frame])
            else:
                snare_strengths.append(0)

        if snare_strengths:
            strength_threshold = np.percentile(snare_strengths, 60)
            strong_frames = [
                f for f, s in zip(snare_onset_frames, snare_strengths)
                if s >= strength_threshold
            ]
            snare_onsets = librosa.frames_to_time(np.array(strong_frames), sr=sr)
        else:
            snare_onsets = librosa.onset.onset_detect(y=y_snare, sr=sr, units='time')
    else:
        snare_onsets = np.array([])

    # ── Hi-hat detection (6000Hz+) ─────────────────────────────
    sos_hat = butter(4, 6000 / nyquist, btype='high', output='sos')
    y_hat = sosfilt(sos_hat, y_drums)
    hat_onsets = librosa.onset.onset_detect(y=y_hat, sr=sr, units='time')

    # ── Map onsets to beat positions ───────────────────────────
    def onsets_to_beat_positions(onsets, beat_times):
        """Map onset times to positions relative to beats (0.0 = on beat, 0.5 = halfway)."""
        positions = []
        for onset in onsets:
            if len(beat_times) < 2:
                continue
            # Find nearest beat
            beat_idx = np.argmin(np.abs(beat_times - onset))
            if beat_idx < len(beat_times) - 1:
                beat_start = beat_times[beat_idx]
                beat_end = beat_times[min(beat_idx + 1, len(beat_times) - 1)]
                beat_len = beat_end - beat_start
                if beat_len > 0:
                    position = (onset - beat_start) / beat_len
                    positions.append(round(position % 1.0, 2))
        return positions

    kick_positions = onsets_to_beat_positions(kick_onsets, beat_times)
    snare_positions = onsets_to_beat_positions(snare_onsets, beat_times)
    hat_positions = onsets_to_beat_positions(hat_onsets, beat_times)

    # ── Kick pattern analysis ──────────────────────────────────
    duration = librosa.get_duration(y=y_drums, sr=sr)
    kick_density = round(len(kick_onsets) / max(duration, 1), 2)

    # Determine if kick is on downbeats
    on_beat_kicks = sum(1 for p in kick_positions if p < 0.15 or p > 0.85)
    kick_on_beat_pct = round(on_beat_kicks / max(len(kick_positions), 1) * 100, 1)

    # ── Snare placement ────────────────────────────────────────
    snare_density = round(len(snare_onsets) / max(duration, 1), 2)

    # Check if snare is on a regular backbeat pattern
    # Instead of assuming bar alignment, check if snare hits are evenly spaced
    # A backbeat snare hits every 2 beats — so the interval between snare hits
    # should be roughly equal to 2 beat intervals
    if len(snare_onsets) > 2 and len(beat_times) > 1:
        snare_intervals = np.diff(snare_onsets)
        avg_beat_interval = np.median(np.diff(beat_times))
        two_beat_interval = avg_beat_interval * 2
        four_beat_interval = avg_beat_interval * 4

        # Count how many snare intervals are close to 2-beat or 4-beat spacing
        # 4-beat intervals happen when strength filtering drops one snare hit
        regular_count = 0
        for si in snare_intervals:
            # Within 40% of the expected 2-beat interval
            if abs(si - two_beat_interval) < two_beat_interval * 0.4:
                regular_count += 1
            # Or within 40% of 4-beat interval (one dropped hit)
            elif abs(si - four_beat_interval) < four_beat_interval * 0.4:
                regular_count += 1

        snare_regularity = round(regular_count / max(len(snare_intervals), 1) * 100, 1)

        # Also check: is snare density roughly 1 hit per 2 beats?
        expected_snare_density = 1.0 / two_beat_interval if two_beat_interval > 0 else 0
        density_ratio = snare_density / expected_snare_density if expected_snare_density > 0 else 0

        # Backbeat if intervals are reasonably regular AND density is in the right range
        if snare_regularity > 35 and 0.5 < density_ratio < 1.8:
            snare_style = "Backbeat (2 and 4)"
            snare_backbeat_pct = snare_regularity
        elif snare_density < 0.5:
            snare_style = "Sparse"
            snare_backbeat_pct = 0
        else:
            snare_style = "Syncopated / irregular"
            snare_backbeat_pct = snare_regularity
    else:
        snare_backbeat_pct = 0
        if snare_density < 0.5:
            snare_style = "Sparse"
        else:
            snare_style = "Unknown"

    # ── Hi-hat subdivision ─────────────────────────────────────
    hat_density = round(len(hat_onsets) / max(duration, 1), 2)
    beats_per_second = tempo / 60.0 if 'tempo' in dir() else 2.0

    # Estimate subdivision from density relative to beat rate
    if len(beat_times) > 1:
        avg_beat_interval = np.median(np.diff(beat_times))
        beats_per_sec = 1.0 / avg_beat_interval if avg_beat_interval > 0 else 2.0
        hat_ratio = hat_density / beats_per_sec if beats_per_sec > 0 else 0

        if hat_ratio > 3.5:
            hat_subdivision = "32nd notes"
        elif hat_ratio > 1.8:
            hat_subdivision = "16th notes"
        elif hat_ratio > 0.8:
            hat_subdivision = "8th notes"
        elif hat_ratio > 0.4:
            hat_subdivision = "Quarter notes"
        else:
            hat_subdivision = "Sparse / open"
    else:
        hat_subdivision = "Unknown"

    # ── Swing detection from hat timing ────────────────────────
    if len(hat_positions) > 4:
        # In swung music, off-beat hats are shifted late (position > 0.5)
        offbeat_hats = [p for p in hat_positions if 0.3 < p < 0.8]
        if offbeat_hats:
            avg_offbeat = np.mean(offbeat_hats)
            if avg_offbeat > 0.58:
                swing_feel = "Swung"
            elif avg_offbeat > 0.52:
                swing_feel = "Light swing"
            else:
                swing_feel = "Straight"
        else:
            swing_feel = "Straight"
    else:
        swing_feel = "Unknown"

    return {
        "kick_density_per_sec": kick_density,
        "kick_on_beat_pct": kick_on_beat_pct,
        "snare_density_per_sec": snare_density,
        "snare_style": snare_style,
        "snare_backbeat_pct": snare_backbeat_pct,
        "hat_density_per_sec": hat_density,
        "hat_subdivision": hat_subdivision,
        "swing_feel": swing_feel,
    }


# ── Bass rhythm analysis ──────────────────────────────────────────
def analyse_bass_rhythm(bass_path: str, beat_times, kick_onsets_times=None, sr=44100):
    """
    Analyse the rhythmic pattern of the bass line.
    """
    y_bass, sr = librosa.load(bass_path, sr=sr, mono=True)

    if len(y_bass) < 1024:
        return {"error": "Bass stem too short"}

    bass_onsets = librosa.onset.onset_detect(y=y_bass, sr=sr, units='time')
    duration = librosa.get_duration(y=y_bass, sr=sr)

    # Bass note density
    bass_density = round(len(bass_onsets) / max(duration, 1), 2)

    if bass_density > 6:
        density_label = "Very busy"
    elif bass_density > 4:
        density_label = "Busy"
    elif bass_density > 2:
        density_label = "Moderate"
    elif bass_density > 1:
        density_label = "Sparse"
    else:
        density_label = "Very sparse / sustained"

    # ── Bass-kick relationship ─────────────────────────────────
    if kick_onsets_times is not None and len(kick_onsets_times) > 0 and len(bass_onsets) > 0:
        locked_count = 0
        for bass_onset in bass_onsets:
            min_dist = min(abs(bass_onset - kick) for kick in kick_onsets_times)
            if min_dist < 0.05:  # Within 50ms = locked
                locked_count += 1

        lock_pct = round(locked_count / max(len(bass_onsets), 1) * 100, 1)

        if lock_pct > 60:
            bass_kick_relationship = "Locked (bass follows kick)"
        elif lock_pct > 30:
            bass_kick_relationship = "Semi-locked"
        else:
            bass_kick_relationship = "Independent"
    else:
        lock_pct = 0
        bass_kick_relationship = "Unknown (no kick data)"

    # ── Bass syncopation ───────────────────────────────────────
    if len(beat_times) > 1:
        offbeat_count = 0
        for onset in bass_onsets:
            min_dist = min(abs(onset - bt) for bt in beat_times) if len(beat_times) > 0 else 0
            beat_interval = np.median(np.diff(beat_times))
            if beat_interval > 0 and min_dist > beat_interval * 0.25:
                offbeat_count += 1
        bass_syncopation = round(offbeat_count / max(len(bass_onsets), 1), 2)
    else:
        bass_syncopation = 0

    if bass_syncopation > 0.4:
        bass_sync_label = "Highly syncopated"
    elif bass_syncopation > 0.2:
        bass_sync_label = "Moderately syncopated"
    else:
        bass_sync_label = "On the beat"

    return {
        "bass_note_density": bass_density,
        "bass_density_label": density_label,
        "bass_syncopation": bass_syncopation,
        "bass_syncopation_label": bass_sync_label,
        "bass_kick_lock_pct": lock_pct,
        "bass_kick_relationship": bass_kick_relationship,
    }


# ── Feature vector generation ─────────────────────────────────────
def generate_feature_vector(analysis_data: dict) -> list:
    """
    Generate a normalised feature vector from the analysis data.
    Each dimension is 0.0-1.0, suitable for vector similarity comparisons.
    """
    vector = []

    # 1. Tempo (normalised: 60bpm=0, 180bpm=1)
    tempo = analysis_data.get("tempo_bpm", 100)
    vector.append(min(max((tempo - 60) / 120, 0), 1))

    # 2. Key root (0-11 mapped to 0-1)
    key = analysis_data.get("key", "C Major")
    key_root = key.split()[0] if ' ' in key else key
    vector.append(NOTE_TO_NUM.get(key_root, 0) / 11)

    # 3. Key mode (0 = major, 1 = minor)
    vector.append(1.0 if "Minor" in key else 0.0)

    # 4. Harmonic complexity (already 0-100, normalise to 0-1)
    vector.append(analysis_data.get("harmonic_complexity", 50) / 100)

    # 5. Harmonic rhythm (chord changes per bar, normalise: 0-4)
    vector.append(min(analysis_data.get("harmonic_rhythm_per_bar", 1) / 4, 1))

    # 6. Chromatic borrowing percentage
    harmony = analysis_data.get("harmony_advanced", {})
    vector.append(min(harmony.get("chromatic_borrowing_pct", 0) / 50, 1))

    # 7-8. Melody: chord tone % and chromatic %
    mcr = analysis_data.get("melody_chord_relationship", {})
    vector.append(mcr.get("chord_tone_pct", 50) / 100)
    vector.append(mcr.get("chromatic_tone_pct", 10) / 50)

    # 9. Melody: pentatonic adherence
    melody = analysis_data.get("melody", {})
    vector.append(melody.get("pentatonic_adherence", 0.5))

    # 10. Melody: note density (normalise: 0-10 notes/sec)
    vector.append(min(melody.get("note_density", 3) / 10, 1))

    # 11. Melody: range in semitones (normalise: 0-24)
    vector.append(min(melody.get("melody_range_semitones", 12) / 24, 1))

    # 12-13. Rhythm: syncopation and onset density
    rhythm = analysis_data.get("rhythm", {})
    vector.append(rhythm.get("syncopation_score", 0.3))
    vector.append(min(rhythm.get("onset_density", 3) / 10, 1))

    # 14-15. Spectral: brightness and dynamics
    spectral = analysis_data.get("spectral", {})
    brightness = spectral.get("spectral_centroid_hz", 2000)
    vector.append(min(max((brightness - 500) / 4000, 0), 1))
    dynamic_range = spectral.get("dynamic_range_db", 15)
    vector.append(min(dynamic_range / 60, 1))

    # 16. Voicing width
    voicing = analysis_data.get("voicing", {})
    avg_bw = voicing.get("average_bandwidth_hz", 1800)
    vector.append(min(max((avg_bw - 1000) / 4000, 0), 1))

    # 17-19. Drum patterns (if available)
    drums = analysis_data.get("drum_patterns", {})
    vector.append(min(drums.get("kick_density_per_sec", 2) / 6, 1))
    vector.append(drums.get("kick_on_beat_pct", 70) / 100)
    hat_sub_map = {"Sparse / open": 0, "Quarter notes": 0.25, "8th notes": 0.5, "16th notes": 0.75, "32nd notes": 1.0}
    vector.append(hat_sub_map.get(drums.get("hat_subdivision", "8th notes"), 0.5))

    # 20. Bass rhythm
    bass_r = analysis_data.get("bass_rhythm", {})
    vector.append(min(bass_r.get("bass_note_density", 2) / 8, 1))

    # 21. Bass-kick lock
    vector.append(bass_r.get("bass_kick_lock_pct", 50) / 100)

    # 22-24. Most common intervals (weighted)
    intervals = melody.get("interval_histogram", {})
    vector.append(intervals.get("minor 2nd", 0))
    vector.append(intervals.get("major 2nd", 0))
    vector.append(intervals.get("minor 3rd", 0) + intervals.get("major 3rd", 0))

    # 25. Time signature feel (0 = 4/4, 0.5 = 3/4, 1 = 6/8)
    ts = analysis_data.get("time_signature", "4/4")
    ts_map = {"4/4": 0, "3/4": 0.5, "6/8": 1.0}
    vector.append(ts_map.get(ts, 0))

    # Ensure all values are floats
    vector = [float(v) if not isinstance(v, float) else v for v in vector]

    return vector


# ── Spectral / timbral features ───────────────────────────────────
def extract_spectral_features(y, sr):
    """Extract brightness, warmth, and dynamic range."""

    # Spectral centroid = brightness (higher = brighter)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    brightness = float(np.mean(centroid))

    # Categorise brightness
    if brightness > 3000:
        brightness_label = "Bright"
    elif brightness > 1800:
        brightness_label = "Moderate"
    else:
        brightness_label = "Warm/Dark"

    # Dynamic range (difference between loud and quiet in dB)
    rms = librosa.feature.rms(y=y)
    rms_db = librosa.amplitude_to_db(rms)
    dynamic_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))

    if dynamic_range > 20:
        dynamics_label = "Wide dynamic range"
    elif dynamic_range > 10:
        dynamics_label = "Moderate dynamics"
    else:
        dynamics_label = "Compressed"

    return {
        "spectral_centroid_hz": round(brightness, 1),
        "brightness": brightness_label,
        "dynamic_range_db": round(dynamic_range, 1),
        "dynamics": dynamics_label,
    }


# ── Rhythm analysis ───────────────────────────────────────────────
def analyse_rhythm(y, sr, beat_times):
    """
    Detailed rhythm analysis: syncopation score, onset density,
    rhythmic regularity.
    """
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    # Onset density (onsets per second)
    onset_density = len(onset_times) / max(duration, 1)

    if onset_density > 8:
        density_label = "Very dense"
    elif onset_density > 5:
        density_label = "Dense"
    elif onset_density > 3:
        density_label = "Moderate"
    else:
        density_label = "Sparse"

    # Syncopation: how many onsets fall off the beat grid
    if len(beat_times) > 1:
        offbeat_count = 0
        for ot in onset_times:
            min_dist = min(abs(ot - bt) for bt in beat_times) if len(beat_times) > 0 else 0
            beat_interval = np.median(np.diff(beat_times)) if len(beat_times) > 1 else 0.5
            # If onset is more than 25% of a beat away from any beat, it's syncopated
            if beat_interval > 0 and min_dist > beat_interval * 0.25:
                offbeat_count += 1

        syncopation_score = round(offbeat_count / max(len(onset_times), 1), 2)
    else:
        syncopation_score = 0.0

    if syncopation_score > 0.4:
        sync_label = "High syncopation"
    elif syncopation_score > 0.2:
        sync_label = "Moderate syncopation"
    else:
        sync_label = "Straight / on-grid"

    return {
        "onset_density": round(onset_density, 2),
        "density_label": density_label,
        "syncopation_score": syncopation_score,
        "syncopation_label": sync_label,
    }


# ── Models ────────────────────────────────────────────────────────
class AnalyseURLRequest(BaseModel):
    url: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 5


class SearchResult(BaseModel):
    track_name: str
    artist_name: str
    album_name: str
    preview_url: str
    artwork_url: str
    track_id: int
    genre: str
    release_date: str
    duration_ms: int


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str


class SearchAnalyseRequest(BaseModel):
    query: str = ""
    track_id: int = 0
    preview_url: str = ""


class AnalysisResult(BaseModel):
    song_id: str
    source: str

    # Key & harmony
    key: str
    key_centre: str
    tempo_bpm: float
    feel_tempo_bpm: float
    tempo_feel: str
    time_signature: str
    beats_per_bar: int

    # Chords
    chords: list[str]
    chord_timestamps: list[dict]
    harmony_advanced: dict
    harmonic_rhythm_per_bar: float
    harmonic_complexity: int
    bass_notes_detected: list[str]
    bass_source: str

    # Melody
    melody: dict
    melody_chord_relationship: dict

    # Voicing
    voicing: dict

    # Drums & bass rhythm
    drum_patterns: dict
    bass_rhythm: dict

    # Rhythm
    rhythm: dict

    # Spectral / timbral
    spectral: dict

    # Feature vector
    feature_vector: list[float]

    # Meta
    separation_used: bool
    demucs_debug: str
    notes: str


# ── Helpers ───────────────────────────────────────────────────────
def download_audio(url: str, output_path: str) -> str:
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", output_path,
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise HTTPException(status_code=400, detail=f"Download failed: {result.stderr[:300]}")

    for candidate in [output_path, output_path + ".wav"]:
        if os.path.exists(candidate):
            return candidate
    raise HTTPException(status_code=500, detail="Download succeeded but file not found")


def to_wav(input_path: str, output_path: str) -> str:
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "44100", "-f", "wav", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail="ffmpeg conversion failed")
    return output_path


# ── Core analysis ─────────────────────────────────────────────────
def analyse_audio(wav_path: str, song_id: str) -> dict:
    y, sr = librosa.load(wav_path, sr=44100, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # ── Key detection ──────────────────────────────────────────
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    key_names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    best_corr = -1
    best_key = "C Major"
    for i in range(12):
        rolled = np.roll(chroma_mean, -i)
        maj_corr = np.corrcoef(rolled, major_profile)[0, 1]
        min_corr = np.corrcoef(rolled, minor_profile)[0, 1]
        if maj_corr > best_corr:
            best_corr = maj_corr
            best_key = f"{key_names[i]} Major"
        if min_corr > best_corr:
            best_corr = min_corr
            best_key = f"{key_names[i]} Minor"

    # ── Tempo & time signature ─────────────────────────────────
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    original_tempo = tempo
    time_sig, beats_per_bar, tempo = detect_time_signature(y, sr, tempo)

    # If the tempo was corrected (halved), thin out the beat grid to match
    # Take every Nth beat where N = ratio of original to corrected tempo
    if original_tempo > tempo * 1.3:  # Tempo was significantly reduced
        tempo_ratio = round(original_tempo / tempo)
        if tempo_ratio >= 2:
            # Take every Nth beat frame to match the corrected tempo
            beat_frames = beat_frames[::tempo_ratio]

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # ── Stem separation ────────────────────────────────────────
    stem_dir = str(WORK_DIR / f"{song_id}_stems")
    separation_used = False
    demucs_error = ""
    bass_notes = []
    chord_timestamps = []
    chords = []
    unique_chords = []
    melody_data = {}
    stems = {}

    try:
        stems = separate_full_stems(wav_path, stem_dir)
        separation_used = True

        # Tempo feel (use drum stem if available)
        drum_path = stems.get("drums", None)
        tempo_feel, feel_tempo = detect_tempo_feel(y, sr, tempo, drum_path)

        # Bass roots — check energy first, fall back to low-pass if no bass instrument
        bass_source = "none"
        if "bass" in stems:
            bass_notes, y_bass_stem = detect_bass_roots(stems["bass"], beat_frames, sr)
            bass_energy, bass_label = check_bass_energy(y_bass_stem, sr)

            if bass_label in ("strong", "moderate"):
                bass_source = "bass_stem"
            elif bass_label == "weak" and "other" in stems:
                # Weak bass stem — blend bass stem with low-pass of other stem
                lowpass_notes = detect_bass_roots_lowpass(stems["other"], beat_frames, sr)
                # Use low-pass where bass stem has no clear note
                for i in range(len(bass_notes)):
                    if i < len(lowpass_notes) and bass_notes[i] is None:
                        bass_notes[i] = lowpass_notes[i]
                bass_source = "blended"
            else:
                # Empty bass stem — use low-pass entirely
                if "other" in stems:
                    bass_notes = detect_bass_roots_lowpass(stems["other"], beat_frames, sr)
                    bass_source = "lowpass"

        elif "other" in stems:
            # No bass stem at all — use low-pass
            bass_notes = detect_bass_roots_lowpass(stems["other"], beat_frames, sr)
            bass_source = "lowpass"

        # Harmonic chroma
        harmonic_chroma = None
        if "other" in stems:
            harmonic_chroma = get_harmonic_chroma(stems["other"], beat_frames, sr)

        # Chordino evidence
        chordino_evidence = None
        if "other" in stems:
            chordino_evidence = get_chordino_note_evidence(stems["other"], beat_times)

        # Build chords
        if bass_notes and harmonic_chroma is not None:
            chord_timestamps, chords = build_chord_progression(
                bass_notes, harmonic_chroma, chordino_evidence,
                beat_times, best_key
            )
            unique_chords = list(dict.fromkeys(chords))

        if not chords:
            chords = ["Could not detect"]

        # ── Melody extraction (RMVPE on vocal stem) ─────────────
        # RMVPE's pitch tracking on the Demucs-isolated vocal stem
        # gives cleaner results than either torchcrepe on the vocal stem
        # or RMVPE on the full mix.
        if "vocals" in stems:
            try:
                y_vocal, sr_vocal = librosa.load(stems["vocals"], sr=sr, mono=True)
                melody_data = extract_melody(y_vocal, sr)
            except Exception as e:
                melody_data = {"error": str(e)[:200]}
        else:
            # No vocal stem — fall back to full mix
            try:
                melody_data = extract_melody(y, sr)
            except Exception as e:
                melody_data = {"error": str(e)[:200]}

        # ── Melody-chord relationship ──────────────────────────
        if "error" not in melody_data and chord_timestamps:
            melody_chord_data = analyse_melody_chord_relationship(
                melody_data, chord_timestamps, best_key
            )
        else:
            melody_chord_data = {"error": "Melody or chords not available"}

        # ── Voicing estimation ─────────────────────────────────
        if "other" in stems and chord_timestamps:
            try:
                voicing_data = estimate_voicing_character(
                    stems["other"], chord_timestamps, sr
                )
            except Exception as e:
                voicing_data = {"error": str(e)[:200]}
        else:
            voicing_data = {"error": "Other stem not available"}

    except Exception as e:
        separation_used = False
        demucs_error = f"Demucs failed: {str(e)[:200]}"
        tempo_feel, feel_tempo = detect_tempo_feel(y, sr, tempo)
        bass_source = "fallback_full_mix"

        # Fallback chord detection
        full_chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        if len(beat_frames) > 1:
            beat_chroma = librosa.util.sync(full_chroma, beat_frames, aggregate=np.median)
        else:
            beat_chroma = full_chroma

        for i in range(beat_chroma.shape[1]):
            frame = beat_chroma[:, i]
            if frame.sum() > 0:
                root_idx = frame.argmax()
                root = NOTE_NAMES[root_idx]
                chord_name = identify_chord_quality(root, frame, base_threshold=0.3)
                chord_name = normalise_enharmonic(chord_name, best_key)
                if not chord_timestamps or chord_timestamps[-1]["chord"] != chord_name:
                    time = round(beat_times[i], 2) if i < len(beat_times) else 0.0
                    chord_timestamps.append({"time": time, "chord": chord_name})
                if chord_name not in unique_chords:
                    unique_chords.append(chord_name)

        chords = [e["chord"] for e in chord_timestamps][:12]
        if not chords:
            chords = ["Could not detect"]

        # Melody still works without stems — RMVPE runs on full mix
        try:
            melody_data = extract_melody(y, sr)
        except Exception as e:
            melody_data = {"error": f"Melody extraction failed: {str(e)[:200]}"}

        # Melody-chord relationship (if we got both)
        if "error" not in melody_data and chord_timestamps:
            melody_chord_data = analyse_melody_chord_relationship(
                melody_data, chord_timestamps, best_key
            )
        else:
            melody_chord_data = {"error": "Melody or chords not available"}

        voicing_data = {"error": "Not available without stems"}
        stems = {}

    # ── Key centre ─────────────────────────────────────────────
    key_centre = detect_key_centre(chord_timestamps, best_key)
    key_centre = normalise_enharmonic(key_centre, best_key)

    # ── Harmonic rhythm ────────────────────────────────────────
    harmonic_rhythm = calculate_harmonic_rhythm(
        chord_timestamps, tempo, beats_per_bar, duration
    )

    # ── Harmonic complexity ────────────────────────────────────
    chord_count = len(unique_chords)
    has_extensions = any(
        any(x in c for x in ['7', '9', 'maj', 'dim', 'aug', 'sus', '6', '11', '13', 'b5', '#'])
        for c in unique_chords
    )
    melody_range = melody_data.get("melody_range_semitones", 0)
    complexity = min(100, int(
        (min(chord_count, 10) / 10) * 40 +
        (1 if has_extensions else 0) * 30 +
        (min(melody_range, 24) / 24) * 30
    ))

    # ── Spectral features ──────────────────────────────────────
    spectral = extract_spectral_features(y, sr)

    # ── Rhythm analysis ────────────────────────────────────────
    rhythm = analyse_rhythm(y, sr, beat_times)

    # ── Advanced harmony analysis (roman numerals, movements, cadences)
    harmony_advanced = analyse_harmony_advanced(chord_timestamps, chords, best_key)

    # ── Drum pattern analysis ──────────────────────────────────
    drum_patterns = {}
    kick_onset_times = None
    if separation_used and "drums" in stems:
        try:
            drum_patterns = analyse_drum_patterns(stems["drums"], beat_times, sr)
            # Get kick onsets for bass-kick relationship
            from scipy.signal import butter, sosfilt
            y_drums_temp, _ = librosa.load(stems["drums"], sr=sr, mono=True)
            nyq = sr / 2
            sos_k = butter(4, [20 / nyq, 150 / nyq], btype='band', output='sos')
            y_kick_temp = sosfilt(sos_k, y_drums_temp)
            kick_onset_times = librosa.onset.onset_detect(y=y_kick_temp, sr=sr, units='time')
        except Exception as e:
            drum_patterns = {"error": str(e)[:200]}

    # ── Bass rhythm analysis ───────────────────────────────────
    bass_rhythm = {}
    if separation_used and "bass" in stems:
        try:
            bass_rhythm = analyse_bass_rhythm(stems["bass"], beat_times, kick_onset_times, sr)
        except Exception as e:
            bass_rhythm = {"error": str(e)[:200]}

    # ── Bass notes for output ──────────────────────────────────
    bass_notes_normalised = [
        normalise_enharmonic(n, best_key) if n else "—"
        for n in bass_notes
    ]

    # ── Clean up stems before generating vector ────────────────
    if os.path.exists(stem_dir):
        shutil.rmtree(stem_dir, ignore_errors=True)

    # Remove internal melody data before returning
    clean_melody = {k: v for k, v in melody_data.items() if not k.startswith('_')}

    # ── Build the full result dict ─────────────────────────────
    result = {
        "key": best_key,
        "key_centre": key_centre,
        "tempo_bpm": round(tempo, 1),
        "feel_tempo_bpm": round(feel_tempo, 1),
        "tempo_feel": tempo_feel,
        "time_signature": time_sig,
        "beats_per_bar": beats_per_bar,
        "chords": chords[:12],
        "chord_timestamps": chord_timestamps,
        "harmony_advanced": harmony_advanced,
        "harmonic_rhythm_per_bar": harmonic_rhythm,
        "harmonic_complexity": complexity,
        "bass_notes_detected": bass_notes_normalised[:24],
        "bass_source": bass_source,
        "melody": clean_melody,
        "melody_chord_relationship": melody_chord_data,
        "voicing": voicing_data,
        "drum_patterns": drum_patterns,
        "bass_rhythm": bass_rhythm,
        "rhythm": rhythm,
        "spectral": spectral,
        "separation_used": separation_used,
        "demucs_debug": demucs_error,
    }

    # ── Generate feature vector ────────────────────────────────
    result["feature_vector"] = generate_feature_vector(result)

    return result


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "harmonic-taste-profiling-engine", "version": "8.0"}


@app.post("/analyse/url", response_model=AnalysisResult)
def analyse_url(req: AnalyseURLRequest):
    song_id = str(uuid.uuid4())[:8]
    raw_path = str(WORK_DIR / f"{song_id}_raw")
    wav_path = str(WORK_DIR / f"{song_id}.wav")

    try:
        downloaded = download_audio(req.url, raw_path)
        to_wav(downloaded, wav_path)
        results = analyse_audio(wav_path, song_id)
        return AnalysisResult(song_id=song_id, source="url", notes="Analysis complete", **results)
    finally:
        for f in WORK_DIR.glob(f"{song_id}*"):
            if f.is_file():
                f.unlink(missing_ok=True)


@app.post("/analyse/upload", response_model=AnalysisResult)
async def analyse_upload(file: UploadFile = File(...)):
    song_id = str(uuid.uuid4())[:8]
    upload_path = str(WORK_DIR / f"{song_id}_upload{Path(file.filename).suffix}")
    wav_path = str(WORK_DIR / f"{song_id}.wav")

    try:
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        to_wav(upload_path, wav_path)
        results = analyse_audio(wav_path, song_id)
        return AnalysisResult(song_id=song_id, source="upload", notes="Analysis complete", **results)
    finally:
        for f in WORK_DIR.glob(f"{song_id}*"):
            if f.is_file():
                f.unlink(missing_ok=True)


# ── iTunes Search helpers ─────────────────────────────────────────
import urllib.request
import urllib.parse
import json as json_module


def search_itunes(query: str, limit: int = 5) -> list[dict]:
    """Search iTunes for songs and return results with preview URLs."""
    encoded = urllib.parse.quote(query)
    url = f"https://itunes.apple.com/search?term={encoded}&media=music&entity=song&limit={limit}"

    req = urllib.request.Request(url, headers={"User-Agent": "HarmonicEngine/1.0"})
    with urllib.request.urlopen(req, timeout=10) as response:
        data = json_module.loads(response.read().decode())

    results = []
    for item in data.get("results", []):
        preview_url = item.get("previewUrl", "")
        if not preview_url:
            continue

        results.append({
            "track_name": item.get("trackName", ""),
            "artist_name": item.get("artistName", ""),
            "album_name": item.get("collectionName", ""),
            "preview_url": preview_url,
            "artwork_url": item.get("artworkUrl100", "").replace("100x100", "600x600"),
            "track_id": item.get("trackId", 0),
            "genre": item.get("primaryGenreName", ""),
            "release_date": item.get("releaseDate", ""),
            "duration_ms": item.get("trackTimeMillis", 0),
        })

    return results


def download_preview(preview_url: str, output_path: str) -> str:
    """Download an iTunes preview clip."""
    req = urllib.request.Request(preview_url, headers={"User-Agent": "HarmonicEngine/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        with open(output_path, "wb") as f:
            f.write(response.read())
    return output_path


# ── iTunes Search endpoint ────────────────────────────────────────
@app.post("/search", response_model=SearchResponse)
def search_songs(req: SearchRequest):
    """Search iTunes for songs. Returns track info and preview URLs."""
    try:
        results = search_itunes(req.query, req.limit)
        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            query=req.query,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)[:200]}")


# ── Search + Analyse endpoint ─────────────────────────────────────
@app.post("/analyse/search", response_model=AnalysisResult)
def analyse_search(req: SearchAnalyseRequest):
    """
    Search for a song, download the 30s preview, and analyse it.
    Can provide either:
    - query: searches iTunes, picks first result
    - track_id: looks up a specific iTunes track
    - preview_url: directly downloads and analyses a preview URL
    """
    song_id = str(uuid.uuid4())[:8]
    preview_path = str(WORK_DIR / f"{song_id}_preview.m4a")
    wav_path = str(WORK_DIR / f"{song_id}.wav")

    try:
        # Get the preview URL
        if req.preview_url:
            preview_url = req.preview_url
            track_info = "Direct preview URL"
        elif req.track_id:
            # Look up specific track
            lookup_url = f"https://itunes.apple.com/lookup?id={req.track_id}"
            lookup_req = urllib.request.Request(lookup_url, headers={"User-Agent": "HarmonicEngine/1.0"})
            with urllib.request.urlopen(lookup_req, timeout=10) as response:
                data = json_module.loads(response.read().decode())
            results = data.get("results", [])
            if not results or "previewUrl" not in results[0]:
                raise HTTPException(status_code=404, detail="Track not found or no preview available")
            preview_url = results[0]["previewUrl"]
            track_info = f"{results[0].get('trackName', '')} - {results[0].get('artistName', '')}"
        elif req.query:
            # Search and pick first result
            results = search_itunes(req.query, limit=1)
            if not results:
                raise HTTPException(status_code=404, detail=f"No results found for '{req.query}'")
            preview_url = results[0]["preview_url"]
            track_info = f"{results[0]['track_name']} - {results[0]['artist_name']}"
        else:
            raise HTTPException(status_code=400, detail="Provide query, track_id, or preview_url")

        # Download the preview
        download_preview(preview_url, preview_path)

        # Convert to WAV
        to_wav(preview_path, wav_path)

        # Analyse
        results = analyse_audio(wav_path, song_id)

        return AnalysisResult(
            song_id=song_id,
            source="itunes_preview",
            notes=f"Analysis complete. Source: {track_info}",
            **results,
        )

    finally:
        for f in WORK_DIR.glob(f"{song_id}*"):
            if f.is_file():
                f.unlink(missing_ok=True)


# ── Spotify playlist helpers ──────────────────────────────────────
import base64
import re


def get_spotify_token() -> str:
    """Get a Spotify API access token using client credentials flow."""
    client_id = os.environ.get("SPOTIFY_CLIENT_ID", "")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail="Spotify credentials not configured")

    auth_str = f"{client_id}:{client_secret}"
    auth_b64 = base64.b64encode(auth_str.encode()).decode()

    req_data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
    req = urllib.request.Request(
        "https://accounts.spotify.com/api/token",
        data=req_data,
        headers={
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )

    with urllib.request.urlopen(req, timeout=10) as response:
        data = json_module.loads(response.read().decode())

    return data["access_token"]


def extract_playlist_id(url: str) -> str:
    """Extract playlist ID from a Spotify URL."""
    # Handles: https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M
    # Also: spotify:playlist:37i9dQZF1DXcBWIGoYBM5M
    match = re.search(r'playlist[/:]([a-zA-Z0-9]+)', url)
    if match:
        return match.group(1)
    raise HTTPException(status_code=400, detail="Could not extract playlist ID from URL")


def get_playlist_tracks(playlist_id: str, token: str) -> list[dict]:
    """Fetch track names and artists from a Spotify playlist."""
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks?fields=items(track(name,artists(name),album(name)))"

    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
    )

    with urllib.request.urlopen(req, timeout=10) as response:
        data = json_module.loads(response.read().decode())

    tracks = []
    for item in data.get("items", []):
        track = item.get("track")
        if not track:
            continue

        artist_names = [a["name"] for a in track.get("artists", [])]
        tracks.append({
            "track_name": track.get("name", ""),
            "artist_name": ", ".join(artist_names),
            "album_name": track.get("album", {}).get("name", ""),
        })

    return tracks


# ── Playlist models ───────────────────────────────────────────────
class PlaylistRequest(BaseModel):
    url: str


class PlaylistTrack(BaseModel):
    track_name: str
    artist_name: str
    album_name: str
    search_query: str
    itunes_preview_url: str
    itunes_match: str
    artwork_url: str


class PlaylistResponse(BaseModel):
    playlist_url: str
    total_tracks: int
    matched_tracks: int
    tracks: list[PlaylistTrack]


# ── Playlist endpoint (lightweight — just fetches track list + iTunes matches)
@app.post("/playlist/tracks", response_model=PlaylistResponse)
def get_playlist_tracks_endpoint(req: PlaylistRequest):
    """
    Fetch tracks from a Spotify playlist and find iTunes preview matches.
    Returns the track list with preview URLs — use /analyse/search to
    analyse each track individually.
    """
    token = get_spotify_token()
    playlist_id = extract_playlist_id(req.url)
    tracks = get_playlist_tracks(playlist_id, token)

    if not tracks:
        raise HTTPException(status_code=404, detail="No tracks found in playlist")

    results = []
    matched = 0

    for track in tracks:
        search_query = f"{track['track_name']} {track['artist_name']}"

        playlist_track = PlaylistTrack(
            track_name=track["track_name"],
            artist_name=track["artist_name"],
            album_name=track["album_name"],
            search_query=search_query,
            itunes_preview_url="",
            itunes_match="",
            artwork_url="",
        )

        try:
            itunes_results = search_itunes(search_query, limit=1)
            if itunes_results:
                playlist_track.itunes_preview_url = itunes_results[0]["preview_url"]
                playlist_track.itunes_match = f"{itunes_results[0]['track_name']} - {itunes_results[0]['artist_name']}"
                playlist_track.artwork_url = itunes_results[0]["artwork_url"]
                matched += 1
        except Exception:
            pass

        results.append(playlist_track)

    return PlaylistResponse(
        playlist_url=req.url,
        total_tracks=len(tracks),
        matched_tracks=matched,
        tracks=results,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
