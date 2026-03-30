"""
Harmonic Taste Profiling Engine — MVP Backend v7
Full feature set: Demucs stems, bass+chroma+Chordino chord detection,
torchcrepe melody extraction, time signature, half-time detection,
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
import torchcrepe
from chord_extractor.extractors import Chordino

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


# ── Melody extraction via torchcrepe ──────────────────────────────
def extract_melody(vocal_path: str, sr=44100):
    """
    Use torchcrepe to extract melody from the vocal stem.
    Returns detailed melody analysis.
    """
    y_vocal, sr = librosa.load(vocal_path, sr=16000, mono=True)  # CREPE uses 16kHz
    audio_tensor = torch.from_numpy(y_vocal).unsqueeze(0).float()

    # Run CREPE pitch detection
    pitch, periodicity = torchcrepe.predict(
        audio_tensor,
        sample_rate=16000,
        hop_length=160,  # 10ms at 16kHz
        fmin=65,         # C2
        fmax=2093,       # C7
        model='tiny',    # Use tiny model for speed
        device='cpu',
        return_periodicity=True,
        batch_size=512,
    )

    pitch = pitch.squeeze().numpy()
    periodicity = periodicity.squeeze().numpy()

    # Filter by voicing confidence — only keep frames where voice is present
    voiced_mask = periodicity > 0.5
    voiced_pitches = pitch[voiced_mask]

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
    min_hz = float(np.percentile(voiced_pitches, 5))   # 5th percentile to avoid outliers
    max_hz = float(np.percentile(voiced_pitches, 95))   # 95th percentile
    range_semitones = int(12 * np.log2(max_hz / min_hz)) if min_hz > 0 else 0
    min_note = hz_to_note_name(min_hz)
    max_note = hz_to_note_name(max_hz)

    # ── Convert to MIDI notes and quantise ─────────────────────
    midi_notes = []
    for hz in voiced_pitches:
        m = hz_to_midi(hz)
        if m is not None:
            midi_notes.append(int(round(m)))

    # Remove repeated consecutive notes (held notes)
    melody_sequence = []
    for note in midi_notes:
        if not melody_sequence or note != melody_sequence[-1]:
            melody_sequence.append(note)

    # ── Interval analysis ──────────────────────────────────────
    intervals = []
    for i in range(1, len(melody_sequence)):
        interval = abs(melody_sequence[i] - melody_sequence[i - 1])
        if interval <= 12:  # Ignore octave+ jumps as likely errors
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
    # Check what % of notes fall on pentatonic scale degrees
    # relative to the most common note
    if melody_sequence:
        note_classes = [n % 12 for n in melody_sequence]
        most_common_note = Counter(note_classes).most_common(1)[0][0]

        # Major pentatonic intervals from root: 0, 2, 4, 7, 9
        pentatonic_degrees = {
            (most_common_note + d) % 12 for d in [0, 2, 4, 7, 9]
        }

        penta_count = sum(1 for n in note_classes if n in pentatonic_degrees)
        pentatonic_adherence = round(penta_count / len(note_classes), 2)
    else:
        pentatonic_adherence = 0.0

    # ── Note density (notes per second) ────────────────────────
    duration = len(y_vocal) / 16000
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

    # ── Build melody notes list ────────────────────────────────
    melody_notes = []
    step = max(1, len(melody_sequence) // 32)  # Cap at ~32 notes for output
    for i in range(0, len(melody_sequence), step):
        note = melody_sequence[i]
        note_name = librosa.midi_to_note(note)
        melody_notes.append(note_name)

    return {
        "melody_notes": melody_notes[:32],
        "melody_range_low": min_note,
        "melody_range_high": max_note,
        "melody_range_semitones": range_semitones,
        "interval_histogram": interval_histogram,
        "most_common_intervals": most_common,
        "pentatonic_adherence": pentatonic_adherence,
        "note_density": note_density,
        "melody_contour": contour,
        # Internal data for chord-relation analysis (not displayed directly)
        "_timestamped_pitches": list(zip(
            np.where(voiced_mask)[0] * (0.01),  # timestamps at 10ms hop
            voiced_pitches.tolist()
        )),
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
    harmonic_rhythm_per_bar: float
    harmonic_complexity: int
    bass_notes_detected: list[str]

    # Melody
    melody: dict
    melody_chord_relationship: dict

    # Voicing
    voicing: dict

    # Rhythm
    rhythm: dict

    # Spectral / timbral
    spectral: dict

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

        # Bass roots
        if "bass" in stems:
            bass_notes = detect_bass_roots(stems["bass"], beat_frames, sr)

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

        # ── Melody extraction (torchcrepe on vocal stem) ───────
        if "vocals" in stems:
            try:
                melody_data = extract_melody(stems["vocals"], sr)
            except Exception as e:
                melody_data = {"error": str(e)[:200]}
        else:
            melody_data = {"error": "No vocal stem available"}

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
        melody_data = {"error": "Stem separation failed, melody not available"}
        melody_chord_data = {"error": "Not available without stems"}
        voicing_data = {"error": "Not available without stems"}

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

    # ── Bass notes for output ──────────────────────────────────
    bass_notes_normalised = [
        normalise_enharmonic(n, best_key) if n else "—"
        for n in bass_notes
    ]

    # Clean up stems
    if os.path.exists(stem_dir):
        shutil.rmtree(stem_dir, ignore_errors=True)

    # Remove internal melody data before returning
    clean_melody = {k: v for k, v in melody_data.items() if not k.startswith('_')}

    return {
        "key": best_key,
        "key_centre": key_centre,
        "tempo_bpm": round(tempo, 1),
        "feel_tempo_bpm": round(feel_tempo, 1),
        "tempo_feel": tempo_feel,
        "time_signature": time_sig,
        "beats_per_bar": beats_per_bar,
        "chords": chords[:12],
        "chord_timestamps": chord_timestamps,
        "harmonic_rhythm_per_bar": harmonic_rhythm,
        "harmonic_complexity": complexity,
        "bass_notes_detected": bass_notes_normalised[:24],
        "melody": clean_melody,
        "melody_chord_relationship": melody_chord_data,
        "voicing": voicing_data,
        "rhythm": rhythm,
        "spectral": spectral,
        "separation_used": separation_used,
        "demucs_debug": demucs_error,
    }


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "harmonic-taste-profiling-engine", "version": "7.0"}


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
