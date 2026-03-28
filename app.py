"""
Harmonic Taste Profiling Engine — MVP Backend v6
Demucs stems + bass roots + chroma + Chordino note evidence for quality.
"""

import os
import uuid
import tempfile
import subprocess
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import librosa
import numpy as np
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


def normalise_enharmonic(name: str, key: str) -> str:
    key_root = key.split()[0] if ' ' in key else key
    use_flats = key_root in FLAT_KEYS or 'b' in key_root
    if not use_flats:
        return name
    for sharp, flat in ENHARMONIC_MAP.items():
        if name.startswith(sharp):
            return flat + name[len(sharp):]
    return name


# ── Extract note set from a Chordino chord label ─────────────────
def chord_label_to_notes(label: str) -> set:
    """
    Convert a Chordino chord label like 'C#:maj7' or 'Ebm' into
    a set of MIDI pitch classes (0-11).
    """
    if label == 'N' or not label:
        return set()

    # Parse root
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

    # Build note set based on quality
    notes = {root}  # Always include root

    quality_lower = quality.lower()

    # 3rd
    if 'm' in quality_lower and 'maj' not in quality_lower:
        notes.add((root + 3) % 12)  # minor 3rd
    else:
        notes.add((root + 4) % 12)  # major 3rd

    # 5th
    if 'dim' in quality_lower:
        notes.add((root + 6) % 12)  # diminished 5th
    elif 'aug' in quality_lower:
        notes.add((root + 8) % 12)  # augmented 5th
    else:
        notes.add((root + 7) % 12)  # perfect 5th

    # 7th
    if 'maj7' in quality_lower or 'maj9' in quality_lower:
        notes.add((root + 11) % 12)  # major 7th
    elif '7' in quality_lower or '9' in quality_lower:
        notes.add((root + 10) % 12)  # minor 7th

    # 9th
    if '9' in quality_lower:
        notes.add((root + 2) % 12)

    # 6th
    if '6' in quality_lower:
        notes.add((root + 9) % 12)

    # sus4
    if 'sus4' in quality_lower or 'sus' in quality_lower:
        notes.discard((root + 4) % 12)  # remove major 3rd
        notes.discard((root + 3) % 12)  # remove minor 3rd
        notes.add((root + 5) % 12)      # add perfect 4th

    return notes


def chordino_notes_to_chroma_boost(chordino_notes: set) -> np.ndarray:
    """
    Convert a set of pitch classes from Chordino into a 12-bin
    chroma-like array that can be blended with the actual chroma.
    """
    boost = np.zeros(12)
    for note in chordino_notes:
        boost[note % 12] = 1.0
    return boost


# ── Chord quality detection from combined evidence ────────────────
def identify_chord_quality(root_name: str, combined_chroma: np.ndarray,
                           base_threshold: float = 0.3) -> str:
    """
    Given a root note and a 12-bin combined chroma frame (from harmonic
    stem + Chordino note evidence), determine the chord quality.
    """
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

    # Interval checks — 9th uses a higher threshold to avoid false positives
    has_minor_3rd = has_interval(3)
    has_major_3rd = has_interval(4)
    has_perfect_4th = has_interval(5)
    has_dim_5th = has_interval(6)
    has_perfect_5th = has_interval(7)
    has_minor_6th = has_interval(8)
    has_major_6th = has_interval(9)
    has_minor_7th = has_interval(10)
    has_major_7th = has_interval(11)
    has_9th = has_interval(2, threshold=0.55)  # Higher threshold for 9th

    # ── Match chord qualities (most specific first) ────────────

    # Diminished: b3 + b5
    if has_minor_3rd and has_dim_5th and not has_perfect_5th:
        if has_minor_7th:
            return root_name + "m7b5"
        return root_name + "dim"

    # Augmented: 3 + #5
    if has_major_3rd and has_minor_6th and not has_perfect_5th:
        return root_name + "aug"

    # Sus4: 4 + 5, no 3rd
    if has_perfect_4th and not has_major_3rd and not has_minor_3rd:
        if has_minor_7th:
            return root_name + "7sus4"
        return root_name + "sus4"

    # Major 7th: 3 + 7
    if has_major_3rd and has_major_7th:
        if has_9th:
            return root_name + "maj9"
        return root_name + "maj7"

    # Dominant 7th: 3 + b7
    if has_major_3rd and has_minor_7th:
        if has_9th:
            return root_name + "9"
        return root_name + "7"

    # Minor 7th: b3 + b7
    if has_minor_3rd and has_minor_7th:
        if has_9th:
            return root_name + "m9"
        return root_name + "m7"

    # Minor maj7: b3 + 7
    if has_minor_3rd and has_major_7th:
        return root_name + "mMaj7"

    # Major 6th: 3 + 6
    if has_major_3rd and has_major_6th:
        return root_name + "6"

    # Minor 6th: b3 + 6
    if has_minor_3rd and has_major_6th:
        return root_name + "m6"

    # Plain minor: b3
    if has_minor_3rd:
        return root_name + "m"

    # Plain major: 3
    if has_major_3rd:
        return root_name + "maj"

    # Powerchord
    if has_perfect_5th:
        return root_name + "5"

    return root_name


# ── Demucs stem separation ────────────────────────────────────────
def separate_full_stems(wav_path: str, output_dir: str) -> dict:
    cmd = [
        "python", "-m", "demucs",
        "-n", "htdemucs",
        "--out", output_dir,
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise Exception(f"Demucs failed: {result.stderr[:300]}")

    song_name = Path(wav_path).stem
    stem_dir = Path(output_dir) / "htdemucs" / song_name

    stems = {}
    for stem_name in ["bass", "drums", "vocals", "other"]:
        stem_path = stem_dir / f"{stem_name}.wav"
        if stem_path.exists():
            stems[stem_name] = str(stem_path)

    if not stems:
        raise Exception(f"No stems found in {stem_dir}")
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


# ── Harmonic chroma from "other" stem ─────────────────────────────
def get_harmonic_chroma(other_path: str, beat_frames, sr=44100):
    y_other, sr = librosa.load(other_path, sr=sr, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y_other, sr=sr)

    if len(beat_frames) > 1:
        beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    else:
        beat_chroma = chroma
    return beat_chroma


# ── Chordino note evidence from "other" stem ──────────────────────
def get_chordino_note_evidence(other_path: str, beat_times):
    """
    Run Chordino on the other stem. For each beat, find which Chordino
    chord is active and extract its implied note set as a chroma boost.
    Returns a 12 x n_beats array.
    """
    try:
        chord_changes = chordino.extract(other_path)
    except Exception:
        return None

    if not chord_changes:
        return None

    n_beats = len(beat_times)
    evidence = np.zeros((12, n_beats))

    for beat_idx, bt in enumerate(beat_times):
        # Find which Chordino chord is active at this beat time
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
    """
    Combine bass roots + harmonic chroma + Chordino note evidence.
    Bass = root (locked). Chroma + Chordino notes = quality evidence.
    """
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

    # ── Step 2: For each group, combine evidence and determine quality
    chord_timestamps = []
    progression = []

    for root, start, end in groups:
        if root is None:
            continue

        # Skip very short groups (1 beat) at the start
        duration_beats = end - start + 1
        if duration_beats <= 1 and start == 0:
            continue

        # Average chroma across group
        chroma_frames = []
        for i in range(start, end + 1):
            if i < harmonic_chroma.shape[1]:
                chroma_frames.append(harmonic_chroma[:, i])

        if chroma_frames:
            avg_chroma = np.mean(chroma_frames, axis=0)
        else:
            avg_chroma = np.zeros(12)

        # Average Chordino note evidence across group
        if chordino_evidence is not None:
            chordino_frames = []
            for i in range(start, end + 1):
                if i < chordino_evidence.shape[1]:
                    chordino_frames.append(chordino_evidence[:, i])
            if chordino_frames:
                avg_chordino = np.mean(chordino_frames, axis=0)
            else:
                avg_chordino = np.zeros(12)
        else:
            avg_chordino = np.zeros(12)

        # Blend: chroma (weight 0.7) + Chordino note evidence (weight 0.3)
        combined = (avg_chroma * 0.7) + (avg_chordino * 0.3)

        # Determine chord quality from combined evidence
        chord_name = identify_chord_quality(root, combined, base_threshold=0.3)
        chord_name = normalise_enharmonic(chord_name, key)

        time = round(beat_times[start], 2) if start < len(beat_times) else 0.0

        if not chord_timestamps or chord_timestamps[-1]["chord"] != chord_name:
            chord_timestamps.append({"time": time, "chord": chord_name})

        if not progression or progression[-1] != chord_name:
            progression.append(chord_name)

    # ── Step 3: Filter out chords shorter than ~1 second ───────
    if len(chord_timestamps) > 1:
        filtered = []
        for i, entry in enumerate(chord_timestamps):
            if i + 1 < len(chord_timestamps):
                duration = chord_timestamps[i + 1]["time"] - entry["time"]
            else:
                duration = float('inf')
            if duration >= 1.0:
                filtered.append(entry)
            elif i == len(chord_timestamps) - 1:
                filtered.append(entry)

        chord_timestamps = filtered
        progression = [e["chord"] for e in chord_timestamps]

    return chord_timestamps, progression


# ── Models ────────────────────────────────────────────────────────
class AnalyseURLRequest(BaseModel):
    url: str


class AnalysisResult(BaseModel):
    song_id: str
    source: str
    key: str
    tempo_bpm: float
    time_signature: str
    chords: list[str]
    chord_timestamps: list[dict]
    bass_notes_detected: list[str]
    melody_range: str
    melody_contour: str
    rhythm_syncopation: str
    swing_ratio: float
    harmonic_complexity: int
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

    # ── Tempo ──────────────────────────────────────────────────
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # ── Stem separation ────────────────────────────────────────
    stem_dir = str(WORK_DIR / f"{song_id}_stems")
    separation_used = False
    demucs_error = ""
    bass_notes = []
    chord_timestamps = []
    chords = []
    unique_chords = []

    try:
        stems = separate_full_stems(wav_path, stem_dir)
        separation_used = True

        # Bass roots from isolated bass stem
        if "bass" in stems:
            bass_notes = detect_bass_roots(stems["bass"], beat_frames, sr)

        # Harmonic chroma from "other" stem
        harmonic_chroma = None
        if "other" in stems:
            harmonic_chroma = get_harmonic_chroma(stems["other"], beat_frames, sr)

        # Chordino note evidence from "other" stem
        chordino_evidence = None
        if "other" in stems:
            chordino_evidence = get_chordino_note_evidence(stems["other"], beat_times)

        # Build chords from all three sources
        if bass_notes and harmonic_chroma is not None:
            chord_timestamps, chords = build_chord_progression(
                bass_notes, harmonic_chroma, chordino_evidence,
                beat_times, best_key
            )
            unique_chords = list(dict.fromkeys(chords))

        if not chords:
            chords = ["Could not detect"]

    except Exception as e:
        separation_used = False
        demucs_error = f"Demucs failed: {str(e)[:200]}"

        # Fallback: full mix analysis
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

    # ── Melody ─────────────────────────────────────────────────
    try:
        if separation_used and "vocals" in stems:
            y_melody, _ = librosa.load(stems["vocals"], sr=sr, mono=True)
        else:
            y_melody = y

        pitches, magnitudes = librosa.piptrack(y=y_melody, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if pitch_values:
            min_pitch = min(pitch_values)
            max_pitch = max(pitch_values)
            min_note = librosa.hz_to_note(min_pitch)
            max_note = librosa.hz_to_note(max_pitch)
            semitone_range = int(12 * np.log2(max_pitch / min_pitch)) if min_pitch > 0 else 0
            melody_range = f"{min_note} - {max_note} ({semitone_range} semitones)"

            mid = len(pitch_values) // 2
            first_half = np.mean(pitch_values[:mid])
            second_half = np.mean(pitch_values[mid:])
            if second_half > first_half * 1.05:
                contour = "Ascending"
            elif second_half < first_half * 0.95:
                contour = "Descending"
            else:
                contour = "Arch / Stable"
        else:
            melody_range = "Could not detect"
            contour = "Unknown"
            semitone_range = 0
    except Exception:
        melody_range = "Could not detect"
        contour = "Unknown"
        semitone_range = 0

    # ── Rhythm ─────────────────────────────────────────────────
    if len(beat_frames) > 1:
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        offbeat_count = 0
        for ot in onset_times:
            for i in range(len(beat_times) - 1):
                mid_point = (beat_times[i] + beat_times[i + 1]) / 2
                if abs(ot - mid_point) < 0.05:
                    offbeat_count += 1
        syncopation = offbeat_count / max(len(onset_times), 1)
    else:
        syncopation = 0.0

    if syncopation > 0.3:
        sync_label = "High syncopation"
    elif syncopation > 0.15:
        sync_label = "Moderate syncopation"
    else:
        sync_label = "Low syncopation (straight)"

    # ── Harmonic complexity ────────────────────────────────────
    chord_count = len(unique_chords)
    has_extensions = any(
        any(x in c for x in ['7', '9', 'maj', 'dim', 'aug', 'sus', '6', '11', '13', 'b5', '#'])
        for c in unique_chords
    )
    complexity = min(100, int(
        (min(chord_count, 10) / 10) * 40 +
        (1 if has_extensions else 0) * 30 +
        (min(semitone_range, 24) / 24) * 30
    ))

    # ── Swing ──────────────────────────────────────────────────
    if len(beat_frames) > 2:
        ioi = np.diff(beat_times)
        if len(ioi) > 1:
            ratios = ioi[:-1] / ioi[1:]
            swing = float(np.median(ratios))
            swing = round(min(max(swing, 0.5), 0.75), 2)
        else:
            swing = 0.5
    else:
        swing = 0.5

    bass_notes_normalised = [
        normalise_enharmonic(n, best_key) if n else "—"
        for n in bass_notes
    ]

    # Clean up
    if os.path.exists(stem_dir):
        shutil.rmtree(stem_dir, ignore_errors=True)

    return {
        "key": best_key,
        "tempo_bpm": round(tempo, 1),
        "time_signature": "4/4",
        "chords": chords[:12],
        "chord_timestamps": chord_timestamps,
        "bass_notes_detected": bass_notes_normalised[:24],
        "melody_range": melody_range,
        "melody_contour": contour,
        "rhythm_syncopation": sync_label,
        "swing_ratio": swing,
        "harmonic_complexity": complexity,
        "separation_used": separation_used,
        "demucs_debug": demucs_error,
    }


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "harmonic-taste-profiling-engine", "version": "6.0"}


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
