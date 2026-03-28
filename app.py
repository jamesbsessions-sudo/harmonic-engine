"""
Harmonic Taste Profiling Engine — MVP Backend v3
Chordino for chord recognition + bass-focused root correction,
enharmonic normalisation, and minimum duration filtering.
"""

import os
import uuid
import tempfile
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import librosa
import numpy as np
from scipy.signal import butter, sosfilt
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

# Initialise Chordino once at startup
chordino = Chordino(roll_on=1)


# ── Enharmonic spelling table ─────────────────────────────────────
ENHARMONIC_MAP = {
    'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb',
}
FLAT_KEYS = {'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb',
             'Dm', 'Gm', 'Cm', 'Fm', 'Bbm', 'Ebm'}


def normalise_enharmonic(chord_name: str, key: str) -> str:
    """Respell sharp chord names as flats when the key uses flats."""
    key_root = key.split()[0] if ' ' in key else key
    use_flats = key_root in FLAT_KEYS or 'b' in key_root

    if not use_flats:
        return chord_name

    for sharp, flat in ENHARMONIC_MAP.items():
        if chord_name.startswith(sharp):
            return flat + chord_name[len(sharp):]

    return chord_name


# ── Bass note detection via low-pass filter ───────────────────────
def detect_bass_notes(y, sr, beat_frames):
    """
    Low-pass filter the audio to isolate bass frequencies,
    then detect the dominant pitch per beat to find bass roots.
    """
    nyquist = sr / 2
    cutoff = 300
    sos = butter(5, cutoff / nyquist, btype='low', output='sos')
    y_bass = sosfilt(sos, y)

    bass_chroma = librosa.feature.chroma_cqt(
        y=y_bass, sr=sr,
        fmin=librosa.note_to_hz('C1'),
        n_octaves=3,
    )

    if len(beat_frames) > 1:
        beat_chroma = librosa.util.sync(bass_chroma, beat_frames, aggregate=np.median)
    else:
        beat_chroma = bass_chroma

    note_names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

    bass_notes = []
    for i in range(beat_chroma.shape[1]):
        frame = beat_chroma[:, i]
        if frame.sum() > 0:
            best_note_idx = frame.argmax()
            bass_notes.append(note_names[best_note_idx])
        else:
            bass_notes.append(None)

    return bass_notes


def correct_chord_with_bass(chord_name: str, bass_note: str) -> str:
    """
    If the bass note disagrees with the chord root, and the bass note
    is a more likely root, replace the chord root while keeping the quality.
    """
    if not bass_note or chord_name == 'N':
        return chord_name

    if len(chord_name) > 1 and chord_name[1] in ('#', 'b'):
        chord_root = chord_name[:2]
        chord_quality = chord_name[2:]
    else:
        chord_root = chord_name[0]
        chord_quality = chord_name[1:]

    enharmonic_equiv = {
        'C#': 'Db', 'Db': 'C#', 'D#': 'Eb', 'Eb': 'D#',
        'F#': 'Gb', 'Gb': 'F#', 'G#': 'Ab', 'Ab': 'G#',
        'A#': 'Bb', 'Bb': 'A#',
    }

    roots_match = (
        chord_root == bass_note or
        enharmonic_equiv.get(chord_root) == bass_note
    )

    if roots_match:
        return chord_name

    note_to_num = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
    }

    chord_num = note_to_num.get(chord_root, -1)
    bass_num = note_to_num.get(bass_note, -1)

    if chord_num == -1 or bass_num == -1:
        return chord_name

    interval = (chord_num - bass_num) % 12

    # If the interval is a 3rd (3 or 4 semitones) or 5th (7 semitones),
    # the bass note is likely the true root and Chordino detected an inversion
    if interval in (3, 4, 7):
        return bass_note + chord_quality

    return chord_name


# ── Chord filtering ───────────────────────────────────────────────
def filter_chord_changes(chord_changes, min_duration=0.8):
    """
    Remove chord changes that last less than min_duration seconds.
    These are likely detection artefacts rather than real harmonic changes.
    """
    if not chord_changes:
        return chord_changes

    filtered = []
    for i, change in enumerate(chord_changes):
        if change.chord == 'N':
            continue

        if i + 1 < len(chord_changes):
            duration = chord_changes[i + 1].timestamp - change.timestamp
        else:
            duration = float('inf')

        if duration >= min_duration:
            filtered.append(change)

    return filtered


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
def analyse_audio(wav_path: str) -> dict:
    y, sr = librosa.load(wav_path, sr=44100, mono=True)

    # ── Key detection (Krumhansl-Schmuckler) ───────────────────
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

    # ── Bass note detection ────────────────────────────────────
    bass_notes = detect_bass_notes(y, sr, beat_frames)

    # ── Chord detection (Chordino) + bass correction ───────────
    try:
        chord_changes = chordino.extract(wav_path)

        # Step 1: Filter out short-lived chords
        chord_changes = filter_chord_changes(chord_changes, min_duration=0.8)

        # Step 2: Build timestamped chord list with bass correction
        chord_timestamps = []
        unique_chords = []
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        for change in chord_changes:
            chord_name = change.chord
            if chord_name == 'N':
                continue

            # Find the closest beat to this chord change
            if len(beat_times) > 0 and len(bass_notes) > 0:
                beat_idx = np.argmin(np.abs(beat_times - change.timestamp))
                if beat_idx < len(bass_notes) and bass_notes[beat_idx]:
                    chord_name = correct_chord_with_bass(chord_name, bass_notes[beat_idx])

            # Normalise enharmonic spelling
            chord_name = normalise_enharmonic(chord_name, best_key)

            chord_timestamps.append({
                "time": round(change.timestamp, 2),
                "chord": chord_name,
            })

            if chord_name not in unique_chords:
                unique_chords.append(chord_name)

        # Build progression (consecutive duplicates removed)
        progression = []
        for entry in chord_timestamps:
            if not progression or progression[-1] != entry["chord"]:
                progression.append(entry["chord"])

        chords = progression[:12] if progression else ["Could not detect"]

    except Exception as e:
        chords = [f"Detection error: {str(e)[:100]}"]
        chord_timestamps = []
        unique_chords = []

    # ── Melody ─────────────────────────────────────────────────
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
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

    # ── Rhythm ─────────────────────────────────────────────────
    if len(beat_frames) > 1:
        beat_times_r = librosa.frames_to_time(beat_frames, sr=sr)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        offbeat_count = 0
        for ot in onset_times:
            for i in range(len(beat_times_r) - 1):
                mid_point = (beat_times_r[i] + beat_times_r[i + 1]) / 2
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
        ioi = np.diff(librosa.frames_to_time(beat_frames, sr=sr))
        if len(ioi) > 1:
            ratios = ioi[:-1] / ioi[1:]
            swing = float(np.median(ratios))
            swing = round(min(max(swing, 0.5), 0.75), 2)
        else:
            swing = 0.5
    else:
        swing = 0.5

    # Normalise bass notes for output
    bass_notes_normalised = [
        normalise_enharmonic(n, best_key) if n else "—"
        for n in bass_notes
    ]

    return {
        "key": best_key,
        "tempo_bpm": round(tempo, 1),
        "time_signature": "4/4",
        "chords": chords,
        "chord_timestamps": chord_timestamps,
        "bass_notes_detected": bass_notes_normalised[:16],
        "melody_range": melody_range,
        "melody_contour": contour,
        "rhythm_syncopation": sync_label,
        "swing_ratio": swing,
        "harmonic_complexity": complexity,
    }


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "harmonic-taste-profiling-engine", "version": "3.0"}


@app.post("/analyse/url", response_model=AnalysisResult)
def analyse_url(req: AnalyseURLRequest):
    song_id = str(uuid.uuid4())[:8]
    raw_path = str(WORK_DIR / f"{song_id}_raw")
    wav_path = str(WORK_DIR / f"{song_id}.wav")

    try:
        downloaded = download_audio(req.url, raw_path)
        to_wav(downloaded, wav_path)
        results = analyse_audio(wav_path)
        return AnalysisResult(song_id=song_id, source="url", notes="Analysis complete", **results)
    finally:
        for f in WORK_DIR.glob(f"{song_id}*"):
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
        results = analyse_audio(wav_path)
        return AnalysisResult(song_id=song_id, source="upload", notes="Analysis complete", **results)
    finally:
        for f in WORK_DIR.glob(f"{song_id}*"):
            f.unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
