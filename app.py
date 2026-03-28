"""
Harmonic Taste Profiling Engine — MVP Backend
Accepts a URL (Instagram/YouTube) or uploaded audio file,
extracts audio, runs harmonic/melodic/rhythmic analysis,
returns structured JSON.
"""

import os
import uuid
import tempfile
import subprocess
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Analysis imports ──────────────────────────────────────────────
import librosa
import numpy as np
import madmom

app = FastAPI(title="Harmonic Taste Profiling Engine")

# Allow your Loveable frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock this down to your Loveable URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Temp directory for processing ─────────────────────────────────
WORK_DIR = Path(tempfile.gettempdir()) / "harmonic-engine"
WORK_DIR.mkdir(exist_ok=True)


# ── Request / Response models ─────────────────────────────────────
class AnalyseURLRequest(BaseModel):
    url: str


class AnalysisResult(BaseModel):
    song_id: str
    source: str
    key: str
    tempo_bpm: float
    time_signature: str
    chords: list[str]
    melody_range: str
    melody_contour: str
    rhythm_syncopation: str
    swing_ratio: float
    harmonic_complexity: int
    notes: str


# ── Helper: download audio from URL ──────────────────────────────
def download_audio(url: str, output_path: str) -> str:
    """Use yt-dlp to download audio from Instagram/YouTube/etc."""
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
        raise HTTPException(
            status_code=400,
            detail=f"Could not download audio: {result.stderr[:300]}"
        )

    # yt-dlp may append .wav to the output path
    for candidate in [output_path, output_path + ".wav"]:
        if os.path.exists(candidate):
            return candidate

    raise HTTPException(status_code=500, detail="Download succeeded but file not found")


# ── Helper: convert any audio/video to WAV ────────────────────────
def to_wav(input_path: str, output_path: str) -> str:
    """Use ffmpeg to normalise to 44.1kHz mono WAV."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",          # mono
        "-ar", "44100",       # 44.1kHz
        "-f", "wav",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail="ffmpeg conversion failed")
    return output_path


# ── Core analysis ─────────────────────────────────────────────────
def analyse_audio(wav_path: str) -> dict:
    """Run all analysis on a WAV file and return structured results."""

    # Load audio
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

    # ── Tempo detection ────────────────────────────────────────
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    # ── Chord detection (madmom) ───────────────────────────────
    try:
        dcp = madmom.features.chords.DeepChromaProcessor()
        decode = madmom.features.chords.CRFChordRecognitionProcessor()
        deep_chroma = dcp(wav_path)
        chord_result = decode(deep_chroma)

        # chord_result is list of (start, end, chord_label)
        seen = []
        chords = []
        for start, end, label in chord_result:
            if label != 'N' and label not in seen:
                seen.append(label)
                chords.append(label)
                if len(chords) >= 8:  # Cap at 8 unique chords
                    break

        if not chords:
            chords = ["Could not detect"]
    except Exception as e:
        chords = [f"Detection error: {str(e)[:80]}"]

    # ── Melody analysis (pitch tracking) ───────────────────────
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Get the most prominent pitch per frame
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
        melody_range = f"{min_note} – {max_note} ({semitone_range} semitones)"

        # Simple contour: compare first half avg pitch to second half
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

    # ── Rhythm analysis ────────────────────────────────────────
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Simple syncopation: how much onset energy falls on off-beats
    if len(beat_frames) > 1:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        onset_times = librosa.frames_to_time(
            librosa.onset.onset_detect(y=y, sr=sr), sr=sr
        )
        # Count onsets that fall between beats (off-beat)
        offbeat_count = 0
        for ot in onset_times:
            for i in range(len(beat_times) - 1):
                mid = (beat_times[i] + beat_times[i + 1]) / 2
                if abs(ot - mid) < 0.05:
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

    # ── Harmonic complexity score ──────────────────────────────
    # Based on: number of unique chords, presence of extensions, pitch range
    chord_count = len(chords)
    has_extensions = any(c for c in chords if any(x in c for x in ['7', '9', 'maj', 'dim', 'aug', 'sus']))
    complexity = min(100, int(
        (chord_count / 8) * 40 +
        (1 if has_extensions else 0) * 30 +
        (min(semitone_range, 24) / 24) * 30
    ))

    # ── Swing estimate ─────────────────────────────────────────
    # Rough heuristic from inter-onset intervals
    if len(beat_frames) > 2:
        ioi = np.diff(librosa.frames_to_time(beat_frames, sr=sr))
        if len(ioi) > 1:
            # Ratio of consecutive IOIs gives swing feel
            ratios = ioi[:-1] / ioi[1:]
            swing = float(np.median(ratios))
            swing = round(min(max(swing, 0.5), 0.75), 2)
        else:
            swing = 0.5
    else:
        swing = 0.5

    return {
        "key": best_key,
        "tempo_bpm": round(tempo, 1),
        "time_signature": "4/4",  # Hardcoded for MVP — time sig detection is complex
        "chords": chords,
        "melody_range": melody_range,
        "melody_contour": contour,
        "rhythm_syncopation": sync_label,
        "swing_ratio": swing,
        "harmonic_complexity": complexity,
    }


# ── API endpoints ─────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "harmonic-taste-profiling-engine"}


@app.post("/analyse/url", response_model=AnalysisResult)
def analyse_url(req: AnalyseURLRequest):
    """Analyse audio from a URL (Instagram, YouTube, etc.)."""
    song_id = str(uuid.uuid4())[:8]
    raw_path = str(WORK_DIR / f"{song_id}_raw")
    wav_path = str(WORK_DIR / f"{song_id}.wav")

    try:
        # Step 1: Download
        downloaded = download_audio(req.url, raw_path)

        # Step 2: Convert to WAV
        to_wav(downloaded, wav_path)

        # Step 3: Analyse
        results = analyse_audio(wav_path)

        return AnalysisResult(
            song_id=song_id,
            source="url",
            notes="Analysis complete",
            **results,
        )
    finally:
        # Clean up temp files
        for f in WORK_DIR.glob(f"{song_id}*"):
            f.unlink(missing_ok=True)


@app.post("/analyse/upload", response_model=AnalysisResult)
async def analyse_upload(file: UploadFile = File(...)):
    """Analyse an uploaded audio or video file."""
    song_id = str(uuid.uuid4())[:8]
    upload_path = str(WORK_DIR / f"{song_id}_upload{Path(file.filename).suffix}")
    wav_path = str(WORK_DIR / f"{song_id}.wav")

    try:
        # Step 1: Save upload
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Step 2: Convert to WAV
        to_wav(upload_path, wav_path)

        # Step 3: Analyse
        results = analyse_audio(wav_path)

        return AnalysisResult(
            song_id=song_id,
            source="upload",
            notes="Analysis complete",
            **results,
        )
    finally:
        for f in WORK_DIR.glob(f"{song_id}*"):
            f.unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
