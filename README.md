# Harmonic Taste Profiling Engine — MVP

A simple API that takes a URL (Instagram, YouTube, etc.) or uploaded audio/video file, analyses the harmonic/melodic/rhythmic content, and returns structured results.

---

## What's in the box

- `app.py` — The full backend (FastAPI)
- `requirements.txt` — Python packages needed
- `Dockerfile` — So Railway knows how to build and run it

---

## Step-by-step: Getting this running (all in browser, nothing on your laptop)

### Step 1: Create a GitHub account (if you don't have one)
Go to [github.com](https://github.com) and sign up. This is where your code lives.

### Step 2: Create a new repository
- Click the **+** button (top right) → **New repository**
- Name it `harmonic-engine`
- Keep it **Private**
- Click **Create repository**

### Step 3: Upload these files
- On your new repo page, click **"uploading an existing file"**
- Drag in all three files: `app.py`, `requirements.txt`, `Dockerfile`
- Click **Commit changes**

### Step 4: Deploy to Railway
- Go to [railway.app](https://railway.app) and sign in with your GitHub account
- Click **New Project** → **Deploy from GitHub Repo**
- Select your `harmonic-engine` repo
- Railway will detect the Dockerfile and start building automatically
- Wait for it to deploy (first build takes 3-5 minutes)
- Once deployed, Railway gives you a URL like `https://harmonic-engine-production-xxxx.up.railway.app`

### Step 5: Test it
Open your Railway URL in a browser. You should see:
```json
{"status": "ok", "service": "harmonic-taste-profiling-engine"}
```

To test the analysis, you can use the auto-generated docs:
- Go to `https://your-railway-url.up.railway.app/docs`
- Click on `/analyse/url` → **Try it out**
- Paste a YouTube or Instagram URL
- Hit **Execute**
- See the analysis results

### Step 6: Connect to Loveable
In your Loveable app, replace the dummy setTimeout with a real fetch call:
```javascript
const response = await fetch('https://your-railway-url.up.railway.app/analyse/url', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ url: inputUrl })
});
const data = await response.json();
```

---

## What the API returns

```json
{
  "song_id": "a1b2c3d4",
  "source": "url",
  "key": "Db Major",
  "tempo_bpm": 92.0,
  "time_signature": "4/4",
  "chords": ["Db:maj", "Fm", "Bbm", "Gb:maj7"],
  "melody_range": "C4 – F5 (14 semitones)",
  "melody_contour": "Arch / Stable",
  "rhythm_syncopation": "Moderate syncopation",
  "swing_ratio": 0.58,
  "harmonic_complexity": 72,
  "notes": "Analysis complete"
}
```

---

## Costs

- **GitHub**: Free
- **Railway**: Free trial, then ~$5/month for the resources this needs
- **Everything else**: Free (all libraries are MIT/BSD licensed)

---

## What's NOT included yet (future iterations)

- Demucs stem separation (adds accuracy but needs more RAM)
- MIDI extraction (Basic-Pitch)
- User profiles / taste aggregation
- Apple Music API integration
- Database storage
