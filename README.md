# VoiceGuard — HYPNEXIS AI-01

A free, offline research application for 2–10 second speech authenticity analysis. React + TypeScript + Vite, FastAPI/WebSockets, official AASIST-L inference, a trained PyTorch adaptation head, and bounded FFmpeg decoding. No OpenAI key or paid service is used.

**Acoustic authenticity is separate from caller identity and scam intent.** Outputs are Likely synthetic, Likely bona fide, or Inconclusive. Neither a bona-fide verdict nor a low score makes a call safe. Microphone/replay modes do not intercept cellular calls.

## Start on this computer

Double-click `START_VOICEGUARD.cmd`, or run:

```powershell
$env:VOICEGUARD_DATA = 'D:\VoiceGuardData'
$env:VOICEGUARD_THREADS = '6'
& 'D:\VoiceGuardData\venv\Scripts\python.exe' -m uvicorn voiceguard.server:app --host 127.0.0.1 --port 8765 --ws-max-size 32012 --ws-max-queue 2
```

Open **http://127.0.0.1:8765**. Stop another server using that port first. Setup is `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup-voiceguard.ps1`. Setup creates the environment/cache on D: and builds the frontend. Python 3.11+ and Node 22.12+ are required; this workstation was tested with Python 3.14 and Node 24. CPU inference is the default.

After setup and asset acquisition, the app works offline. Weights are included; `python tools/check_voiceguard_assets.py` verifies them. `python tools/fetch_aasist_assets.py` restores missing official assets with pinned checksums. `python tools/compile_aasist.py` regenerates the optional fixed-input execution cache and checks numerical parity.

CPU serving uses the numerically verified ONNX export of the same frozen AASIST-L encoder when available, with PyTorch for the trained head. PyTorch remains the training and fallback inference path. `tools/export_aasist_onnx.py` and `tools/verify_onnx_head.py` reproduce the export and development-prefix parity checks; no weights or thresholds change. The measured ONNX replay achieved 247 ms p95 round-trip latency and a 500.5 ms median update interval on this workstation. This is a short replay timing test, not a sustained microphone or telephony guarantee.

`requirements-voiceguard-transitive.txt` records the tested native environment's exact package versions and constrains setup. It also lists optional training packages; runtime installation only selects dependencies needed by `requirements-voiceguard.txt`. This workstation's environment reused installed dependency files after RECORD-hash checks; those local wheels stay on D: and are not included in the submission ZIP.

## Screens

- **Analyse Clip:** WAV/FLAC/MP3/OGG/WebM/M4A, waveform/playback, quality checks, actual inference, window timeline, JSON/HTML export.
- **Live Monitor:** browser microphone or prerecorded replay; timestamped 500 ms packets, rolling past-only context, warm-up, latency, stale and overload states.
- **Evaluation Lab:** actual baseline/candidate artifacts, EER, ROC-AUC, operating errors, abstention coverage, localization and requirement status. Missing results display not evaluated.

Recordings are processed temporarily and deleted after decoding. History is off. Public dataset/cache files and exported reports are intentionally retained. Audio does not leave the local application for inference. Existing OneDrive synchronization is separate from this app.

## Data and training

Large datasets, caches and environment: **D:/VoiceGuardData** (override `VOICEGUARD_DATA`). Complete dataset downloads and subset model evaluation are different scopes. The reproducible working ASVspoof subset preserves original train/development/evaluation IDs and labels. In-the-Wild is evaluation-only. The existing IndicTTS subset covers English/Hindi/Tamil but lacks verified speaker/tool identities.

```powershell
python tools/fetch_public_subset.py --per-class 120
python tools/prepare_asv_viewer.py
python tools/prepare_benchmark_subsets.py --dataset in_the_wild
python tools/download_asv_mirror_full.py
python tools/download_benchmarks.py --dataset in_the_wild --extract
python tools/extract_asv_parquet.py
python -m voiceguard.datasets itw D:/VoiceGuardData/datasets/in_the_wild itw-manifest.jsonl
```

The extractor writes `D:/VoiceGuardData/datasets/asvspoof2019_la_mirror/manifest.jsonl`. If you independently download and extract the original LA archive, its adapter is `python -m voiceguard.datasets asvspoof <extracted-LA-directory> manifest.jsonl`.

The original Edinburgh LA archive transfer was slow; the pinned full Parquet mirror is checked against published SHA256 hashes and every extracted row against corpus protocol IDs/labels. Inspect the download receipts and extraction logs for completion; running the commands alone does not establish completion. Original downloads/partial blocks are retained for resume. `tools/prepare_asv_viewer.py` prepares the declared bounded subset from a pinned mirror; `tools/prepare_benchmark_subsets.py --dataset in_the_wild` prepares the evaluation subset. Downloader source URLs, hashes, counts and limitations are recorded on D:.

Train the actual adaptation head and select its epoch/threshold only on development data:

```powershell
& D:/VoiceGuardData/venv/Scripts/Activate.ps1
python -m pip install -r requirements-training.txt
python -m voiceguard.train data/indictts/manifest.jsonl D:/VoiceGuardData/subsets/asvspoof2019_la/manifest.jsonl
python -m voiceguard.partials data/indictts/manifest.jsonl --count 100
python -m voiceguard.evaluate data/indictts/manifest.jsonl D:/VoiceGuardData/subsets/asvspoof2019_la/manifest.jsonl D:/VoiceGuardData/subsets/in_the_wild/manifest.jsonl --partials D:/VoiceGuardData/partials/manifest.jsonl
python tools/enrich_voiceguard_reports.py
python tools/evaluate_reference_codecs.py
```

Training fits a 248→64→1 PyTorch head over frozen AASIST embeddings + LFCC features. It does not retrain the encoder. Training-only augmentations are class-independent. `--device cuda` supports head training with an appropriate CUDA PyTorch installation; CUDA was not available/tested here. Restart the server after replacing model assets.

The combined multilingual training protocol includes external speech and is **not an official ASVspoof competition submission**. Test recordings, crops, descendants and In-the-Wild never enter training/calibration. No universal unseen-generator claim is made.

## Evaluation interpretation

Measured candidate clean EER: **0.00% / 101 IndicTTS clips**, **7.14% / 88 ASVspoof clips**, **46.38% / 157 In-the-Wild clips**. Phone robustness and absolute reference retention fail. Clean mixed-clip F1 is 0.216 before abstention and 0.056 for displayed alerts. The 30-second repeated-proxy stream passed the measured timing check: 345 ms p95 round trip, 500.8 ms median update interval, 60/60 packets received. These are exploratory subset results, not production reliability.

The original AASIST-L checkpoint is frozen as the new Phase 1 reference; historical contest Phase 1 results were not supplied. Two points means an **absolute difference of two percentage points**, evaluated per dataset. Phone robustness, timing and retention are measured targets, not assumed passes.

AASIST uses the first 64600 samples, repeating shorter input. The adaptation additionally uses LFCC statistics over the supplied clip/window. Repetition adds no speech evidence. Duration conditions use only recordings with sufficient actual audio. The temporal detector is an explicitly windowed baseline, not a trained 500 ms segmenter. Partial proxies include 100 mixed clips, 100 sham edits and 40 full-class controls with matched processing.

See `reports/voiceguard/evaluation.json`, `raw-scores.jsonl`, `localization-scores.jsonl`, `development-scores.json`, `training.json`, and `streaming.json`. The original CallGuard v1 app/source/results remain preserved; its previous 2% clean EER is not an AASIST result.

## Verification

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_voiceguard.py -q
node --test tests/voiceguard-worklet.test.cjs
cd frontend
npm ci
npm run build
npx tsc --ignoreConfig src/stream.ts src/types.ts --target ES2022 --module ES2022 --moduleResolution bundler --lib ES2022,DOM --skipLibCheck --outDir ../tmp/stream-test
cd ..
node --test tests/voiceguard-stream-interface.test.mjs
python tools/verify_voiceguard_stream.py --port 8765 --model candidate
```

Interface mocks are explicitly identified in the transport tests. Python and timed WebSocket checks execute the real checkpoint. Physical microphone hardware/permissions remain a manual device check. Do not claim successful cellular interception.

## Docker

```powershell
$env:VOICEGUARD_DATA = 'D:/VoiceGuardData'
docker compose up --build
```

Docker is optional and was not available on this workstation; its runtime remains not run unless a verification artifact says otherwise. The container port binds to localhost and dataset mounts are read-only.

## Review package

- `docs/voiceguard/MODEL_CARD.md`: provenance, polarity, input policy, training and limitations.
- `docs/voiceguard/ARCHITECTURE.md`: data flow, queues, privacy and integration scope.
- `docs/voiceguard/REQUIREMENTS_MATRIX.md`: requirement → implementation → test → evidence.
- `docs/voiceguard/JUDGE_DEMO.md`: three-minute script and presentation outline.
- `docs/voiceguard/DATA_NOTES.md`: dataset provenance and licences.
- `docs/voiceguard/PROGRESS.md`: work checklist.

No guaranteed accuracy, novelty, hackathon victory or production readiness is claimed.
