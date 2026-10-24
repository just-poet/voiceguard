# CallGuard

A free, local research application for short-clip speech authenticity analysis and streaming evidence. Built for a Ryzen 7 U-series laptop with 16 GB RAM and an integrated GPU. Inference and training use the CPU; there is no paid API, account, or hosted service.

**Scope:** English, Hindi, and Tamil speech in the public demonstration dataset. Language coverage is not a claim of language-wide reliability. Scores are model evidence, not calibrated probabilities, identity verification, or proof of a scam.

## Run

Double-click **START_CALLGUARD.cmd**, or use PowerShell in this folder:

```powershell
.\start.ps1
```

Open **http://127.0.0.1:8765**. On first run the launcher creates `.venv` and installs dependencies. An internet connection is needed for dependency installation and dataset acquisition; afterward the application runs locally and offline.

If Python and the dependencies are already installed:

```powershell
python -m uvicorn callguard.server:app --host 127.0.0.1 --port 8765 --ws-max-size 32768
```

Python 3.11+ is required. See `requirements-lock.txt` for the versions verified on this computer. Do not expose this development server to the public internet.

The submission ZIP contains the app, selected model, six public speech examples, 100 annotated partial clips, and measured reports. The full 720-clip training subset can be recreated with the downloader.

## What works

- Upload WAV, FLAC, or OGG clips, 2–10 seconds and up to 8 MB.
- Live microphone input over a local WebSocket, in 500 ms chunks, with a 2-second warm-up.
- A separate 500 ms localization classifier and a 2-second context classifier.
- Three-view analysis: native audio, narrowband G.711 μ-law, and G.711 A-law.
- Signal level and clipping checks, uncertainty when views disagree or scores are near the threshold.
- Half-second timeline, selectable segments, audio playback, and JSON report export.
- Public examples in English, Hindi, and Tamil, selected by manifest order from the held-out subset, not by prediction correctness.
- Reproducible training, 100 partial-fake construction, EER, segment F1, CPU latency, language breakdowns, and channel ablation.
- Strict speaker/tool/content leakage checks for datasets that provide the necessary metadata.

This browser application does **not** intercept cellular calls. Microphone capture and a replayed audio stream are the demo integration points. Actual phone/VoIP integration is a separate engineering task.

## Reproduce the public-data experiment

```powershell
python tools/fetch_public_subset.py --per-class 120
python -m callguard.cli train data/indictts/manifest.jsonl --exploratory --classifier linear --output models/baseline-linear.json
python -m callguard.cli train data/indictts/manifest.jsonl --exploratory --classifier extra-trees --output models/candidate-trees.json
python -m callguard.cli train data/indictts/manifest.jsonl --exploratory --classifier rbf-svm --output models/candidate-svm.json
python tools/robust_select.py data/indictts/manifest.jsonl models/baseline-linear.json models/candidate-trees.json models/candidate-svm.json
python -m callguard.cli partials data/indictts/manifest.jsonl --count 100 --exploratory
python -m callguard.cli evaluate data/indictts/manifest.jsonl --partials data/partials/manifest.jsonl
```

Restart the server after replacing a model. Training writes `models/detector.json`; evaluation writes `reports/evaluation.json`.

To rebuild the PDF/ZIP after reproducing the full dataset experiment, install `requirements-report.txt`, run `python tools/verify_stream.py` while the local server is running, then run `python tools/build_submission.py`.

The bounded downloader retains 720 clips, 120 per class per language. Clips shorter than 2 seconds are excluded and longer clips are cropped to 6 seconds. A deterministic normalized-transcript hash assigns about 70%/15%/15% to train/development/test, with identical text kept together. The actual prepared split contains **520 training, 99 development, and 101 test clips**.

The dataset does not publish speaker or generator identities. `--exploratory` explicitly records that speaker and unseen-tool generalization cannot be verified. It does not relax audio-content or transcript leakage checks. Partial clips use same-language donors but speaker identity cannot be verified, so speaker changes may make localization easier. These limitations must accompany the results.

## Use a rigorous dataset

Create a JSON Lines manifest with paths relative to the manifest file:

```json
{"path":"audio/real.wav","label":0,"speaker":"speaker_01","generator":"human","language":"Hindi","split":"train"}
{"path":"audio/fake.wav","label":1,"speaker":"speaker_01","generator":"tts_tool_A","language":"Hindi","split":"train"}
```

Include both classes in each split. Use disjoint speakers in train/dev/test and hold test synthesis tools out of training. Add optional `text_group_sha256` to enforce transcript-group separation. Partial files additionally provide `fake_segments: [[start_seconds, end_seconds]]` and source audio hashes.

For strict training, omit `--exploratory`. For strict partial creation, genuine and synthetic donors must have the same known speaker ID. Paths, labels, exact source hashes, and segment validity are checked. Re-encoded or paraphrased duplicates need a separate corpus audit.

## Method

1. Decode to mono, then resample to 16 kHz.
2. Extract 88-dimensional LFCC, temporal-delta and spectral-summary features.
3. Train two lightweight classifiers (linear, Extra Trees, or a bounded RBF SVM) on clean and phone-channel/noise augmentations: a 2-second context head and a 0.5-second localization head.
4. At inference, score native, μ-law and A-law views. Apply development-selected weights and measure their spread. The selected model uses native audio for classification; codec views remain disagreement probes. Equal averaging performed worse in the development robustness comparison.
5. Classify each 500 ms bin with a development-selected threshold. Pool the top two context scores for a clip decision. Use a separate development-selected clip threshold.
6. Abstain when signal quality is poor, view disagreement exceeds 0.25, or the score is within 0.08 of its threshold. These last two values are declared heuristics, not statistically calibrated confidence bounds.

The working research hypothesis is that short-context localization plus counterfactual telephone-channel consistency provides useful evidence and a useful reason to abstain. Related research already explores compression robustness and partial-fake detection. **No claim of world-first novelty is made.** See `docs/RESEARCH.md` for the experimental contribution and prior work.

## Phase 2 protocol

| Requirement | Implementation | Limit |
|---|---|---|
| 8 kHz resampling, two codecs, noise | Narrowband resampling, G.711 μ-law/A-law, 10 dB Gaussian noise | Simulated channels, not measured network calls |
| Worst EER ≤2.5× clean | Reported automatically | A measured target, not an assumed pass |
| 100 partial fakes | WAVs plus timestamp and source-hash manifests | Demo uses unknown speaker identities |
| F1 at 0.5s | Separate local head, ≥50% synthetic overlap label rule | Warm-up counted as negative; no bins hidden |
| Update every 500 ms | Browser AudioWorklet + WebSocket | Device/browser scheduling also matters |
| Phase 1 EER within 2 points | `--phase1-eer VALUE` on the same test protocol | Original Phase 1 results were not supplied |
| Unseen cloning tools | Strict manifest validation supports tool holdout | Not verifiable with the demonstration dataset |

EER is computed on all full-clip raw scores, including abstentions. Localization reports all bins. Acceptance coverage is reported separately, so the system cannot improve EER by hiding difficult examples. An EER ratio is undefined when clean EER is zero; the absolute inequality is still evaluated.

## Verification

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
node --check web/app.js
node --check web/processor.js
node --test tests/worklet.test.cjs
```

Tests cover decoding limits, G.711 reference values, deterministic transformations, EER ties and extremes, causal windows, leakage checks, 100-file partial creation, upload failures, WebSocket timestamps, and rejected origins. Diagnostic tones test plumbing only; their speech-authenticity scores are suppressed in the interface.

## Files

```text
callguard/       audio, features, classifiers, API, training and evaluation
web/             dependency-free local browser interface
tools/           bounded public dataset acquisition and report tools
tests/           automated integration and numerical tests
data/indictts/   downloaded public subset, manifest and provenance
data/partials/  100 generated clips and segment annotations
models/          trained model artifacts
reports/         measured evaluation and submission outputs
docs/            research explanation and demonstration notes
```

The local folder is inside OneDrive. The application itself sends no audio to a cloud service, but your operating system's existing OneDrive configuration may sync files placed in this folder. Uploaded and microphone audio are processed in memory; dataset downloads and exported reports are files.

## Data credit

Demonstration speech: [SherryT997 / IndicTTS Deepfake Detection Challenge](https://huggingface.co/datasets/SherryT997/IndicTTS-Deepfake-Challenge-Data), licensed **CC BY 4.0**, revision `57347517658ae989597d8cef303cffb647ed2434`. Modifications: bounded subsampling, mono/16 kHz conversion, duration filtering/cropping, telephone transformations and annotated splices. The original dataset card and provenance accompany the download. The original test split has no usable labels; this experiment derives its splits only from the labeled public train split.
