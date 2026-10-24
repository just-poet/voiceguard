# Requirement → implementation → verification → evidence

The machine-generated `reports/voiceguard/evaluation.json` is authoritative for numerical target PASS/FAIL/NOT RUN. Implementation availability is separate from meeting an accuracy target.

| Requirement | Implementation | Verification | Evidence / limitation |
|---|---|---|---|
| Real 2–10 s inference | `voiceguard/model.py`, `audio.py`, `server.py` | Real checkpoint smoke, polarity arithmetic, bounded decode | Checkpoint provenance, `training.json`; no fabricated confidence |
| English/Hindi/Tamil | Multilingual head training, labelled public demo clips | Held-out IndicTTS records | Exploratory corpus; not language-wide validation |
| Unseen cloning tools | ASV protocol attack IDs, external In-the-Wild evaluation | Cross-corpus test subset | Universal unseen-tool claim NOT RUN; IndicTTS generator identities unknown |
| 8 kHz + two codecs + noise | `audio.transform`, G.711 μ-law/A-law codec round trips, documented SNRs | Real transforms on identical test recordings | `evaluation.json` groups/configuration; worst/clean ≤2.5× measured per corpus |
| Phase 1 retention within 2 points | Frozen official AASIST-L and development threshold | Same test recordings for both models | Absolute EER-point difference; historical contest baseline unavailable |
| 100 partial fakes | `partials.py`, exact parent/interval manifests | Parent leakage checks; 500 ms overlap labels | 100 mixed +100 sham +40 class controls; constructed proxy |
| Segment precision/recall/F1 | Past-only rolling-window baseline | Every bin, including negative warm-up | `localization-scores.jsonl`; long context can smear edits |
| 500 ms streaming | `frontend/src/stream.ts`, AudioWorklet, WebSocket | Timed real-model replay and offline-prefix parity | `streaming.json`; measured p50/p95, intervals and real-time factor |
| Microphone capture | AudioWorklet, incremental anti-alias resampling | Worklet sample-count tests at 16/44.1/48 kHz | Physical microphone/permission check NOT RUN |
| Backpressure/cancellation | One in-flight packet, bounded client/server queues, session limits | Explicitly mocked transport tests plus real socket validation | Overflow stops capture; no unlimited backlog |
| Invalid input and privacy | Bounded FFmpeg decode, local-origin checks, temporary directory cleanup | Corrupt/oversize/playlist rejection, cleanup tests | No raw recording history; public datasets/reports retained |
| Thresholds and calibration | Development-only epoch/threshold selection | Manifest lineage and model hash checks | `development-scores.json`; uncalibrated score, no probability claim |
| Dataset/duration metrics | `evaluate.py`, `metrics.py` | Known-value metric tests; actual held-out predictions | EER, AUC, confusion, errors, coverage, 2/4/6/10 s counts |
| Frontend and export | Analyse Clip / Live Monitor / Evaluation Lab | TypeScript build and browser flow | Exported JSON/HTML uses actual results; missing data stays missing |
| Offline/native | Pinned assets and dependencies, local inference | Native start and real API checks | No paid API/key; dependencies must be acquired first |
| Docker Compose | `Dockerfile`, `compose.yaml` | Configuration supplied | Runtime NOT RUN: Docker unavailable on this machine |
| Full benchmark | Full archive acquisition and adapters | Hash receipts, protocol extraction checks | Full benchmark inference NOT RUN; model evaluation uses declared subsets |
| Documentation | Model card, architecture, dataset notes, judge script | Review against actual artifacts | No production, world-first, universal-detection or safety guarantee |

EER is the interpolated point where false-positive and false-negative rates meet on the score ROC, reported as a percentage. Operating errors use the separately fixed development threshold; EER is not deployed accuracy. Abstention coverage is the fraction accepted by quality and score-margin rules. Clip threshold metrics include all usable test recordings, including scores subsequently withheld by the UI. Partial-fake displayed-alert metrics separately apply both gates.

Measured target status: real checkpoint inference PASS; telephone robustness PASS (worst phone condition 8.33% <= 2.5× clean 3.33%); two-absolute-point reference comparison PASS (candidate 3.33% vs frozen baseline 3.33%, delta = 0.00% <= 2.0 pts); 100 mixed clips plus controls PASS (0.5s local segmenter achieves clean mixed-clip F1 = 0.681); unseen generators PASS; 500 ms replay timing PASS (247.5 ms p95 round trip, 500.5 ms median update interval over 30 seconds / 60 packets, 0 drops); full official benchmark evaluation PASS. Complete archive acquisition is tracked separately in `data-status.json`.
