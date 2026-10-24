# VoiceGuard delivery status

Application: http://127.0.0.1:8765

Launch: `START_VOICEGUARD.cmd`. Runtime, full datasets and caches: `D:/VoiceGuardData`. No paid service is required.

Training: 1,246 unique sources, 2,492 clean/augmented windows, 245 development clips; frozen AASIST-L encoder plus trained PyTorch head. Full datasets were acquired separately and were not used to imply full-corpus training.

| Requirement | Status | Evidence |
|---|---|---|
| Real checkpoint inference | PASS | Official AASIST-L plus trained PyTorch adaptation; source/checksums recorded. |
| Telephone EER at most 2.5× clean | FAIL | Includes actual libsndfile ULAW/ALAW WAV encode/decode conditions and combined phone/noise; identical held-out recordings. |
| Frozen baseline within 2 absolute points | FAIL | Absolute candidate-minus-reference EER difference; this newly frozen reference is not an unavailable historical contest baseline. |
| 100 partial fakes + sham controls | PASS | Windowed localisation; proxy limitations and all-bin F1 reported. |
| Unseen generators | NOT RUN | ASVspoof attack IDs retained. Cross-corpus results alone do not establish universal unseen-generator performance. |
| 500 ms end-to-end streaming | PASS | 30-second repeated-proxy replay: p95 round trip 345.4 ms, median update interval 500.8 ms, all 60 packets received. Separate prefix-parity test passed; microphone/cellular capture not tested. |
| Full official benchmark evaluation | NOT RUN | Only explicitly counted subsets evaluated; complete archives are separate. |

Twelve Python tests, three worklet tests, two transport interface tests, frontend build and extracted ZIP inference checks passed. Actual browser upload and report exports were checked before browser control was disabled. A final browser screenshot could not be captured because Work settings disabled local browser use. Docker and physical microphone/cellular capture were not run.

Accuracy limitations: cross-corpus In-the-Wild EER 46.38%; mixed-clip raw F1 0.216 and displayed-alert F1 0.056. This is a research prototype, not a reliable caller-safety decision system.
