# VoiceGuard upgrade checklist

- [x] Preserve CallGuard v1 checkpoint and evaluation under frozen paths.
- [x] Inspect hardware: Ryzen 7 7730U, 16 GB RAM, integrated Radeon, CPU PyTorch.
- [x] Install local FFmpeg decoder; no paid API.
- [x] Confirm D: has 229 GiB free; use D:/VoiceGuardData for large assets.
- [x] Finish full pinned ASVspoof mirror and In-the-Wild downloads, verify archive hashes, extract and match all ASV protocol records. All 121,461 extracted ASV hashes and all 31,779 In-the-Wild recording CRCs passed.
- [x] Verify AASIST-L checkpoint/polarity; train CPU-feasible candidate using training data only.
- [x] Implement bounded decoding, serving and timestamped causal streaming.
- [x] Build React/TypeScript Analyse Clip, Live Monitor and Evaluation Lab.
- [x] Add sham controls, duration/robustness/baseline evaluation implementation; execute the real evaluation.
- [x] Run real-model and end-to-end tests; record failures and missing evidence.
- [x] Deliver native/Docker setup, model card, matrix, demo and presentation outline.

All full archives go to D:. CPU training/evaluation may use declared, deterministic subsets within the official splits. Complete downloads do not imply complete benchmark evaluation.

Training finished: 1,246 unique training sources, 2,492 clean/augmented windows, 245 development clips, 53 epochs executed. Development EER 1.5748%; this is not a test result. The working ASV subset contains 800 train / 160 development / 108 evaluation records; requested balanced evaluation counts were not filled by the bounded metadata pages.

Verification so far: frontend build, 10 real-model/Python tests, 3 resampling tests and 2 explicitly mocked transport tests passed. Browser upload, JSON/HTML export and mixed replay were exercised. Concurrent evaluation overloaded live replay, correctly triggering the bounded queue stop. Isolated final streaming timing and complete evaluation metrics remain pending.

Completed evaluation: 346 usable full clips, 240 partial/control clips, real-container codec conditions, and 30-second replay (60 packets; 345 ms p95 round trip; 500.8 ms median update interval). Native D: runtime and independent extracted package verified. Twelve Python tests plus five JavaScript tests pass. Full archive acquisition is complete; Docker and physical microphone hardware remain NOT RUN.

Final data integrity: COMPLETE. Full archive SHA256 receipts, original ASV split-protocol matches, all extracted ASV file SHA256 checks and all In-the-Wild member CRC checks passed. See data-status.json.
