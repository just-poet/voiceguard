# Three-minute demonstration

1. **0:00–0:25 — Claim.** “VoiceGuard examines acoustic authenticity, not caller identity or scam intent. It runs locally without a paid API.” Show the chosen checkpoint and the model card.
2. **0:25–1:00 — Analyse Clip.** Select a public held-out speech example, run actual inference, inspect the score and quality. Explain that the score is uncalibrated. Keep incorrect predictions visible.
3. **1:00–1:35 — Localisation.** Load the partial control, show its ground-truth interval and predicted windows, then compare the sham real-to-real edit. Explain long-window boundary smearing.
4. **1:35–2:10 — Live Monitor.** Replay the selected clip. Show two-second warm-up, 500 ms packets, last update and measured latency. Say explicitly that this is prerecorded replay, not intercepted telephony. Microphone requires the user's browser permission.
5. **2:10–2:45 — Evaluation Lab.** Compare original and adapted models, clean and degraded conditions, EER/ROC-AUC and false alarms. Show the pass/fail/not-run ledger. Distinguish complete downloads from subset evaluation.
6. **2:45–3:00 — Evidence.** Export JSON and HTML. Point to raw scores, source provenance and reproducible commands. State the strongest remaining failure; do not claim production readiness.

# Presentation outline
1. Problem: convincing voices cannot authenticate callers.
2. System: local React/FastAPI application, official AASIST-L, trained adaptation.
3. Data: original ASVspoof splits, evaluation-only In-the-Wild, exploratory IndicTTS; subset counts and provenance.
4. Experiments: telephone transformations, duration conditions, matched sham edits, streaming causality.
5. Results: measured baseline/candidate table and requirement ledger, including failures.
6. Limits and next steps: genuine speech in scams, speaker/tool identity gaps, real telephony and independent validation.

# Actual results to state

Candidate clean EER: 0.00% on 101 IndicTTS clips (30 English, 43 Hindi, 28 Tamil), 7.14% on 88 ASVspoof clips, and 46.38% on 157 In-the-Wild clips. Do not lead with the 0% without also showing the cross-corpus failure. Telephone robustness and absolute reference retention fail. On the 100 clean mixed clips, raw F1 is 0.216 and displayed-alert F1 is 0.056. The 30-second repeated-proxy replay delivered 60 packets with 345 ms p95 round-trip latency and 500.8 ms median cadence. Physical microphone and ordinary cellular capture were not verified.

The engineering contribution is a reproducible local system with causal timing, explicit abstention, sham controls, provenance, and visible failure evidence. Novelty relative to all prior research has not been established.
