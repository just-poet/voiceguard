# Five-minute demonstration

1. Start the local server and open `http://127.0.0.1:8765`.
2. Explain the problem: a convincing voice alone cannot authenticate a caller. Show the trained-model status and the experiment's limitations.
3. Load one held-out English example, one Hindi example, and one Tamil example. Compare the dataset label with the detector's output; keep mistakes visible.
4. Point out the difference between the clip score and 500 ms segment scores. Show the waveform, low-quality warnings, and channel disagreement.
5. Load an annotated partial clip from `data/partials`. Compare the predicted timeline with the `fake_segments` interval in its manifest. Do not equate a highlight with ground truth.
6. Demonstrate live streaming through the microphone or audio replay. The first two seconds are warm-up; updates arrive in 500 ms chunks.
7. Export the JSON evidence report. Explain that audio stays in the local processing path and the export contains analysis, not the raw waveform recording.
8. Show the measured EER and F1 table. State which targets passed, failed, or could not be assessed.

## Short pitch

“CallGuard is a local speech-authenticity research tool for English, Hindi and Tamil. It combines a half-second localization model with a two-second context model, then compares native and simulated phone-channel views. If evidence is weak or inconsistent, it reports uncertainty. We built a CPU-only prototype, prepared a public dataset subset, and evaluated channel robustness and annotated partial fakes. Our next validation step is a speaker- and generator-disjoint corpus plus real telephone recordings.”

## Questions to expect

**Why two models?** A short window offers tighter localization; a longer window provides more acoustic context. Their thresholds are selected separately on development data.

**Is the displayed score a probability?** No. It is a classifier score. Neither probability calibration nor real-world fraud prevalence has been established.

**What is new?** The proposed project contribution is a small, auditable local system combining two timescales with channel-consistency evidence. Prior work overlaps with the components; novelty needs a literature review and controlled experiments.

**Does it handle phone calls?** It processes local microphone or replay input and simulates phone channels. Direct cellular/VoIP integration remains separate.

**How do you prove unseen-tool performance?** This demonstration dataset cannot prove that because tool identities are unavailable. The strict training mode requires identified, held-out generators for that experiment.

**Why not claim 100% accuracy?** Small samples and source artifacts can inflate results. The report shows measured errors and limitations instead.
