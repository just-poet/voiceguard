# Research contribution and honest claims

## Proposed contribution

**CallGuard: two-timescale speech evidence with counterfactual channel consistency.**

The application asks two different questions: whether a two-second region contains synthetic evidence, and where a short suspicious span begins or ends. A short-window head localizes evidence at 500 ms resolution. A longer-window head stabilizes clip-level evidence. Each analyzes the native signal and two transformed telephone views. Disagreement can produce an uncertain result instead of an overconfident binary answer.

This is an engineering and experimental contribution to investigate. Its components have prior art. It must not be described as a technique nobody has ever tried.

## Falsifiable hypotheses

1. Averaging channel views improves worst-channel EER compared with the same model's native-only score. The evaluator reports both values.
2. A 500 ms head localizes short manipulations more accurately than a two-second rolling head. Compare at identical bin resolution and on the same held-out source audio.
3. Disagreement abstention reduces accepted-set error without unacceptable loss of coverage. Always report both error and coverage; never omit abstained examples from the main EER.

The present small-data experiment can demonstrate the pipeline. It cannot establish unseen-generator robustness, because its source metadata does not contain generator IDs. Unknown speaker identities also prevent a proper speaker-disjoint experiment.

## Required next rigorous experiment

- Use multiple identified synthesis families in each target language and reserve whole families for test.
- Keep speaker, source utterance, and transcript groups disjoint between train, development and test.
- Preserve an untouched test set while selecting model settings on development data.
- Build same-speaker, same-language partial fakes; vary insertion duration, boundary alignment, loudness and crossfade style.
- Include genuine edit controls and real phone recordings to avoid learning splice boundaries, recording sources, or codecs as shortcuts.
- Compare native-only, view averaging, view averaging with abstention, long-only localization, and the two-head system.
- Evaluate Hindi, English and Tamil separately as well as mixed-language/code-switched clips. A pooled score can conceal weak language performance.
- Report EER, ROC-AUC, frame precision/recall/F1, accepted-set error and coverage, CPU p50/p95, and confidence intervals grouped by speaker/source.

## Prior work consulted

1. [ASVspoof 2021 official baselines and evaluation package](https://github.com/asvspoof-challenge/2021). Provides established anti-spoofing evaluation and telephone/compression conditions.
2. [Unmasking Deepfakes: Leveraging Augmentations and Features Variability for Deepfake Speech Detection](https://arxiv.org/abs/2501.05545). Studies augmentation and compression-aware training. Compression robustness is already a research direction.
3. [Half-Truth Audio Detection and Localisation: A Lightweight Cross-Attentive Architecture and a Cross-Corpus Diagnostic Study](https://arxiv.org/abs/2605.29531). Studies partial manipulation and illustrates why cross-corpus performance must be measured rather than inferred from an in-domain result.
4. [IndicSynth dataset](https://huggingface.co/datasets/vdivyasharma/IndicSynth). A potential larger source for Indic-language synthetic speech; matching genuine data and usage conditions must be prepared for a future experiment.

## Claims suitable for the demonstration

- Runs locally on a CPU with no inference API costs.
- Was trained on a bounded public subset containing English, Hindi and Tamil.
- Supports 500 ms streaming updates and distinguishes whole-clip evidence from local segment evidence.
- Measures performance under the declared simulated phone conditions.
- Exports auditable reports and records missing evidence rather than inventing results.

## Claims not supported

- Detects all cloned voices, all scam calls, or arbitrary unseen synthesis tools.
- Authenticates the caller's identity or intent.
- Works reliably in all three languages outside the sampled corpus.
- Has passed the original Phase 1 retention requirement without those original results and test protocol.
- Is the first method of its kind.
