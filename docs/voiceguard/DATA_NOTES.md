# Dataset provenance and scope

Large assets, resumable download blocks, extracted audio, Python environment and feature caches are stored in **D:/VoiceGuardData**. Set `VOICEGUARD_DATA` to change the runtime data location. Some experiment scripts contain the explicit workstation path; review their arguments before moving a training run.

## ASVspoof 2019 Logical Access

[Original Edinburgh release](https://datashare.ed.ac.uk/handle/10283/3336), also published by its authors on [Zenodo](https://zenodo.org/records/6906306). The original LA ZIP is 7,640,952,520 bytes. Its transfer was too slow for the working run; incomplete original download blocks are retained, not counted as a complete archive.

Full corpus acquisition uses the [Bisher Parquet mirror](https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA), pinned to revision `aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c`. Train/development/test Parquet files total 7,543,123,064 bytes. `tools/download_benchmarks.py` verifies their pinned LFS SHA256 hashes. `tools/extract_asv_parquet.py` checks every source ID, label and speaker against the corpus protocols and requires exactly 25,380 training, 24,844 development and 71,237 evaluation IDs. A mirror checksum establishes identity with that mirror, not byte equality with the original Edinburgh audio.

The working subset contains 800 training, 160 development and **108 evaluation records** (28 bona fide, 80 synthetic). Selection uses bounded metadata pages spanning class/tool regions, then a fixed seed; it is not a uniform sample of the full corpus. The requested 80 bona-fide evaluation records could not be filled from those pages, so actual counts are reported rather than pretending the target was met. Dataset-viewer audio may be re-encoded. Protocol matching is documented in its `PROVENANCE.json`. This is an exploratory, imbalanced subset, not the complete benchmark. Duration/decoding exclusions further reduce the usable counts recorded in the evaluation artifact.

The original dataset licence is preserved in `licenses/ASVSPOOF-LICENSE.txt`; its MD5 `fba2ad2f4b0620a49306799ffea58f0d` matches the published release. Consult that text for permitted uses. The custom multilingual head also uses external IndicTTS speech, so its evaluation is not an official ASVspoof competition submission.

## In-the-Wild

[Author website](https://deepfake-total.com/in_the_wild) links the [author-hosted dataset](https://huggingface.co/datasets/mueller91/In-The-Wild). The full 8,161,489,023-byte ZIP is pinned to revision `eee168f92c367f8c82ff2cf42b6f61e362fd6211`; expected SHA256 is `46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a`.

The 200-record evaluation subset uses 100 bona-fide and 100 fake records, seed 42. Ranged ZIP members were CRC-checked. In-the-Wild is evaluation-only: no records enter fitting, early stopping or calibration. Full archive acquisition does not imply full-corpus evaluation.

The author website and Hugging Face card expose differing licence labels (Apache 2.0 and CC BY-SA 4.0). Preserve attribution and review the release terms before redistribution. The submission package does **not** redistribute In-the-Wild audio.

## IndicTTS exploratory multilingual corpus

[SherryT997/IndicTTS-Deepfake-Challenge-Data](https://huggingface.co/datasets/SherryT997/IndicTTS-Deepfake-Challenge-Data) provides the existing 720-record local subset covering English, Hindi and Tamil (240 each), revision `57347517658ae989597d8cef303cffb647ed2434`. See `data/indictts/PROVENANCE.json` for acquisition metadata. The corpus card declares CC BY 4.0; attribution is included with the six demo recordings. Custom splits are transcript-group-disjoint partitions of the public labelled training split, not the source challenge's official partitions. The source does not provide verified speaker and generator identities. Transcript-group and exact-audio disjointness cannot establish complete speaker/tool disjointness.

Stored splits contain 520 train, 99 development and 101 test records before decoding/duration exclusions. Final training and evaluation reports give the actual usable counts. Records shorter than two seconds are excluded from the new model experiment rather than padded into purported additional evidence.

## Partial-fake proxies

`voiceguard.partials` constructs 100 mixed recordings, 100 real-to-real sham edits, 20 fully real and 20 fully synthetic controls exclusively from held-out IndicTTS sources. Manifests record exact intervals, parents, hashes, seed, normalization and crossfade settings. All variants of a parent remain in the same split. Sources may repeat to form six-second examples; the result is a controlled proxy, not a naturally occurring telephone-call benchmark. Unknown speaker identities and long-context score smearing are explicit limitations.

Only four control examples are bundled for the portable demo; the complete 240-record proxy corpus remains on D:. Benchmark audio is not required for ordinary offline inference.

## Receipts and reproducibility

Inspect `D:/VoiceGuardData/archives/*.provenance.json`, extraction `*-VERIFIED.json`, subset manifests and `reports/voiceguard/data-status.json`. Progress percentages alone do not prove checksum verification or successful extraction. Raw score files retain source IDs and hashes. Development determines epoch and thresholds; final test metrics never select the model.

## Telephone codec verification

The original hand-written G.711 codeword transforms differ slightly at quantization boundaries from FFmpeg (the full PCM sweep is recorded in `g711-reference-check.json`). They remain explicitly identified in the original conditions. Additional `g711_mulaw_wav`, `g711_alaw_wav` and `g711_mulaw_wav_noise10` conditions encode and decode actual ULAW/ALAW WAV containers with libsndfile. Tests verify those encoded containers decode identically in FFmpeg. These additional conditions use the same 346 held-out recordings, unchanged model weights and development thresholds. Training augmentations retain the original transforms; no final-test-driven retraining occurred.
