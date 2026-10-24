---
dataset_info:
  features:
  - name: speaker_id
    dtype: string
  - name: audio_file_name
    dtype: string
  - name: audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: system_id
    dtype: string
  - name: key
    dtype:
      class_label:
        names:
          '0': bonafide
          '1': spoof
  splits:
  - name: train
    num_bytes: 1515101755.52
    num_examples: 25380
  - name: validation
    num_bytes: 1462206958.528
    num_examples: 24844
  - name: test
    num_bytes: 4334147022.707
    num_examples: 71237
  download_size: 7543123064
  dataset_size: 7311455736.755
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---
