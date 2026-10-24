---
datasets: null
license: cc-by-sa-4.0
task_categories:
- audio-classification
language:
- en
modalities:
- audio
tags:
- audio
- deepfake
- detection
- in-the-wild
- deepfake-detection
- audio-deepfake-detection
- antispoofing
pretty_name: In The Wild
size_categories:
- 10K<n<100K
---

# In-the-Wild: A Deepfake Detection Dataset

Welcome to **In-the-Wild**, a dataset for evaluationg *audio deepfake detection*.  
It accompanies the paper: Does Audio Deepfake Detection Generalize?  [arXiv:2203.16263](https://arxiv.org/abs/2203.16263)

---

## Dataset Summary

The **In-the-Wild** dataset contains real and synthetic speech recordings of **58 celebrities and politicians**, collected from online videos.

It provides a realistic benchmark for testing how well *audio deepfake detection models generalize* beyond laboratory data such as ASVspoof.

- **Task:** Audio Classification (Deepfake / Genuine)  
- **Languages:** English  
- **Modality:** Audio  
- **Size:** 37.9 hours total  
  - 17.2 hours fake  
  - 20.7 hours real  

---

## Download

You can download the full dataset as a single ZIP file directly from this repository or via the Hugging Face `datasets` library.

### Option 1: With the `datasets` library

```python
from datasets import load_dataset

ds = load_dataset("mueller91/In-The-Wild")
```
### Option 2: wget
```
wget https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip
unzip release_in_the_wild.zip
```

## Citation
```
@article{muller2022does,
  title={Does audio deepfake detection generalize?},
  author={M{\"u}ller, Nicolas M and Czempin, Pavel and Dieckmann, Franziska and Froghyar, Adam and B{\"o}ttinger, Konstantin},
  journal={arXiv preprint arXiv:2203.16263},
  year={2022}
}
```

