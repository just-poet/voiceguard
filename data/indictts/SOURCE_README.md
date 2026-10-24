---
license: cc-by-4.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: text
    dtype: string
  - name: id
    dtype: string
  - name: language
    dtype: string
  - name: is_tts
    dtype: int64
  - name: audio
    dtype:
      audio:
        sampling_rate: 16000
  splits:
  - name: train
    num_bytes: 17420974680.498
    num_examples: 31102
  - name: test
    num_bytes: 1510248128.015
    num_examples: 2635
  download_size: 18318102787
  dataset_size: 18931222808.513
---
## **IndicTTS Deepfake Detection Challenge**  

Participants will use the **`SherryT997/IndicTTS-Deepfake-Challenge-Data`** dataset, hosted on **Hugging Face**. This dataset consists of **train** and **test** splits and contains speech samples in **16 Indian languages**, along with metadata for each audio clip.  

### **🚀 Dataset to Use: `SherryT997/IndicTTS-Deepfake-Challenge-Data`**  
This is the **official dataset** for the challenge and must be used for training and evaluation.  

---

## **📌 Dataset Structure**  
The dataset consists of the following columns:  

| Column   | Type      | Description |  
|----------|----------|-------------|  
| **id**   | `int`    | Unique identifier for each audio sample |  
| **audio** | `Audio()` | Audio file in a format supported by Hugging Face's `datasets` library |  
| **text** | `string` | Transcription of the spoken audio |  
| **language** | `string` | Language of the audio sample |  
| **is_tts** | `int` | Label indicating if the audio is AI-generated (`1`) or real (`0`) (**only available in train set**) |  

🔹 **Important:** The **test set does not contain the `is_tts` labels**. Instead, for all test samples, `is_tts` is set to `-1`, and participants must predict its value.  

---

## **📊 Dataset Statistics**  
- **Train Set:** **31,102** samples (contains `is_tts` labels)  
- **Test Set:** **2,664** samples (`is_tts = -1` for all rows)  

---

## **🗣️ Languages Included**  
This dataset covers **16 Indian languages**, ensuring diversity in speech patterns and phonetics:  

✅ Assamese  
✅ Bengali  
✅ Bodo  
✅ Dogri  
✅ Kannada  
✅ Malayalam  
✅ Marathi  
✅ Sanskrit  
✅ Nepali  
✅ English  
✅ Telugu  
✅ Hindi  
✅ Odia  
✅ Manipuri  
✅ Gujarati  
✅ Tamil  

---

## **📥 Accessing the Dataset**  
Participants can load the dataset directly from **Hugging Face** using the `datasets` library:  

```python
from datasets import load_dataset

# Official dataset for this challenge
dataset = load_dataset("SherryT997/IndicTTS-Deepfake-Challenge-Data")  

# Train and test splits
train_data = dataset["train"]  # Contains 'is_tts' labels
test_data = dataset["test"]  # 'is_tts' is -1 for all rows
```