
# 🧠 RoBERTa Token Classification with Additional PLODv2 Data

This repository contains two Python scripts that fine-tune a `roberta-base` model for Named Entity Recognition (NER) on the [PLOD-CW-25](https://huggingface.co/datasets/surrey-nlp/PLOD-CW-25) dataset, with **25%** and **50%** additional samples from the `PLODv2-filtered` dataset. The goal is to evaluate performance gains in token-level classification using partial data augmentation.

---

## 📁 Files

| Filename | Description |
|----------|-------------|
| `RoBERTa+25%data.py` | Fine-tunes `roberta-base` on PLOD-CW-25 with 25% of `PLODv2-filtered` samples added to training and validation sets. |
| `RoBERTa+50%data.py` | Fine-tunes `roberta-base` on PLOD-CW-25 with 50% of `PLODv2-filtered` samples added to training and validation sets. |

---

## 🔍 Dataset Overview

- **PLOD-CW-25**: Legal NER dataset with annotated tokens from case law.
- **PLODv2-filtered**: A filtered version of PLODv2 for optional fine-tuning enhancement.

Each script:
- Loads both datasets.
- Randomly samples a fraction of `PLODv2-filtered`.
- Merges it with the original training and validation splits.
- Converts the data into Hugging Face Datasets format.

---

## 🔧 Model Architecture

- Model: `roberta-base`
- Task: Token classification
- Tags: `O`, `B-AC`, `B-LF`, `I-LF`
- Tokenizer: `RobertaTokenizerFast` with `add_prefix_space=True`
- Optimizer: `Adafactor`
- Epochs: 3
- Batch Size: 16
- Scheduler: Constant LR

---

## 📊 Metrics & Evaluation

- Evaluation Metric: [`seqeval`](https://huggingface.co/metrics/seqeval)
- Reports:
  - Overall: Precision, Recall, F1, Accuracy
  - Entity-wise: Per-class F1
  - Visuals: Confusion Matrix & Bar Plot for metrics

---

## 📈 Visual Outputs

Each script generates:
- 📋 Classification report
- 📊 Bar chart for precision/recall/F1 by entity
- 🔲 Confusion matrix for true vs. predicted labels

---

## 📊 Side-by-Side Result Comparison

| Metric      | 25% Additional Data | 50% Additional Data |
|-------------|---------------------|----------------------|
| Precision   | ~0.88               | ~0.90                |
| Recall      | ~0.89               | ~0.91                |
| F1 Score    | ~0.88–0.89          | ~0.90+               |
| Accuracy    | ~0.89               | ~0.90                |

*(Note: Replace these with actual values from your output for a final report.)*

---

## 🧪 Reproducibility

Random seeds (`SEED = 42`) are set for:
- NumPy
- `random`
- PyTorch (CPU & GPU)
- Transformers

---

## 🛠️ Installation

```bash
pip install datasets transformers huggingface_hub evaluate seqeval nbconvert
```

---

## 📌 Notes

- Scripts are optimized for execution in Google Colab.
- Designed for experimentation with data augmentation in NER tasks.
- Results can help in assessing trade-offs between more data vs. training time.

---

## 📤 Output Example

```json
{
  "precision": 0.89,
  "recall": 0.91,
  "f1": 0.90,
  "accuracy": 0.90
}
```

📁 Check the detailed output in the confusion matrix and classification report printed at the end of each script.

---

## ✍️ Author

Aaditya Singh – MSc Business Analytics  
Contributions: Data Augmentation, Model Training, Evaluation, Reporting  
For academic use and performance benchmarking of NLP models in the legal domain.

---
