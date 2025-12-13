# 📘 Visual Language Models & Finetuning – Final Project

EE596 Deep Learning – University of Washington

---

## 🧠 **Project Overview**

This project explores **Visual Question Answering (VQA)** for **medical imaging** and **DAQUAR**, comparing the performance of multiple vision–language architectures:

* **ResNet50 + BERT (Baseline)**
* **PubMedCLIP**
* **ViLT**
* **Qwen2-VL (Vision-Language LLM)**

We evaluate these models on a curated medical VQA dataset containing radiology images and short-answer questions.
Both **token-level accuracy** and **semantic similarity** (SBERT cosine similarity) are used to assess answer quality.

---

# 📂 Repository Structure

```
EE596_DEEPLEARNING/
│── checkpoints/              # Saved model weights (empty; weights via download link)
│── data/                     # (Optional) Public dataset references only
│── demo/                     # Example notebooks for running inference
│   ├── [Final] project_training_Medical_*.ipynb
│── results/                  # Plots, tables, and analysis
│── src/
│   ├── main.py               # Entry point placeholder (see note below)
│   ├── model.py              # Model definition placeholder
│   └── utils.py              # Utility function placeholder
│── requirements.txt
└── README.md   ← (this file)
```

---

# ❗ About `main.py`, `model.py`, and `utils.py`

The assignment template requires these three files.
**Because this project was implemented entirely in Jupyter Notebooks** (to support training, visualization, and evaluation workflows), the notebook cells already contain:

* Model definitions
* Data preprocessing
* Training loops
* Evaluation logic

---

# 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/EE596_DEEPLEARNING.git
cd EE596_DEEPLEARNING
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Create a clean virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# ▶️ How to Run the Demo

A minimal demo notebook is provided:

```
demo/[Final]_final_project_demo_result.ipynb
```

### Example Usage

```python
df_results = pd.DataFrame({
    "gt_answer": ground_truths,
    "pred_answer": predictions,
    "is_correct": [1 if p.lower().strip() == g.lower().strip() else 0 for p, g in zip(predictions, ground_truths)],
    "similarity": similarities
})
# [NEW] Calculate and Print Final Metrics
final_acc = df_results["is_correct"].mean()
avg_sim = df_results["similarity"].mean()

print("="*30)
print(f"Final Test Accuracy: {final_acc:.4f}")
print(f"Final SBERT Similarity: {avg_sim:.4f}")
print("="*30)
```

---

# 🎯 Expected Output

Example output from the demo:

```
==============================
Final Test Accuracy: 0.6089
Final SBERT Similarity: 0.7222
==============================
```

Plots generated in the results folder include:

* Accuracy comparison across models
* SBERT similarity comparison
* Training curves

---

# 🧪 Training & Evaluation Summary

## **Medical Dataset Results**

| Model               | Test Accuracy | SBERT Similarity | Training Time |
| ------------------- | ------------- | ---------------- | ------------- |
| **ResNet50 + BERT** | **60.89%**    | **72.22%**       | 1 hour        |
| PubMedCLIP          | 60.44%        | 70.78%           | 2 hours       |
| ViLT                | 56.89%        | 69.87%           | 2 hours       |
| Qwen2-VL            | 56.00%        | 67.94%           | > 5 hours     |

---

# 🔗 Pre-trained Model Download

To keep the repository light, model weights are stored externally:

https://drive.google.com/drive/folders/1q4h3m9zmGKlhcSPcC0Y9Qfb4wTzPEGe2?usp=sharing


