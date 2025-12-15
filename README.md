# 🚀 Prompt-Gen (Hybrid + Ranker)

A **hybrid AI prompt generation framework** with:
- ✨ **Streamlit UI** for easy usage  
- 🗄️ **SQLite + SQLAlchemy ORM** for storage (Users, Prompts, Choices, Feedback)  
- 🧠 **Hybrid ranker** that learns from user choices (TF-IDF & SentenceTransformer embeddings)  
- 📊 **Evaluation scripts** for model comparison & reproducibility  

---

## 📂 Project Structure
- `prompt_generator.py` → Streamlit app (UI, history, admin panel)  
- `database.py` → Database ORM (Users, Prompts, Choices, Feedback)  
- `ranker.py` → Ranker training & inference (TF-IDF + embeddings)  
- `kfold_cv.py` → K-fold experiments, writes `kfold_results.csv`  
- `ranker_retrain.py` → CLI retraining from dataset  
- `evaluation.py` → Accuracy, confusion matrix evaluation  
- `requirements.txt` → Dependencies  

---

## 🔮 Model Choices (for Ranker)
- 🟢 **all-MiniLM-L6-v2** → Small, fast, great default  
- 🔵 **all-MiniLM-L12-v2** → Larger, more accurate (≥300 examples)  
- 🟣 **paraphrase-MiniLM-L6-v2** → Good for paraphrase similarity  
- 🟠 **paraphrase-MiniLM-L12-v2** → Larger paraphrase model, better with more data  

👉 Start with `all-MiniLM-L6-v2` and scale up when dataset grows.  

---

## ⚡ Quick Start
1️⃣ Install dependencies  
pip install -r requirements.txt
2️⃣ Launch the app  
streamlit run prompt_generator.py

---

## ✨ Features

- **Prompt Generation Modes**
  - **Offline**: Deterministic template-based generator (no external API).
  - **Gemini**: Uses Google Gemini (model configured via `GEMINI_MODEL` in Streamlit secrets).
  - **Hybrid**: Generates both Offline + Gemini variants and lets user (or ranker) choose.
  - **Ensemble**: Generates both variants and synthesizes a merged prompt via `ensemble_synthesize`, saved with `model_used="ensemble"`.

- **Admin Panel**
  - **Train Ranker**: Train TF‑IDF or embedding-based LogisticRegression using choice data.
  - **Explain Ranker (SHAP)**: Visualize global feature importance for the TF‑IDF ranker using SHAP values.
  - **Export Data**: Download anonymized JSONL/CSV exports of `Prompt` and `Choice` tables with `user_id` replaced by an HMAC using `ANON_EXPORT_SALT`.
  - **Choices Dataset**: Inspect and download the training data used for the ranker.

---

## 🔁 Reproducible Experiments

- **K‑Fold Comparisons**
  - Run cross‑validated comparisons between TF‑IDF and multiple embedding models:
 
  python kfold_cv.py
    - Writes `kfold_results.csv` with per‑embedding model metrics.

- **Ranker Evaluation**
  - Evaluate a trained ranker on the current choices dataset:
 
  python evaluation.py --ranker ranker.pkl --out eval_report.json
    - Produces `eval_report.json` (accuracy & z‑score vs random) and `confusion_matrix.png`.
