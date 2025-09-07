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
```bash
pip install -r requirements.txt
