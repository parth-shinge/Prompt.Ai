Prompt-Gen (Hybrid + Ranker)

Overview
- `prompt_generator.py`: Streamlit app (generation UI, history, admin panel)
- `database.py`: SQLite + SQLAlchemy ORM (Users, Prompts, Choices, Feedback)
- `ranker.py`: Training and inference utilities (TF‑IDF and SentenceTransformer embeddings)
- `kfold_cv.py`: K‑fold comparison script, writes `kfold_results.csv`
- `ranker_retrain.py`: CLI retraining from the choices dataset
- `evaluation.py`: Offline evaluation (accuracy, confusion matrix)
- `requirements.txt`: Python packages

Model choices (embeddings for the ranker, not LLMs)
- all‑MiniLM‑L6‑v2: 6‑layer MiniLM, small and fast; general‑purpose similarity. Great default.
- all‑MiniLM‑L12‑v2: 12‑layer version; higher capacity, slower; can help with more data (≥100–300 examples).
- paraphrase‑MiniLM‑L6‑v2: fine‑tuned for paraphrase similarity; good when user choices depend on wording/semantics.
- paraphrase‑MiniLM‑L12‑v2: larger paraphrase model; better with larger datasets.
Guidance: start with `all‑MiniLM‑L6‑v2`. Try 12‑layer or paraphrase variants when you have more labeled choices and want to compare.

Quick start
1) Install
   pip install -r requirements.txt

2) Configure secrets (Streamlit Cloud or .streamlit/secrets.toml locally)
   - `GEMINI_API_KEY` (optional for Gemini generation)
   - `ADMIN_USERNAME`, `ADMIN_PW_SALT`, `ADMIN_PW_HASH` (for Admin panel)

3) Initialize DB (auto on first run). If migrating from older DB, back up or remove `promptgen.db`.

4) Run
   streamlit run prompt_generator.py

Collecting training data (Choices)
- In Generate tab, enable “Generate both variants and choose (Hybrid mode)”.
- Pick Offline or Gemini for each pair. These choices populate the `choices` table.
- Aim for balanced labels across both classes.

Training the ranker (Admin panel)
- Admin → Train Ranker
  - Select embedding model and click “Train embedding‑based ranker”.
  - For tiny datasets you may see `mean: NaN (note: no_cv)`. The model still saves to `ranker.pkl` and is usable.
  - “Run quick comparison” compares TF‑IDF vs the selected embedding and stores temporary models under `artifacts/`.

CLI training and evaluation
- Retrain (embedding):
   python ranker_retrain.py --model all-MiniLM-L6-v2 --out ranker.pkl
- TF‑IDF fallback:
   python ranker_retrain.py --tfidf --out ranker.pkl
- K‑fold experiments:
   python kfold_cv.py
- Evaluate current model:
   python evaluation.py --ranker ranker.pkl

Using the ranker in the app
- Hybrid (checkbox off): app auto‑generates both and lets the ranker pick; if no model, you’ll be prompted to choose.
- Any mode + checkbox on: always generates both, shows both, and records your choice (good for building the dataset).

Artifacts and temp files
- Main model: `ranker.pkl`
- Temporary quick‑comparison models: `artifacts/tmp_tfidf.pkl`, `artifacts/tmp_emb.pkl`

GitHub: push this project
1) Create a new GitHub repo (empty).
2) In this folder:
   git init
   git remote add origin https://github.com/<your-username>/<repo>.git
   git add .
   git commit -m "Initial commit: Prompt-Gen hybrid + ranker"
   git branch -M main
   git push -u origin main

Deploy on Streamlit Cloud
1) Connect your GitHub repo in Streamlit Cloud.
2) App path: `prompt_generator.py`
3) Python version: 3.12 (or your local), and provide `requirements.txt`.
4) Add Secrets (in Streamlit Cloud):
   GEMINI_API_KEY = "..."
   ADMIN_USERNAME = "admin"
   ADMIN_PW_SALT = "<hex>"
   ADMIN_PW_HASH = "<hex>"
5) Deploy. The SQLite file `promptgen.db` will be created in the app’s working directory on first run.

Generating secure admin password hash
- Run `python gen_admin_hash.py` (if present) or any PBKDF2 tool to create salt+hash.

Notes
- Keep collecting choices; retrain periodically.
- For reproducibility, pin model names and record `requirements.txt` versions if publishing results.
