# News Similarity Finder — Chat Context & Code Flow

## Project Purpose
Find similar news articles using a FastAPI backend (TF-IDF + KNN) and a Streamlit UI. All code and artifacts are in `C:\fastapi_news_similarity`.

## Code Flow
1. **User opens Streamlit UI** (`streamlit_app.py`)
2. **UI sends search/recommend requests** to FastAPI backend (`src/news_similarity_api/app.py`)
3. **Backend loads artifacts** (`artifacts/`):
    - `articles.parquet` (news data)
    - `tfidf.joblib` (vectorizer)
    - `knn.joblib` (KNN model)
4. **Backend endpoints:**
    - `/health`: Check server/model status
    - `/articles/search`: Search articles by keyword
    - `/recommend`: Get similar articles by index
5. **Recommender logic** (`recommender.py`):
    - Loads artifacts
    - Finds similar articles using TF-IDF + KNN
6. **UI displays results** to user

## Key Files
- `streamlit_app.py`: Streamlit UI (search, select, recommend)
- `src/news_similarity_api/app.py`: FastAPI backend
- `src/news_similarity_api/recommender.py`: Loads artifacts, computes recommendations
- `artifacts/`: Prebuilt model files
- `scripts/`: Data download and model training (maintainers only)

## Usage
1. Start backend:
   - `uv run uvicorn news_similarity_api.app:app --app-dir src --host 127.0.0.1 --port 8000`
2. Start UI:
   - `uv run streamlit run streamlit_app.py`
3. Open [http://localhost:8501](http://localhost:8501)

## Troubleshooting
- **Port in use:** Kill any process using port 8000
- **Backend shuts down:** Check for missing artifacts or code errors
- **UI can't connect:** Ensure backend is running before UI

## Automated Tests
- Run: `uv run pytest -q`

---
**This file is for chatbots and new contributors to quickly understand the project context and code flow.**
