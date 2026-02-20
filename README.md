# News Article Similarity Finder

**Intelligent content discovery using TF-IDF and machine learning**

---

## 1️⃣ Executive Summary

### Problem Statement
News readers and content curators face information overload when trying to discover related articles. Manual search and categorization is time-consuming and inconsistent.

### Solution
An intelligent news similarity system that automatically finds related articles using natural language processing (NLP) and machine learning, enabling:
- **Content discovery** through semantic similarity
- **Keyword-based search** across large article datasets
- **Real-time recommendations** via REST API

### Target Audience
- News platforms and media organizations
- Content recommendation systems
- Research and analytics teams
- Data scientists and ML engineers

### Value Proposition
- ⚡ **Fast**: Sub-second response times for similarity queries
- 🎯 **Accurate**: TF-IDF vectorization captures semantic relationships
- 🔌 **Flexible**: RESTful API enables easy integration
- 📊 **Scalable**: Efficient KNN-based retrieval on 7,600+ articles

---

## 2️⃣ Challenges

### Technical Challenges
- **Scale**: Processing and indexing thousands of news articles efficiently
- **Semantic Understanding**: Capturing article similarity beyond simple keyword matching
- **Performance**: Delivering real-time recommendations with minimal latency
- **Data Quality**: Handling diverse article formats and missing information

### Constraints
- **Memory Efficiency**: Loading vectorized representations for fast retrieval
- **API Design**: Balancing simplicity with functionality
- **Model Size**: Pre-computed artifacts must be portable and version-controlled
- **Error Handling**: Graceful degradation when models are unavailable

---

## 3️⃣ Solution Overview

### Approach
The solution uses **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization combined with **K-Nearest Neighbors (KNN)** for similarity retrieval.

### End-to-End Flow

```
User Input → Frontend UI → REST API → Model Processing → Similar Articles
     ↓                          ↓              ↓                ↓
  Keyword              FastAPI Backend    TF-IDF + KNN      Ranked Results
  or Index             Validation         Vectorization     with Metadata
```

**Step-by-Step Process:**
1. **User Interaction**: Search by keyword or select article by index
2. **API Request**: Frontend sends request to FastAPI backend
3. **Text Vectorization**: Article text converted to TF-IDF vectors
4. **Similarity Computation**: KNN finds nearest neighbors in vector space
5. **Response Formatting**: Results returned with titles, descriptions, similarity scores
6. **UI Display**: Streamlit presents ranked recommendations

### Component Interaction
- **Frontend** (Streamlit) handles user interaction and visualization
- **Backend** (FastAPI) manages business logic and validation
- **Model Layer** (TF-IDF + KNN) performs similarity computation
- **Artifacts** (pre-trained models) enable fast inference without retraining

---

## 4️⃣ Solution Architecture

### High-Level System Design

**Frontend Layer**
- **Technology**: Streamlit
- **Responsibilities**: 
  - User interface for search and recommendations
  - API client for backend communication
  - Result visualization and formatting
- **Port**: 8501

**API Layer**
- **Technology**: FastAPI
- **Responsibilities**:
  - RESTful endpoint management
  - Request/response validation (Pydantic schemas)
  - Error handling and status codes
  - Model lifecycle management
- **Port**: 8000

**Backend Logic**
- **Recommender Module**: Core similarity computation
- **Artifact Loader**: Model and data initialization
- **Exception Handling**: Custom error types for different failure modes

**Model Layer**
- **TF-IDF Vectorizer**: Transforms text into numerical vectors
- **KNN Model**: Finds k-nearest neighbors using cosine similarity
- **Pre-computed Artifacts**: Stored in `artifacts/` for fast loading

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND (Streamlit)                   │
│  - Search Interface  - Article Selection  - Results Display │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP Requests
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    API LAYER (FastAPI)                      │
│  /health  │  /articles/search  │  /recommend                │
└────────────────────────────┬────────────────────────────────┘
                             │ Function Calls
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   BUSINESS LOGIC                            │
│  - recommender.py: Similarity computation                   │
│  - schemas.py: Data validation models                       │
└────────────────────────────┬────────────────────────────────┘
                             │ Model Loading
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    MODEL ARTIFACTS                          │
│  TF-IDF Vectorizer  │  KNN Model  │  Articles Dataset       │
│  (tfidf.joblib)     │ (knn.joblib)│ (articles.parquet)      │
└─────────────────────────────────────────────────────────────┘
```

### Solution Diagram

> **Note**: For the presentation, include a visual diagram showing:
> - User → Streamlit UI → FastAPI Backend → ML Models → Response
> - Component interactions and data flow
> - Technology stack icons for each layer

---

## 5️⃣ Tooling & Engineering Practices

### Technology Stack

**Frontend**
- **Streamlit**: Interactive web UI framework
- **Requests**: HTTP client for API communication

**Backend**
- **FastAPI**: Modern async web framework
- **Pydantic**: Data validation and schema management
- **Uvicorn**: ASGI server for production deployment

**Machine Learning**
- **Scikit-learn**: TF-IDF vectorization and KNN implementation
- **Pandas**: Data manipulation and analysis
- **Joblib**: Model serialization and loading

**Development Tools**
- **uv**: Fast Python package and project manager
- **Pytest**: Testing framework with automated test suite
- **Ruff**: Linting and code formatting
- **Python 3.13**: Latest language features and performance

### Engineering Best Practices

**Code Quality**
- ✅ Automated linting with Ruff (all checks passing)
- ✅ Consistent code formatting
- ✅ Type hints and schema validation
- ✅ Comprehensive error handling

**Testing & Validation**
- ✅ Unit tests for all critical endpoints
- ✅ Input validation (negative indices, invalid queries)
- ✅ Service health checks
- ✅ Test coverage for error scenarios

**Architecture Principles**
- **Modular Design**: Clear separation of concerns (API, logic, models)
- **Dependency Injection**: State management for artifact loading
- **Error Segregation**: Custom exceptions for different failure modes
- **Logging**: Debug logging throughout request lifecycle

**Version Control**
- `.gitignore` excludes raw data and virtual environments
- Model artifacts committed for reproducibility
- Clear project structure with `src/` layout

**Environment Management**
- `pyproject.toml`: Centralized dependency management
- Virtual environment isolation with `uv`
- Cross-platform compatibility (Windows/Linux/Mac)

**Deployment Readiness**
- Health check endpoint for monitoring
- Graceful degradation when models unavailable
- Configurable host/port settings
- Auto-reload for development (`--reload` flag)

---

## 6️⃣ API Development

### Endpoint Overview

| Endpoint | Method | Purpose | Status Codes |
|----------|--------|---------|--------------|
| `/health` | GET | Service health check | 200 |
| `/articles/search` | GET | Search articles by keyword | 200, 503 |
| `/recommend` | POST | Get similar articles | 200, 404, 422, 503 |

### API Documentation

**1. Health Check**
```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "model_ready": true
}
```

**2. Article Search**
```http
GET /articles/search?q=pension&k=5
```

**Parameters:**
- `q` (required): Search keyword (1-50 characters)
- `k` (optional): Number of results (default: 20)

**Response:**
```json
{
  "query": "pension",
  "results": [
    {
      "idx": 0,
      "title": "Fears for T N pension after talks",
      "description": "Unions representing workers at Turner..."
    }
  ]
}
```

**3. Recommend Similar Articles**
```http
POST /recommend
Content-Type: application/json

{
  "article_idx": 0,
  "k": 3
}
```

**Request Schema:**
```python
{
  "article_idx": int (>= 0),  # Article index
  "k": int (1-10)             # Number of recommendations
}
```

**Response:**
```json
{
  "input_idx": 0,
  "recommendations": [
    {
      "idx": 867,
      "title": "Federal-Mogul May Sell Turner & Newall Assets",
      "reason": "Textually similar article"
    }
  ]
}
```

### Design Principles

**RESTful Design**
- Resource-oriented URLs (`/articles/search`, `/recommend`)
- Standard HTTP methods (GET for retrieval, POST for operations)
- Meaningful status codes (200, 404, 422, 503)
- JSON request/response format

**Validation & Error Handling**
- **Schema Validation**: Pydantic models enforce data contracts
- **Input Sanitization**: Query length limits, range constraints
- **Custom Exceptions**: `ModelNotReadyError`, `ArticleNotFoundError`
- **Graceful Degradation**: 503 status when models unavailable

**Status Code Strategy**
- `200 OK`: Successful operation
- `404 Not Found`: Article index out of range
- `422 Unprocessable Entity`: Validation error (e.g., negative index)
- `503 Service Unavailable`: Model not loaded

**Scalability Considerations**
- Startup event for one-time model loading
- In-memory artifact caching for fast retrieval
- Async-ready framework (FastAPI)
- Pagination support via `k` parameter

---

## 7️⃣ Model & Approach

### Model Selection: TF-IDF + KNN

**Why TF-IDF?**
- **Interpretable**: Clear understanding of feature importance
- **Fast**: Efficient sparse matrix operations
- **Effective**: Captures term relevance across document corpus
- **No training required**: Pre-computed on full dataset

**Why K-Nearest Neighbors?**
- **Simplicity**: No complex hyperparameter tuning
- **Accuracy**: Direct similarity measurement in vector space
- **Flexibility**: Works well for recommendation tasks
- **Cosine Similarity**: Captures semantic relatedness regardless of document length

### Data Processing Pipeline

**1. Data Acquisition**
- **Dataset**: AG News Classification Dataset (7,600 articles)
- **Source**: Kaggle (optional download for maintainers)
- **Columns**: Class Index, Title, Description

**2. Text Preprocessing**
- Combine title and description into `full_text`
- No stemming or lemmatization (preserves readability)
- Stop word removal via TF-IDF (`stop_words='english'`)

**3. Feature Engineering**
- **Vectorization**: TF-IDF with max 10,000 features
- **Normalization**: L2 normalization for cosine similarity
- **Dimensionality**: Sparse matrix representation for memory efficiency

**4. Model Training**
```bash
uv run python scripts/train_tfidf.py
```
- Fits TF-IDF vectorizer on full corpus
- Builds KNN index with `n_neighbors=10`
- Saves artifacts: `tfidf.joblib`, `knn.joblib`, `articles.parquet`

### Validation Strategy

**Offline Validation**
- Sample recommendation review during training
- Sanity checks for similar article retrieval

**Runtime Validation**
- Index range verification
- Model readiness checks on startup
- Error handling for edge cases

### Limitations & Future Enhancements

**Current Limitations**
- **Static Index**: No real-time article addition
- **English Only**: Limited to English-language articles
- **Bag-of-Words**: Doesn't capture word order or context
- **No Personalization**: Same results for all users

**Potential Enhancements**
- 🚀 **Transformer Models**: BERT/Sentence-BERT for semantic embeddings
- 📈 **Online Learning**: Incremental model updates
- 🌐 **Multilingual Support**: Cross-language similarity
- 👤 **User Profiles**: Personalized recommendations
- 📊 **Hybrid Models**: Combine collaborative + content-based filtering

---

## 8️⃣ How to Run

### Prerequisites
- **Python**: 3.13 or higher
- **uv**: Fast Python package manager ([installation guide](https://docs.astral.sh/uv/))
- **Operating System**: Windows, macOS, or Linux

### Quick Start

**1. Clone the Repository**
```bash
git clone <repository-url>
cd fastapi_news_similarity
```

**2. Install Dependencies**
```bash
uv sync
```

**3. Start the Backend (Terminal 1)**
```bash
uv run uvicorn news_similarity_api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

**4. Start the Frontend (Terminal 2)**
```bash
uv run streamlit run streamlit_app.py
```

**5. Access the Application**
- **Streamlit UI**: http://localhost:8501
- **API Documentation**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health

### Development Workflow

**Run Tests**
```bash
uv run pytest -v
```

**Code Quality Checks**
```bash
uv run ruff check .        # Lint code
uv run ruff format .       # Format code
```

**Development Mode** (auto-reload on changes)
```bash
uv run uvicorn news_similarity_api.app:app --app-dir src --reload
```

### Project Structure
```
fastapi_news_similarity/
├── src/
│   └── news_similarity_api/
│       ├── app.py              # FastAPI application
│       ├── recommender.py      # Core similarity logic
│       └── schemas.py          # Pydantic models
├── tests/
│   ├── test_health.py          # Health endpoint tests
│   └── test_recommend.py       # Recommendation tests
├── scripts/
│   ├── download_data.py        # Dataset download (optional)
│   └── train_tfidf.py          # Model training
├── artifacts/
│   ├── articles.parquet        # Article dataset
│   ├── tfidf.joblib            # TF-IDF vectorizer
│   └── knn.joblib              # KNN model
├── streamlit_app.py            # Frontend UI
└── pyproject.toml              # Dependencies
```

---

## Dataset Download Instructions (For Maintainers Only)

**Testers and users do NOT need to download the AG News dataset or set up a Kaggle API key.**

All required model artifacts will be provided in the repository (excluding raw data), so you can run, test, and use the app without any dataset download steps.

---

### For Maintainers (One-Time Setup)

If you need to re-generate the model artifacts from scratch, follow these steps:

#### 1. Get your Kaggle API key

1. Go to https://www.kaggle.com/ → Sign in
2. Click on your profile picture (top right) → Account
3. Scroll down to the "API" section
4. Click "Create New API Token" or copy the API key provided
5. Copy your API key string (it will look like a long random string)

#### 2. Set your API key as an environment variable

- **In VS Code Terminal (Windows PowerShell):**
	- Open a new terminal in VS Code (Terminal → New Terminal)
	- Paste and run:
		```powershell
		$env:KAGGLE_API_TOKEN="your_actual_api_key"
		```
- **Linux/Mac:**
	```bash
	export KAGGLE_API_TOKEN=your_actual_api_key
	```

You must run this command in the same terminal session before running the download script.

#### 3. Download the dataset

Once your API key is set, run:

```
uv run python scripts/download_data.py
```

This will download and extract the AG News dataset into the `data/` directory.

**Note:** The `data/` directory is excluded from version control and will not be uploaded to GitHub.

---

## 📚 Additional Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **Scikit-learn TF-IDF**: https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
- **AG News Dataset**: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset

---

## 📄 License & Contact

**Project**: News Article Similarity Finder  
**Version**: 0.1.0  
**Authors**: [Your Team Name]  
**Contact**: [Your Email]

---

*Built with ❤️ using FastAPI, Streamlit, and Scikit-learn*
