from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from news_similarity_api.recommender import (
    ArticleNotFoundError,
    ModelNotReadyError,
    load_artifacts,
    recommend_by_index,
)
from news_similarity_api.schemas import RecommendRequest, RecommendResponse


def create_app() -> FastAPI:
    app = FastAPI(
        title="News Article Similarity API (TF-IDF + AG News)",
        version="0.1.0",
        description="Input an article index and get TF-IDF-based similar articles.",
    )
    artifacts_dir = Path(__file__).resolve().parents[2] / "artifacts"
    state = {"art": None}

    @app.on_event("startup")
    def _startup():
        print("[Startup] Loading artifacts from:", artifacts_dir)
        try:
            state["art"] = load_artifacts(artifacts_dir)
            print("[Startup] Artifacts loaded successfully.")
        except ModelNotReadyError as e:
            print(f"[Startup] ModelNotReadyError: {e}")
            state["art"] = None
        except Exception as e:
            print(f"[Startup] Unexpected error: {type(e).__name__}: {e}")
            state["art"] = None

    @app.get("/health")
    def health():
        print("[Endpoint] /health called")
        return {"status": "ok", "model_ready": state["art"] is not None}

    @app.get("/articles/search")
    def search_articles(q: str = Query(..., min_length=1, max_length=50)):
        print(f"[Endpoint] /articles/search called with q={q}")
        try:
            if state["art"] is None:
                print("[Endpoint] Model not ready error in /articles/search")
                raise HTTPException(status_code=503, detail="Model not ready. Train it first.")
            articles = state["art"].articles
            # Search in Title and Description columns, not Class Index
            mask = articles["Title"].str.contains(q, case=False, na=False) | articles[
                "Description"
            ].str.contains(q, case=False, na=False)
            hits = articles[mask].head(20)
            # Return index, title, and description for each match
            results = [
                {
                    "idx": int(idx),
                    "title": row["Title"],
                    "description": row["Description"][:100] + "..."
                    if len(row["Description"]) > 100
                    else row["Description"],
                }
                for idx, row in hits.iterrows()
            ]
            print(f"[Endpoint] /articles/search returning {len(results)} results")
            return {"query": q, "results": results}
        except Exception as e:
            print(f"[Endpoint] /articles/search exception: {type(e).__name__}: {e}")
            raise

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest) -> RecommendResponse:
        print(f"[Endpoint] /recommend called with idx={req.article_idx}, k={req.k}")
        try:
            if state["art"] is None:
                print("[Endpoint] Model not ready error in /recommend")
                raise HTTPException(status_code=503, detail="Model not ready. Train it first.")
            recs = recommend_by_index(state["art"], req.article_idx, req.k)
            print(f"[Endpoint] /recommend returning {len(recs)} recommendations")
            rec_response = RecommendResponse(
                input_idx=req.article_idx,
                recommendations=recs,
            )
            return rec_response
        except ArticleNotFoundError as e:
            print(f"[Endpoint] ArticleNotFoundError: {e}")
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:
            print(f"[Endpoint] /recommend exception: {type(e).__name__}: {e}")
            raise

    @app.exception_handler(Exception)
    def unhandled_exception_handler(_, exc: Exception):
        return JSONResponse(
            status_code=500, content={"detail": "Internal error", "type": type(exc).__name__}
        )

    return app


app = create_app()
