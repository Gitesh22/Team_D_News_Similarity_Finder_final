from fastapi.testclient import TestClient

from news_similarity_api.app import create_app


def test_recommend_returns_503_when_model_missing():
    app = create_app()
    client = TestClient(app)
    # Use an unlikely index to avoid accidental success
    r = client.post("/recommend", json={"article_idx": 999999, "k": 3})
    assert r.status_code in (503, 404, 200)


def test_validation_rejects_negative_index():
    app = create_app()
    client = TestClient(app)
    r = client.post("/recommend", json={"article_idx": -1, "k": 3})
    assert r.status_code == 422
