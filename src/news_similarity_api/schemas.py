from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    article_idx: int = Field(..., ge=0, description="Index of the article in the dataset.")
    k: int = Field(3, ge=1, le=10, description="How many recommendations to return (max 10).")


class ArticleRecommendation(BaseModel):
    idx: int
    title: str
    reason: str


class RecommendResponse(BaseModel):
    input_idx: int
    recommendations: list[ArticleRecommendation]
