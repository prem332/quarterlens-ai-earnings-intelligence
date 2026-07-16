from pydantic import BaseModel, Field
from .shared import Company, Quarter


class AnalysisRequest(BaseModel):
    company: Company
    quarter: Quarter = Field(..., example="Q2_2025")
    comparison_quarters: list[Quarter] = Field(
        default_factory=list,
        max_length=3,
        example=["Q1_2025", "Q4_2024"],
    )
    query: str = Field(
        default="Summarize key earnings findings and verify management claims.",
        max_length=500,
    )