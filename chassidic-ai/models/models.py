# Request model
from pydantic import BaseModel
from typing import Any


class RelevanceJudgement(BaseModel):
    passage_id: str
    model: str
    score: float
    reason: str
    timestamp: str
    other_params: str


class CandidatePassage(BaseModel):
    compound_id: str
    dataset: str
    book_name: str
    section: str
    topic: str
    torah_num: int
    passage_num: int
    hebrew_text: str
    en_translation: str
    he_summary: str
    en_summary: str
    keywords: list[str]
    relevance_judgments: list[RelevanceJudgement]


class QueryRequest(BaseModel):
    query: str


# Response models
class FoundPassage(BaseModel):
    location: str
    title: str
    summary: str
    detailedText: str
    relevance: str
    score: float
    reason: str
    stable_id: str
    dataset: str
    keywords: list[str]


class QueryResponse(BaseModel):
    uuid: str
    question: str
    run_parameters: dict[str, Any]
    rewrite_result: dict[str, Any]
    hybrid_params: dict[str, Any]
    timestamp: str
    passages: list[CandidatePassage]
