from typing import List
from pydantic import BaseModel
import json


class Question(BaseModel):
    question: str
    answers: List[str]
    relevant_passage_refs: List[str]
    relevant_passage_texts: List[str]
    dicta_output_file: str


class ExperimentSpecification(BaseModel):
    experiment_name: str
    questions: List[Question]


def load_from_json(json_str: str) -> ExperimentSpecification:
    data = json.loads(json_str)
    return ExperimentSpecification(**data)
