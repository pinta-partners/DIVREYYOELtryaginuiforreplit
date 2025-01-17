from pydantic import BaseModel
from typing import List
import json

# Example usage:
# # Load and parse file
# question = HebrewQuestion.from_file('34e23ed1-060c-4033-b95e-7e6eaf8559bb.json')

# # Access parsed answer content
# answer = question.parsed_answer
# print(answer.summary_analysis)
# for item in answer.highly_relevant:
#     print(f"Title: {item.title}")


class HighlyRelevantItem(BaseModel):
    title: str
    full_text: str


class AnswerContent(BaseModel):
    summary_analysis: str
    highly_relevant: List[HighlyRelevantItem]


class HebrewQuestion(BaseModel):
    question: str
    answer: str  # Raw JSON string

    @property
    def parsed_answer(self) -> AnswerContent:
        """Parse the nested answer JSON string into AnswerContent object"""
        return AnswerContent.parse_raw(self.answer)

    @classmethod
    def from_file(cls, filepath: str) -> "HebrewQuestion":
        """Load and parse JSON file with UTF-8 encoding"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return cls(**data)
