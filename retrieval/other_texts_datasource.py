from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
from pymongo import MongoClient


class TextFromCSV(BaseModel):
    compound_id: str
    book_name: str
    section: str
    topic: str
    torah_num: int
    passage_num: int
    hebrew_text: str
    en_translation: str
    he_summary: str
    en_summary: str
    keywords: List[str]
    updated_at: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_mongo(self) -> dict:
        return self.dict()

    @classmethod
    def from_mongo(cls, data: dict) -> "TextFromCSV":
        return cls(**data)


def insert_text(collection, text: TextFromCSV) -> None:
    collection.insert_one(text.to_mongo())


def insert_many_texts(collection, texts: List[TextFromCSV]) -> None:
    mongo_docs = [text.to_mongo() for text in texts]
    collection.insert_many(mongo_docs)


def get_all_texts(collection) -> List[TextFromCSV]:
    cursor = collection.find({})
    return [TextFromCSV.from_mongo(doc) for doc in cursor]


def find_text_by_id(collection, compound_id: str) -> TextFromCSV | None:
    doc = collection.find_one({"compound_id": compound_id})
    return TextFromCSV.from_mongo(doc) if doc else None


def find_texts_by_keywords(collection, keywords: List[str]) -> List[TextFromCSV]:
    cursor = collection.find({"keywords": {"$in": keywords}})
    return [TextFromCSV.from_mongo(doc) for doc in cursor]


def init_texts_from_csv_collection(client: MongoClient):
    db = client["other_books"]
    coll_name = "texts_from_csv"
    if coll_name not in db.list_collection_names():
        db.create_collection(coll_name)
    collection = db[coll_name]
    collection.create_index("compound_id", unique=True)
    collection.create_index("book_name")
    collection.create_index("keywords")
    return collection
