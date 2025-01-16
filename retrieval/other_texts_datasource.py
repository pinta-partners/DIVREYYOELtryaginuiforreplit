from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient


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
        return self.model_dump()

    @classmethod
    def from_mongo(cls, data: dict) -> "TextFromCSV":
        return cls(**data)


async def insert_text(collection, text: TextFromCSV) -> None:
    await collection.insert_one(text.to_mongo())


async def insert_many_texts(collection, texts: List[TextFromCSV]) -> None:
    mongo_docs = [text.to_mongo() for text in texts]
    await collection.insert_many(mongo_docs)


async def get_all_texts(collection) -> List[TextFromCSV]:
    cursor = collection.find({})
    docs = await cursor.to_list(length=None)
    return [TextFromCSV.from_mongo(doc) for doc in docs]


async def find_text_by_id(collection, compound_id: str) -> TextFromCSV | None:
    doc = await collection.find_one({"compound_id": compound_id})
    return TextFromCSV.from_mongo(doc) if doc else None


async def find_texts_by_keywords(collection, keywords: List[str]) -> List[TextFromCSV]:
    cursor = collection.find({"keywords": {"$in": keywords}})
    docs = await cursor.to_list(length=None)
    return [TextFromCSV.from_mongo(doc) for doc in docs]


async def init_texts_from_csv_collection(client: AsyncIOMotorClient):
    db = client["other_books"]
    coll_name = "texts_from_csv"
    if coll_name not in await db.list_collection_names():
        await db.create_collection(coll_name)
    collection = db[coll_name]
    await collection.create_index("compound_id", unique=True)
    await collection.create_index("book_name")
    await collection.create_index("keywords")
    return collection
