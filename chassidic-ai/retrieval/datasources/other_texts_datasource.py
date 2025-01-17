from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient

DATABASE_NAME = "other_books"
COLLECTION_NAME = "texts_from_csv"


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


class OtherTextsDatasource:
    def __init__(self, client: AsyncIOMotorClient):
        self.client = client
        # self.db = None
        # self.collection = None

    @classmethod
    async def create(cls, client: AsyncIOMotorClient):
        self = cls(client)
        await self.initialize()
        return self

    async def initialize(self):
        self.db = self.client[DATABASE_NAME]
        if COLLECTION_NAME not in await self.db.list_collection_names():
            await self.db.create_collection(COLLECTION_NAME)
        self.collection = self.db[COLLECTION_NAME]
        await self.collection.create_index("compound_id", unique=True)
        await self.collection.create_index("book_name")
        await self.collection.create_index("keywords")

    async def insert_text(self, text: TextFromCSV) -> None:
        await self.collection.insert_one(text.to_mongo())

    async def insert_many_texts(self, texts: List[TextFromCSV]) -> None:
        mongo_docs = [text.to_mongo() for text in texts]
        await self.collection.insert_many(mongo_docs)

    async def get_all_texts(self) -> List[TextFromCSV]:
        cursor = self.collection.find({})
        docs = await cursor.to_list(length=None)
        return [TextFromCSV.from_mongo(doc) for doc in docs]

    async def find_text_by_id(self, compound_id: str) -> TextFromCSV | None:
        doc = await self.collection.find_one({"compound_id": compound_id})
        return TextFromCSV.from_mongo(doc) if doc else None

    # Find multiple texts by compound_id
    async def find_texts_by_compound_ids(
        self, compound_ids: List[str]
    ) -> List[TextFromCSV]:
        cursor = self.collection.find({"compound_id": {"$in": compound_ids}})
        docs = await cursor.to_list(length=None)
        return [TextFromCSV.from_mongo(doc) for doc in docs]

    async def find_texts_by_keywords(self, keywords: List[str]) -> List[TextFromCSV]:
        cursor = self.collection.find({"keywords": {"$in": keywords}})
        docs = await cursor.to_list(length=None)
        return [TextFromCSV.from_mongo(doc) for doc in docs]
