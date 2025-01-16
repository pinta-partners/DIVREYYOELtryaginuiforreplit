from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
import os

client = AsyncIOMotorClient(os.getenv("MONGO_URI"))


class VectorDocument(BaseModel):
    dataset: str
    stable_id_in_ds: str
    book_name: str
    vec_heb_text_openai_par: list = Field(default_factory=list)
    vec_heb_text_claude_par: list = Field(default_factory=list)
    origin_doc_update_ts: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_mongo(self) -> dict:
        return self.dict()

    @classmethod
    def from_mongo(cls, data: dict) -> "VectorDocument":
        return cls(**data)


async def insert_one(collection: AsyncIOMotorCollection, doc: VectorDocument) -> bool:
    try:
        await collection.insert_one(doc.to_mongo())
        return True
    except Exception as e:
        raise Exception(f"Failed to insert document: {e}")


async def insert_many(
    collection: AsyncIOMotorCollection, docs: List[VectorDocument]
) -> bool:
    try:
        await collection.insert_many([d.to_mongo() for d in docs])
        return True
    except Exception as e:
        raise Exception(f"Failed to insert documents: {e}")


async def get_all(collection: AsyncIOMotorCollection) -> List[VectorDocument]:
    try:
        cursor = collection.find()
        documents = []
        async for doc in cursor:
            documents.append(VectorDocument.from_mongo(doc))
        return documents
    except Exception as e:
        raise Exception(f"Failed to retrieve documents: {e}")


async def find_by_dataset_and_id(
    collection: AsyncIOMotorCollection, dataset: str, stable_id_in_ds: str
) -> Optional[VectorDocument]:
    try:
        doc = await collection.find_one(
            {"dataset": dataset, "stable_id_in_ds": stable_id_in_ds}
        )
        return VectorDocument.from_mongo(doc) if doc else None
    except Exception as e:
        raise Exception(f"Failed to find document: {e}")


async def find_by_knn_heb_text_openai_par(
    collection: AsyncIOMotorCollection, query_vector: list, k: int = 5
) -> List[VectorDocument]:
    try:
        pipeline = [
            {
                "$search": {
                    "index": "vec_openai_index",
                    "knnBeta": {
                        "vector": query_vector,
                        "path": "vec_heb_text_openai_par",
                        "k": k,
                    },
                }
            }
        ]
        cursor = collection.aggregate(pipeline)
        documents = []
        async for doc in cursor:
            documents.append(VectorDocument.from_mongo(doc))
        return documents
    except Exception as e:
        raise Exception(f"Failed to perform KNN search: {e}")


async def find_by_knn_heb_text_claude_par(
    collection: AsyncIOMotorCollection, query_vector: list, k: int = 5
) -> List[VectorDocument]:
    try:
        pipeline = [
            {
                "$search": {
                    "index": "vec_claude_index",
                    "knnBeta": {
                        "vector": query_vector,
                        "path": "vec_heb_text_claude_par",
                        "k": k,
                    },
                }
            }
        ]
        cursor = collection.aggregate(pipeline)
        documents = []
        async for doc in cursor:
            documents.append(VectorDocument.from_mongo(doc))
        return documents
    except Exception as e:
        raise Exception(f"Failed to perform KNN search: {e}")


async def init_vector_indexes(collection: AsyncIOMotorCollection) -> None:
    """Initialize vector search indexes"""
    try:
        await collection.create_index(
            [("dataset", 1), ("stable_id_in_ds", 1)], unique=True
        )

        # OpenAI vector index
        openai_index = {
            "name": "vec_openai_index",
            "definition": {
                "mappings": {
                    "dynamic": True,
                    "fields": {
                        "vec_heb_text_openai_par": {
                            "type": "knnVector",
                            "dimensions": 1536,
                        }
                    },
                }
            },
        }
        await collection.database.command(
            "createSearchIndex", collection.name, **openai_index
        )

        # Claude vector index
        claude_index = {
            "name": "vec_claude_index",
            "definition": {
                "mappings": {
                    "dynamic": True,
                    "fields": {
                        "vec_heb_text_claude_par": {
                            "type": "knnVector",
                            "dimensions": 1536,
                        }
                    },
                }
            },
        }
        await collection.database.command(
            "createSearchIndex", collection.name, **claude_index
        )

    except Exception as e:
        raise Exception(f"Failed to create vector indexes: {e}")
