from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from contextlib import contextmanager
import os

client = MongoClient(os.getenv("MONGO_URI"))


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


@contextmanager
def safe_cursor(cursor):
    try:
        yield cursor
    finally:
        cursor.close()


def insert_one(collection: Collection, doc: VectorDocument) -> bool:
    try:
        collection.insert_one(doc.to_mongo())
        return True
    except PyMongoError as e:
        raise Exception(f"Failed to insert document: {e}")


def insert_many(collection: Collection, docs: List[VectorDocument]) -> bool:
    try:
        collection.insert_many([d.to_mongo() for d in docs])
        return True
    except PyMongoError as e:
        raise Exception(f"Failed to insert documents: {e}")


def get_all(collection: Collection) -> List[VectorDocument]:
    try:
        with safe_cursor(collection.find()) as cursor:
            return [VectorDocument.from_mongo(doc) for doc in cursor]
    except PyMongoError as e:
        raise Exception(f"Failed to retrieve documents: {e}")


def find_by_dataset_and_id(
    collection: Collection, dataset: str, stable_id_in_ds: str
) -> Optional[VectorDocument]:
    try:
        doc = collection.find_one(
            {"dataset": dataset, "stable_id_in_ds": stable_id_in_ds}
        )
        return VectorDocument.from_mongo(doc) if doc else None
    except PyMongoError as e:
        raise Exception(f"Failed to find document: {e}")


def find_by_knn_heb_text_openai_par(
    collection: Collection, query_vector: list, k: int = 5
) -> List[VectorDocument]:
    try:
        pipeline = [
            {
                "$search": {
                    "index": "vec_openai_index",  # Add index name for consistency
                    "knnBeta": {
                        "vector": query_vector,
                        "path": "vec_heb_text_openai_par",
                        "k": k,
                    },
                }
            }
        ]
        with safe_cursor(collection.aggregate(pipeline)) as cursor:
            return [VectorDocument.from_mongo(doc) for doc in cursor]
    except PyMongoError as e:
        raise Exception(f"Failed to perform KNN search: {e}")


def find_by_knn_heb_text_claude_par(
    collection: Collection, query_vector: list, k: int = 5
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
        with safe_cursor(collection.aggregate(pipeline)) as cursor:
            return [VectorDocument.from_mongo(doc) for doc in cursor]
    except PyMongoError as e:
        raise Exception(f"Failed to perform KNN search: {e}")


def init_vector_indexes(collection: Collection) -> None:
    """Initialize vector search indexes"""
    try:
        collection.create_index([("dataset", 1), ("stable_id_in_ds", 1)], unique=True)

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
        collection.database.command(
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
        collection.database.command(
            "createSearchIndex", collection.name, **claude_index
        )

    except PyMongoError as e:
        raise Exception(f"Failed to create vector indexes: {e}")
