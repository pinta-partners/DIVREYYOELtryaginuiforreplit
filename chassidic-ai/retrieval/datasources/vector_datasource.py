from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
import os
from pymongo.operations import SearchIndexModel
from pymongo.errors import OperationFailure
from bson.objectid import ObjectId

client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
DATABASE_NAME = "vectors"
COLLECTION_NAME = "embeddings"


class VectorDocument(BaseModel):
    _id: Optional[ObjectId] = None
    dataset: str
    stable_id_in_ds: str
    book_name: str
    vec_heb_text_openai_par: list = Field(default_factory=list)
    vec_heb_text_claude_par: list = Field(default_factory=list)
    origin_doc_update_ts: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_mongo(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_mongo(cls, data: dict) -> "VectorDocument":
        return cls(**data)


class VectorSearchResult(BaseModel):
    doc: VectorDocument
    score: float


class EmbeddingsDatasource:

    def __init__(self, client: AsyncIOMotorClient):
        self.client = client
        self.db = client[DATABASE_NAME]
        self.collection = self.db[COLLECTION_NAME]

    @classmethod
    async def create(cls, client: AsyncIOMotorClient):
        self = cls(client)
        await self.initialize()
        return self

    async def initialize(self):
        await self.init_vector_indexes()

    async def insert_one(self, doc: VectorDocument) -> bool:
        try:
            await self.collection.insert_one(doc.to_mongo())
            return True
        except Exception as e:
            raise Exception(f"Failed to insert document: {e}")

    async def insert_many(self, docs: List[VectorDocument]) -> bool:
        try:
            await self.collection.insert_many([d.to_mongo() for d in docs])
            return True
        except Exception as e:
            raise Exception(f"Failed to insert documents: {e}")

    async def get_all_vectors(self) -> List[VectorDocument]:
        try:
            cursor = self.collection.find()
            documents = []
            async for doc in cursor:
                documents.append(VectorDocument.from_mongo(doc))
            return documents
        except Exception as e:
            raise Exception(f"Failed to retrieve documents: {e}")

    async def find_by_dataset_and_id(
        self, dataset: str, stable_id_in_ds: str
    ) -> Optional[VectorDocument]:
        try:
            doc = await self.collection.find_one(
                {"dataset": dataset, "stable_id_in_ds": stable_id_in_ds}
            )
            return VectorDocument.from_mongo(doc) if doc else None
        except Exception as e:
            raise Exception(f"Failed to find document: {e}")

    async def find_by_knn_heb_text_openai_par(
        self, query_vector: list, k: int = 5
    ) -> List[VectorSearchResult]:
        try:

            pipeline = [
                {
                    "$vectorSearch": {
                        "exact": True,
                        "index": "vec_openai_index",
                        "filter": {},  # Hybrid
                        "limit": k,
                        # "numCandidates": k * 20,
                        "path": "vec_heb_text_openai_par",
                        "queryVector": query_vector,
                    }
                },
                {
                    "$project": {
                        "stable_id_in_ds": 1,
                        "dataset": 1,
                        "book_name": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]

            cursor = self.collection.aggregate(pipeline)
            search_results: list[VectorSearchResult] = []
            async for doc in cursor:
                # print("========" * 10)
                # print(doc)
                # print("========" * 10)
                search_result = VectorSearchResult(
                    doc=VectorDocument.from_mongo(doc),
                    score=doc["score"],
                )
                search_results.append(search_result)

            return search_results
        except Exception as e:
            raise Exception(f"Failed to perform KNN search: {e}")

    async def find_by_knn_heb_text_claude_par(
        self, query_vector: list, k: int = 5
    ) -> List[VectorSearchResult]:
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "exact": True,
                        "index": "vec_claude_index",
                        "filter": {},  # Hybrid
                        "limit": k,
                        # "numCandidates": k * 20,
                        "path": "vec_heb_text_claude_par",
                        "queryVector": query_vector,
                    }
                },
                {
                    "$project": {
                        "stable_id_in_ds": 1,
                        "dataset": 1,
                        "book_name": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]
            cursor = self.collection.aggregate(
                pipeline=pipeline,
            )
            search_results: list[VectorSearchResult] = []
            async for doc in cursor:
                search_result = VectorSearchResult(
                    doc=VectorDocument.from_mongo(doc),
                    score=doc["score"],
                )
                search_results.append(search_result)
            return search_results
        except Exception as e:
            raise Exception(f"Failed to perform KNN search: {e}")

    async def init_vector_indexes(self) -> None:
        """Initialize vector search indexes"""
        try:
            await self.collection.create_index(
                [("dataset", 1), ("stable_id_in_ds", 1)], unique=True
            )

            vector_indexes = await self.collection.list_indexes().to_list(length=None)

            index_models_to_create = []
            # OpenAI vector index
            openai_index_model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "vec_heb_text_openai_par",
                            "numDimensions": 1024,
                            "similarity": "cosine",
                        }
                    ],
                },
                name="vec_openai_index",
                type="vectorSearch",
            )
            if not (any(i["name"] == "vec_openai_index" for i in vector_indexes)):
                index_models_to_create.append(openai_index_model)

            claude_index_model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "vec_heb_text_claude_par",
                            "numDimensions": 1024,
                            "similarity": "cosine",
                        }
                    ],
                },
                name="vec_claude_index",
                type="vectorSearch",
            )
            if not (any(i["name"] == "vec_claude_index" for i in vector_indexes)):
                index_models_to_create.append(claude_index_model)

            try:
                await self.collection.create_search_indexes(
                    models=index_models_to_create,
                )
            except OperationFailure as opfail:
                if "Duplicate Index" in str(opfail):
                    print(
                        "NOTE: not creating vector indexes, these seems to already exist"
                    )
                    pass

        except Exception as e:
            raise Exception(f"Failed to create vector indexes: {e}")
