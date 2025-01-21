from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne
from pymongo.operations import SearchIndexModel
from pymongo.errors import OperationFailure
from bson.objectid import ObjectId
from retrieval.embedding.embedders import BaseEmbedder
import logging

logger = logging.getLogger(__name__)


class HybridEmbeddingDoc(BaseModel):
    """
    Document model for hybrid embedding storage in MongoDB.
    Combines text metadata with vector embeddings.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _id: Optional[ObjectId] = Field(default=None, alias="_id")
    stable_id_in_ds: str = Field(..., description="Stable identifier in dataset")
    book_name: str
    parasha: str
    topic: str
    torah_num: int = Field(ge=0)
    passage_num: int = Field(ge=0)
    sentence_num: int = Field(ge=0)
    embedding_vec: List[float] = Field(default_factory=list)
    origin_doc_update_ts: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_mongo(self) -> dict:
        """Convert the model to a MongoDB-compatible dictionary."""
        return self.model_dump(by_alias=True)

    @classmethod
    def from_mongo(cls, data: dict) -> "HybridEmbeddingDoc":
        return cls(**data)


class HybridEmbeddingSearchResult(BaseModel):
    doc: HybridEmbeddingDoc
    score: float


class HybridEmbeddingsColDatasource:

    def __init__(
        self,
        client: AsyncIOMotorClient,
        db_name: str,
        collection_name: str,
        embedder: BaseEmbedder,
    ) -> None:
        self.client = client
        self.db = client[db_name]
        self.collection = self.db[collection_name]
        self.collection_name = collection_name
        self.embedder = embedder
        self.pure_vec_idx_name = f"{collection_name}_vec_idx"
        self.hybrid_vec_idx_name = f"{collection_name}_hybrid_idx"

    @classmethod
    async def create(
        cls,
        client: AsyncIOMotorClient,
        db_name: str,
        collection_name: str,
        embedder: BaseEmbedder,
    ):
        self = cls(client, db_name, collection_name, embedder)
        await self.initialize(embedder.dim)
        return self

    async def initialize(self, embedding_dim: int) -> None:
        await self._init_vector_indexes(embedding_dim)

    async def insert_one(self, doc: HybridEmbeddingDoc) -> bool:
        try:
            await self.collection.insert_one(doc.to_mongo())
            return True
        except Exception as e:
            raise Exception(f"Failed to insert document: {e}")

    async def insert_many(self, docs: list[HybridEmbeddingDoc]) -> bool:
        try:
            await self.collection.insert_many([d.to_mongo() for d in docs])
            return True
        except Exception as e:
            raise Exception(f"Failed to insert documents: {e}")

    async def upsert_many(self, docs: list[HybridEmbeddingDoc]) -> bool:
        try:
            operations = []
            for doc in docs:
                operations.append(
                    UpdateOne(
                        {"stable_id_in_ds": doc.stable_id_in_ds},
                        {"$set": doc.to_mongo()},
                        upsert=True,
                    )
                )
            await self.collection.bulk_write(operations)
            return True
        except Exception as e:
            raise Exception(f"Failed to upsert documents: {e}")

    async def get_all_vectors(self) -> list[HybridEmbeddingDoc]:
        try:
            cursor = self.collection.find()
            documents = []
            async for doc in cursor:
                documents.append(HybridEmbeddingDoc.from_mongo(doc))
            return documents
        except Exception as e:
            raise Exception(f"Failed to retrieve documents: {e}")

    async def find_by_dataset_and_id(
        self, dataset: str, stable_id_in_ds: str
    ) -> Optional[HybridEmbeddingDoc]:
        try:
            doc = await self.collection.find_one(
                {"dataset": dataset, "stable_id_in_ds": stable_id_in_ds}
            )
            return HybridEmbeddingDoc.from_mongo(doc) if doc else None
        except Exception as e:
            raise Exception(f"Failed to find document: {e}")

    async def find_by_query_similarity(
        self,
        query: str,
        k: int = 50,
        book_names: list[str] = [],
        parasha_names: list[str] = [],
    ) -> list[HybridEmbeddingSearchResult]:
        try:
            # Embed the query
            query_vector = await self.embedder.embed_query(text=query, instruction=None)
            is_hybrid = len(book_names) > 0 or len(parasha_names) > 0
            if is_hybrid:
                # Create a mongo filter to search by.
                # If book_names_filter is not empty, then the book name must be one of its values.
                # If parasha_names_filter is not empty, then the parasha name must be one of its values.
                book_names_filter = {"book_name": {"$in": book_names}}
                parasha_names_filter = {"parasha": {"$in": parasha_names}}

                filter = {
                    "$or": [
                        book_names_filter if book_names else {},
                        parasha_names_filter if parasha_names else {},
                    ]
                }
            else:
                filter = {}

            pipeline = [
                {
                    "$vectorSearch": {
                        "exact": True,  # When set to False, set numCandidates
                        "index": (
                            self.hybrid_vec_idx_name
                            if is_hybrid
                            else self.pure_vec_idx_name
                        ),
                        "filter": filter,
                        "limit": k,
                        # "numCandidates": k * 20,
                        "path": "embedding_vec",
                        "queryVector": query_vector,
                    }
                },
                {
                    "$project": {
                        "stable_id_in_ds": 1,
                        "dataset": 1,
                        "book_name": 1,
                        "parasha": 1,
                        "topic": 1,
                        "torah_num": 1,
                        "passage_num": 1,
                        "sentence_num": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]

            cursor = self.collection.aggregate(pipeline)
            search_results: list[HybridEmbeddingSearchResult] = []
            async for doc in cursor:
                search_result = HybridEmbeddingSearchResult(
                    doc=HybridEmbeddingDoc.from_mongo(doc),
                    score=doc["score"],
                )
                search_results.append(search_result)

            return search_results
        except Exception as e:
            raise Exception(f"Failed to perform KNN search: {e}")

    async def _init_vector_indexes(self, embedding_dim: int) -> None:
        """Initialize vector search indexes"""
        try:
            # Index to search for a document by its stable_id_in_ds
            await self.collection.create_index([("stable_id_in_ds", 1)], unique=True)

            # Index to search for a document by book and parasha
            await self.collection.create_index([("book_name", 1), ("parasha", 1)])

            vector_indexes = await self.collection.list_indexes().to_list(length=None)

            index_models_to_create = []

            # Pure vector index
            pure_vec_index_model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding_vec",
                            "numDimensions": embedding_dim,
                            "similarity": "dotProduct",
                        }
                    ],
                },
                name=self.pure_vec_idx_name,
                type="vectorSearch",
            )
            if not (any(i["name"] == self.pure_vec_idx_name for i in vector_indexes)):
                index_models_to_create.append(pure_vec_index_model)

            # Hybrid vector index
            # This index allows searching by book and parasha, and then filtering by the vector similarity
            hybrid_index_model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding_vec",
                            "numDimensions": embedding_dim,
                            "similarity": "dotProduct",
                        },
                        {"type": "string", "path": "book_name"},
                        {"type": "string", "path": "parasha"},
                    ],
                },
                name=self.hybrid_vec_idx_name,
                type="vectorSearch",
            )
            if not (any(i["name"] == self.hybrid_vec_idx_name for i in vector_indexes)):
                index_models_to_create.append(hybrid_index_model)

            try:
                await self.collection.create_search_indexes(
                    models=index_models_to_create,
                )
            except OperationFailure as opfail:
                if "Duplicate Index" in str(opfail):
                    logger.info("Vector indexes already exist, skipping creation")
                else:
                    raise OperationFailure(f"Failed to create vector indexes: {opfail}")
            except Exception as e:
                raise RuntimeError(
                    f"Unexpected error creating vector indexes: {e}"
                ) from e

        except Exception as e:
            raise Exception(f"Failed to create vector indexes: {e}")
