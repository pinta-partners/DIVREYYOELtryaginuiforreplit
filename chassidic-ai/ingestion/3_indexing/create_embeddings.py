"""
Creates embeddings for documents in the texts_from_csv collection and stores them in the vector collection.
"""

# Usage
# PYTHONPATH=$(pwd) python -m chassidic-ai.ingestion.3_indexing.create_embeddings

import asyncio
import os
from datetime import datetime
from pydantic import BaseModel, Field

from motor.motor_asyncio import AsyncIOMotorClient
from ...retrieval.embedding.embedders import (
    voyage_3_large_1024_embedder,
    nv_embedder,
    berel_2_embedder,
    openai_3_large_3027_embedder,
)


from ...retrieval.datasources.other_texts_datasource import (
    TextFromCSV,
    OtherTextsDatasource,
)
from ...retrieval.datasources.embeddings_datasource import (
    HybridEmbeddingDoc,
    HybridEmbeddingsColDatasource,
)

# Configuration
EMBEDDINGS_DB_NAME = "embeddings"
BATCH_SIZE = 250


class EmbedderAndCollection:
    def __init__(
        self, embedder, datasource, embedder_name, collection_name, resolution
    ):
        self.embedder = embedder
        self.datasource = datasource
        self.embedder_name = embedder_name
        self.collection_name = collection_name
        self.resolution = resolution


class EmbeddingCandidate(BaseModel):
    """
    Document model for hybrid embedding storage in MongoDB.
    Combines text metadata with vector embeddings.
    """

    text: str

    origin_doc_stable_id: str
    book_name: str
    parasha: str
    topic: str
    torah_num: int = Field(ge=0)
    passage_num: int = Field(ge=0)
    sentence_num: int = Field(ge=0)
    origin_doc_update_ts: datetime


mongo_client = AsyncIOMotorClient(os.getenv("MONGO_URI"))


async def process_documents(
    source_docs: list[EmbeddingCandidate],
    eac: EmbedderAndCollection,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Ensure that the embeddings for the source documents are up-to-date in the vector collection.
    1. Get the embeddings for the source documents
    2. Compare the embeddings with the existing embeddings
    3. Insert new embeddings
    4. Update existing embeddings if the source document has been updated since the last embedding
    """

    # Get all existing vectors, and create a map for quick lookup
    existing_vectors = await eac.datasource.get_all_vectors()
    existing_embeddings_map = {doc.stable_id_in_ds: doc for doc in existing_vectors}

    # Determine which documents need to be inserted or updated
    docs_without_embedding = []
    docs_with_out_of_date_embedding = []

    for doc in source_docs:
        # Docs without embedding are docs that are not in the existing map
        if doc.origin_doc_stable_id not in existing_embeddings_map:
            docs_without_embedding.append(doc)
        else:
            # Docs with out-of-date embedding are docs that have been updated since the last embedding
            existing = existing_embeddings_map[doc.origin_doc_stable_id]
            if existing.origin_doc_update_ts != doc.origin_doc_update_ts:
                docs_with_out_of_date_embedding.append(doc)

    print(
        f"Processing {len(docs_without_embedding)} docs without embedding and {len(docs_with_out_of_date_embedding)} docs with out-of-date embedding"
    )

    all_docs_to_embed = docs_without_embedding + docs_with_out_of_date_embedding

    # Process in batches
    for i in range(0, len(all_docs_to_embed), batch_size):
        batch = all_docs_to_embed[i : i + batch_size]
        print(
            f"Embedding - batch {i//batch_size + 1}/{len(all_docs_to_embed)//batch_size + 1}"
        )
        # Calculate the embeddings
        embeddings = await eac.embedder.embed_passages([doc.text for doc in batch])

        # Upsert the embeddings
        await eac.datasource.upsert_many(
            [
                HybridEmbeddingDoc(
                    stable_id_in_ds=doc.stable_id_in_ds,
                    book_name=doc.book_name,
                    parasha=doc.parasha,
                    topic=doc.topic,
                    torah_num=doc.torah_num,
                    passage_num=doc.passage_num,
                    sentence_num=doc.sentence_num,
                    origin_doc_update_ts=doc.origin_doc_update_ts,
                    embedding_vec=embeddings[i],
                    updated_at=datetime.utcnow(),
                )
                for i, doc in enumerate(batch)
            ]
        )

    # Calculate which vectors need to be deleted since they don't match any document
    vectors_to_delete = []
    for existing in existing_vectors:
        if existing.stable_id_in_ds not in {
            doc.origin_doc_stable_id for doc in source_docs
        }:
            vectors_to_delete.append(existing)

    print(
        f"Deleting {len(vectors_to_delete)} obsolete vectors (have no match in source docs)"
    )
    await eac.datasource.delete_many(vectors_to_delete)


def generate_document_embedding_candidates_from_csv_docs(
    source_docs: list[TextFromCSV],
) -> list[EmbeddingCandidate]:
    """
    Generate document embedding candidates from the source documents.
    """
    return [
        EmbeddingCandidate(
            text=doc.en_translation,
            origin_doc_stable_id=doc.compound_id,
            book_name=doc.book_name,
            parasha=doc.section,
            topic=doc.topic,
            torah_num=doc.torah_num,
            passage_num=doc.passage_num,
            sentence_num=0,
            origin_doc_update_ts=doc.updated_at,
        )
        for doc in source_docs
    ]


def generate_sentence_embedding_candidates_from_docs(
    source_docs: list[TextFromCSV],
) -> list[EmbeddingCandidate]:
    """
    Generate sentence embedding candidates from the source documents.
    """
    candidates = []
    for doc in source_docs:
        sentences = doc.en_translation.split(".")
        for i, sentence in enumerate(sentences):
            candidates.append(
                EmbeddingCandidate(
                    text=sentence,
                    origin_doc_stable_id=doc.compound_id,
                    book_name=doc.book_name,
                    parasha=doc.section,
                    topic=doc.topic,
                    torah_num=doc.torah_num,
                    passage_num=doc.passage_num,
                    sentence_num=i,
                    origin_doc_update_ts=doc.updated_at,
                )
            )
    return candidates


async def process_csv_datasource_embedders(
    mongo_client: AsyncIOMotorClient,
    embedders_and_collections: dict[str, EmbedderAndCollection],
) -> list[asyncio.Task]:
    """
    Process the CSV datasource with the given embedders and collections.
    """

    # Initialize collections
    texts_from_csv_ds = await OtherTextsDatasource.create(mongo_client)

    # TODO: Different datasets - this logic is for the CSV based dataset
    source_docs = await texts_from_csv_ds.get_all_texts()
    document_embeddings_candidates = (
        generate_document_embedding_candidates_from_csv_docs(
            source_docs=source_docs,
        )
    )
    sentence_embeddings_candidates = generate_sentence_embedding_candidates_from_docs(
        source_docs=source_docs,
    )

    # For each embedder, process the documents
    tasks = []
    for key, eac in embedders_and_collections.items():
        candidates = (
            document_embeddings_candidates
            if "document" in eac.resolution
            else sentence_embeddings_candidates
        )

        tasks.append(
            process_documents(
                source_docs=candidates,
                eac=eac,
            )
        )

    return tasks


async def main():
    # Create embeddings for sentence and passage levels
    # using the four embedders:
    embedders = {
        "voyage_3_large_1024": voyage_3_large_1024_embedder,
        "nv": nv_embedder,
        "berel_2": berel_2_embedder,
        "openai_3_large_3027": openai_3_large_3027_embedder,
    }

    embedders_and_collections = {}

    # Create the datasources and register them along with collection names and embedders
    for key, embedder in embedders.items():
        for resolution in ["document", "sentence"]:
            collection_name = f"{resolution}_{key}"
            datasource = await HybridEmbeddingsColDatasource.create(
                client=mongo_client,
                db_name=EMBEDDINGS_DB_NAME,
                collection_name=collection_name,
                embedder=embedder,
            )
            embedders_and_collections[key] = EmbedderAndCollection(
                embedder=embedder,
                datasource=datasource,
                embedder_name=key,
                collection_name=collection_name,
                resolution=resolution,
            )

    # Load and convert all documents
    tasks = []

    # ************************************************************************
    # CSV Texts pipeline
    # ************************************************************************
    tasks.extend(
        await process_csv_datasource_embedders(
            mongo_client=mongo_client,
            embedders_and_collections=embedders_and_collections,
        )
    )

    # ************************************************************************
    # Chassidic Sentences pipeline
    # ************************************************************************

    # ************************************************************************
    # Sefaria pipeline
    # ************************************************************************

    # Wait for all tasks to complete
    res = await asyncio.gather(*tasks)

    print("Processing complete")


if __name__ == "__main__":
    asyncio.run(main())
