"""
Creates embeddings for documents in the texts_from_csv collection and stores them in the vector collection.
"""

# Usage
# PYTHONPATH=$(pwd) python -m chassidic-ai.ingestion.preprocessing.6_create_embeddings

import asyncio
import os
from datetime import datetime
from litellm import aembedding

# from litellm.types.utils import EmbeddingResponse
# from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReplaceOne

# import voyageai
# import voyageai.client
# import voyageai.client_async


from ...retrieval.datasources.other_texts_datasource import (
    TextFromCSV,
    OtherTextsDatasource,
)
from ...retrieval.datasources.vector_datasource import (
    VectorDocument,
    EmbeddingsDatasource,
)

# Configuration
BATCH_SIZE = 250
mongo_client = AsyncIOMotorClient(os.getenv("MONGO_URI"))

# Anthropic client
# voyageai_client = voyageai.client_async.AsyncClient()


async def process_documents(
    source_docs: list[TextFromCSV],
    existing_vectors: list[VectorDocument],
    batch_size: int = BATCH_SIZE,
) -> tuple[list[VectorDocument], list[VectorDocument]]:
    """Process documents in batches and return new/updated vectors"""
    new_docs = []
    updated_docs = []
    existing_map = {
        f"{doc.dataset}_{doc.stable_id_in_ds}": doc for doc in existing_vectors
    }

    # Process in batches
    for i in range(0, len(source_docs), batch_size):
        batch = source_docs[i : i + batch_size]
        print(
            f"Processing batch {i//batch_size + 1}/{len(source_docs)//batch_size + 1}"
        )

        texts = [doc.hebrew_text for doc in batch]
        # Get OpenAI embeddings for the batch
        t1 = aembedding(
            model="openai/text-embedding-3-large",
            input=texts,
            dimensions=1024,
        )

        # t2 = voyageai_client.embed(
        #     texts,
        #     model="voyage-3-large",
        #     input_type="document",
        # )

        # # Get Claude embeddngs for the batch
        # t2 = aembedding(
        #     model="voyage-3-large",
        #     input=texts,
        #     input_type="document",
        #     # output_dimension=1024,
        #     # dimensions=1024,
        #     custom_llm_provider="anthropic",
        # )

        # Wait for the embeddings
        embedding_tasks = [t1]  # ,t2
        embeddings = await asyncio.gather(*embedding_tasks)

        for idx, doc in enumerate(batch):
            vector_id = f"texts_from_csv_{doc.compound_id}"
            t1_embeddings = embeddings[0].data
            # t2_embeddings = embeddings[1].embeddings

            vector_doc = VectorDocument(
                dataset="texts_from_csv",
                stable_id_in_ds=doc.compound_id,
                book_name=doc.book_name,
                vec_heb_text_openai_par=t1_embeddings[idx]["embedding"],
                vec_heb_text_claude_par=[],
                # vec_heb_text_claude_par=t2_embeddings[idx],
                origin_doc_update_ts=doc.updated_at,
                updated_at=datetime.now(tz=datetime.now().astimezone().tzinfo),
            )

            if vector_id in existing_map:
                existing = existing_map[vector_id]
                if existing.origin_doc_update_ts != doc.updated_at:
                    updated_docs.append(vector_doc)
            else:
                new_docs.append(vector_doc)

    return new_docs, updated_docs


async def main():
    # Initialize collections
    texts_from_csv_ds = await OtherTextsDatasource.create(mongo_client)
    embeddings_ds = await EmbeddingsDatasource.create(mongo_client)

    # Get all documents
    source_docs = await texts_from_csv_ds.get_all_texts()
    existing_vectors = await embeddings_ds.get_all_vectors()

    # Process documents
    new_docs, updated_docs = await process_documents(source_docs, existing_vectors)

    # Batch insert new documents
    if new_docs:
        await embeddings_ds.insert_many(new_docs)
        print(f"Inserted {len(new_docs)} new vector documents")

    # Batch update existing documents
    if updated_docs:
        operations = [
            ReplaceOne(
                {"dataset": doc.dataset, "stable_id_in_ds": doc.stable_id_in_ds},
                doc.to_mongo(),
            )
            for doc in updated_docs
        ]
        await embeddings_ds.collection.bulk_write(operations)
        print(f"Updated {len(updated_docs)} existing vector documents")


if __name__ == "__main__":
    asyncio.run(main())
