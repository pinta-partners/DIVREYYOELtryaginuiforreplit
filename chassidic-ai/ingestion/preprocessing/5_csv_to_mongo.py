import asyncio
import argparse
import os
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne
from ...retrieval.datasources.csv_datasource import DataSource, Passage
from ...retrieval.datasources.other_texts_datasource import (
    TextFromCSV,
    # TODO: Use OtherTextsDatasource and move the bulk insert to there
)

TARGET_DB = "other_books"
TARGET_COLLECTION = "texts_from_csv"


# Usage
# PYTHONPATH=$(pwd) python -m chassidic-ai.ingestion.preprocessing.5_csv_to_mongo --csv_path data/dataset.csv


def convert_passage_to_db_doc(passage: Passage) -> TextFromCSV:
    compound_id = f"{passage.book_name}_{passage.section}_{passage.topic}_{passage.torah_number}_{passage.passage_number}"
    keywords = (
        [k.strip() for k in passage.keywords.split("\n")] if passage.keywords else []
    )

    return TextFromCSV(
        compound_id=compound_id,
        book_name=passage.book_name,
        section=passage.section,
        topic=passage.topic,
        torah_num=int(passage.torah_number) if passage.torah_number.isdigit() else 0,
        passage_num=(
            int(passage.passage_number) if passage.passage_number.isdigit() else 0
        ),
        hebrew_text=passage.hebrew_text,
        en_translation=passage.translation,
        he_summary="",  # Note: Original CSV doesn't have separate Hebrew summary
        en_summary=passage.summary,
        keywords=keywords,
        updated_at=datetime.now(tz=datetime.now().astimezone().tzinfo),
    )


async def process_csv_to_mongo(csv_path: str, batch_size: int = 100):
    # Read CSV
    datasource = DataSource(csv_path)
    datasource.read_csv()
    print("Read CSV - complete. Found {} passages".format(len(datasource.passages)))

    # Setup MongoDB
    client = AsyncIOMotorClient(host=os.getenv("MONGO_URI"))
    db = client[TARGET_DB]
    collection = db[TARGET_COLLECTION]

    operations = []
    for idx, passage in enumerate(datasource.passages):
        doc = convert_passage_to_db_doc(passage).to_mongo()
        print(
            f"Processing document {idx + 1}/{len(datasource.passages)}: {doc['compound_id']}"
        )

        # Add dynamic fields
        doc.update(
            {
                "updated_at": datetime.now(tz=datetime.now().astimezone().tzinfo),
                "created_at": datetime.now(tz=datetime.now().astimezone().tzinfo),
                "he_summary": "",  # Default empty Hebrew summary
            }
        )

        operations.append(
            UpdateOne({"compound_id": doc["compound_id"]}, {"$set": doc}, upsert=True)
        )

        if len(operations) >= batch_size:
            result = await collection.bulk_write(operations)
            print(f"Inserted {result.upserted_count} documents")
            operations = []

    # Process remaining operations
    if operations:
        result = await collection.bulk_write(operations)
        print(f"Inserted {result.upserted_count} documents")

    client.close()


def main():
    parser = argparse.ArgumentParser(description="Upload CSV data to MongoDB")
    parser.add_argument(
        "--csv_path", help="Path to the input CSV file", default="data/dataset.csv"
    )
    args = parser.parse_args()

    asyncio.run(process_csv_to_mongo(args.csv_path))


if __name__ == "__main__":
    main()
