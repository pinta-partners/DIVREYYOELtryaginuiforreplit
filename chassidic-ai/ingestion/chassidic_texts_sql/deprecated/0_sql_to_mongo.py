import re
import ast
import asyncio
import argparse
import os
from pathlib import Path
from typing import Generator, List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from datetime import datetime, timezone

BATCH_SIZE = 100  # Maximum number of records per bulk insert


class TextDocument(BaseModel):
    id: str  # UUID as string
    chassidus_text_id: str  # UUID as string
    sentence: str
    sentence_number: int
    context: Optional[str]
    paragraph: str  # Convert to string
    source: str
    tags: List[str]  # Parse JSON array
    sefaria_name: str
    created_at: datetime
    updated_at: datetime
    translation: str
    translation_version: str  # Convert to string
    embedding_large_english_hebrew: List[float] = Field(default_factory=list)


def parse_tags(tags_str: str) -> List[str]:
    if not tags_str or tags_str == "NULL":
        return []
    try:
        # Remove PostgreSQL array syntax and parse
        clean_str = tags_str.strip("{}").replace('"', "")
        return [tag.strip() for tag in clean_str.split(",") if tag.strip()]
    except Exception as e:
        print(f"WARN: Using default due to error parsing tags: {e}")
        return []


def parse_datetime(dt_str: str) -> datetime:
    if not dt_str or dt_str == "NULL":
        return datetime.now(timezone.utc)
    try:
        # Remove timezone info for now
        dt_str = dt_str.split("+")[0]
        return datetime.fromisoformat(dt_str)
    except Exception as e:
        print(f"WARN: Using default due to error parsing datetime: {e}")
        return datetime.now(timezone.utc)


def safe_int(value, default=0):
    if not value or value == "NULL":
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        print(f"Warning: Could not convert {value} to int, using default {default}")
        return default


def read_sql_rows(sql_file_path: str) -> Generator[TextDocument, None, None]:
    # Modified pattern to handle multi-line statements and different quote styles
    insert_pattern = re.compile(
        r"INSERT\s+INTO\s+.*?\s+VALUES\s*"  # Match INSERT INTO ... VALUES
        r"\("  # Opening parenthesis
        r"(.*?)"  # Capture values non-greedy
        r"\)"  # Closing parenthesis
        r"\s*;",  # Optional whitespace and semicolon
        re.DOTALL | re.IGNORECASE,
    )

    # Read entire file content for multi-line statements
    with open(sql_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all INSERT statements
    matches = insert_pattern.finditer(content)

    for match in matches:
        values_part = match.group(1)
        # Split into rows, handling escaped commas
        rows = re.findall(r"\((.*?)\)(?:,|$)", values_part)

        print(f"Found {len(rows)} rows to process")

        for row in rows:
            # Split fields, preserving escaped commas
            fields = re.split(r",(?=(?:[^']*'[^']*')*[^']*$)", row)
            field_values = []

            for field in fields:
                field = field.strip().strip("'")  # Remove quotes
                if field.upper() == "NULL":
                    field_values.append(None)
                else:
                    try:
                        value = ast.literal_eval(field)
                        field_values.append(value)
                    except (ValueError, SyntaxError):
                        field_values.append(field)

            yield TextDocument(
                id=str(field_values[0]),
                chassidus_text_id=str(field_values[1]),
                sentence=field_values[2],
                sentence_number=safe_int(field_values[3]),
                context=field_values[4],
                paragraph=str(field_values[5]),
                source=field_values[6],
                tags=parse_tags(field_values[7]),
                sefaria_name=field_values[8],
                created_at=parse_datetime(field_values[9]),
                updated_at=parse_datetime(field_values[10]),
                translation=field_values[11],
                translation_version=str(field_values[12]),
                embedding_large_english_hebrew=(
                    field_values[13] if field_values[13] else []
                ),
            )


async def batch_insert(
    collection, documents: List[TextDocument], total_processed: int, total_lines: int
):
    result = await collection.insert_many([doc.dict() for doc in documents])
    progress = (total_processed / total_lines) * 100
    print(
        f"Progress: {progress:.1f}% - Inserted batch of {len(result.inserted_ids)} documents"
    )
    return result


async def process_sql_file(
    sql_file_path: str, mongo_uri: str, db_name: str, collection_name: str
):
    # Initialize MongoDB connection
    client = AsyncIOMotorClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Count total inserts for progress
    print("Counting total INSERT statements...")
    with open(sql_file_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for line in f if line.strip().startswith("INSERT"))

    print(f"Total INSERT statements found: {total_lines}")

    documents = []
    total_processed = 0

    print("Processing SQL file...")
    for document in read_sql_rows(sql_file_path):
        documents.append(document)
        total_processed += 1

        if len(documents) >= BATCH_SIZE:
            print("Flushing documents")
            await batch_insert(collection, documents, total_processed, total_lines)
            documents = []

    # Insert remaining documents
    if documents:
        await batch_insert(collection, documents, total_processed, total_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SQL file to MongoDB")
    parser.add_argument("sql_file", help="Path to the SQL file to import")
    args = parser.parse_args()

    # Get MongoDB URI from environment variable, fallback to localhost if not set
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")

    # Extract collection name from SQL filename without extension
    sql_path = Path(args.sql_file)
    collection_name = sql_path.stem

    db_name = "sql_imports"

    # Validate SQL file exists
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {args.sql_file}")

    asyncio.run(process_sql_file(args.sql_file, mongo_uri, db_name, collection_name))
