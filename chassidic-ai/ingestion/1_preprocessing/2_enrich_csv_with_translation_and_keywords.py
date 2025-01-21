import asyncio
import csv
import os
import re
import sys
from litellm import acompletion

# Example how to run:
# python chassidic-ai/ingestion/preprocessing/2_enrich_csv_with_translation_and_keywords.py data/raw_input_csv/divrey_yoel_shemot.csv data/enriched/divrey_yoel_shemot.csv 10
# python chassidic-ai/ingestion/preprocessing/2_enrich_csv_with_translation_and_keywords.py data/raw_input_csv/divrey_yoel_bo.csv data/enriched/divrey_yoel_bo.csv 10
# python chassidic-ai/ingestion/preprocessing/2_enrich_csv_with_translation_and_keywords.py data/raw_input_csv/divrey_yoel_beshalach.csv data/enriched/divrey_yoel_beshalach.csv 10
# python chassidic-ai/ingestion/preprocessing/2_enrich_csv_with_translation_and_keywords.py data/raw_input_csv/divrey_yoel_vaera.csv data/enriched/divrey_yoel_vaera.csv 10


class HebrewTextProcessor:

    def __init__(self, max_concurrent: int):
        self.max_concurrent = max_concurrent  # Bounded parallelism

    async def call_litellm(self, prompt: str) -> str:
        """Make an asynchronous call to LiteLLM with a prompt."""
        try:
            messages = [{"role": "user", "content": prompt}]
            model = "gpt-4o-mini"  # "claude-3-5-sonnet-20241022",
            response = await acompletion(
                model=model,
                messages=messages,
                max_tokens=1500,
                temperature=0,
                num_retries=3,
            )
            assistant_reply = response.choices[0].message.content.strip()
            return assistant_reply
        except Exception as e:
            print(f"API error: {e}")
            return "API call failed"

    async def translate_text(self, hebrew_text: str) -> str:
        """Translate Hebrew text to English using the exact prompt."""
        prompt = f"""Please translate the following Hebrew text into English. This is from Sefer Divrey Yoel:

        Hebrew text: {hebrew_text}

        Provide the translation, preserving Hasidic concepts and terminology. Output only the translation - do not add any explanations or comments."""
        return await self.call_litellm(prompt)

    async def generate_summary(self, hebrew_text: str) -> str:
        """Generate a summary of the Hebrew text using the exact prompt."""
        prompt = f"""Please read and summarize the following Hebrew passage from Sefer Divrey Yoel.
        Focus on the key Hasidic concepts and theological insights:

        Hebrew text: {hebrew_text}

        Provide a clear 3-4 sentence summary in English that captures the theological depth. Output only the English summary - do not add any explanations or comments."""
        return await self.call_litellm(prompt)

    async def generate_keywords(self, hebrew_text: str) -> str:
        """Extract key Hebrew/Jewish theological terms using the exact prompt."""
        prompt = f"""Please extract 10 key Hebrew/Jewish theological terms from this Hebrew passage:

        Hebrew text: {hebrew_text}

        List exactly 10 terms, one per line, focusing on Hasidic and Kabbalistic concepts. Output only the terms - do not add any explanations or comments and do not number the terms."""
        return await self.call_litellm(prompt)

    async def process_passage(self, passage: dict[str, str]) -> dict[str, str]:
        """Enrich a single passage. Returns the updated passage dict."""
        # Strip out any HTML tags
        hebrew_text = re.sub(r"<[^>]*>", "", passage["passage_content"])
        passage["passage_content"] = hebrew_text

        print(
            f"Processing: {passage['book_name']} - {passage['parsha_name']} "
            f"- Torah #{passage['dvar_torah_id']} - Passage #{passage['passage_id']}"
        )

        try:
            # Process asynchronously
            translation_task = asyncio.create_task(self.translate_text(hebrew_text))
            summary_task = asyncio.create_task(self.generate_summary(hebrew_text))
            keywords_task = asyncio.create_task(self.generate_keywords(hebrew_text))

            translation, summary, keywords = await asyncio.gather(
                translation_task, summary_task, keywords_task
            )

            # Enrich the passage
            passage["translation"] = translation
            passage["summary"] = summary
            passage["keywords"] = keywords

        except Exception as e:
            # On error, keep the fields unfilled
            print(f"Error processing passage: {e}")

        return passage


async def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python enrich_csv_incremental.py <input_csv> <output_csv> <max_concurrent>"
        )
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    max_concurrent = int(sys.argv[3])

    # 1. If output CSV doesn't exist, create it with extra fields
    if not os.path.exists(output_csv):
        with (
            open(input_csv, "r", encoding="utf-8") as infile,
            open(output_csv, "w", encoding="utf-8", newline="") as outfile,
        ):
            reader = csv.DictReader(infile)
            if not reader.fieldnames:
                raise ValueError("Input CSV has no header row")
            fieldnames = list(reader.fieldnames) + [
                "translation",
                "summary",
                "keywords",
            ]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                row["translation"] = ""
                row["summary"] = ""
                row["keywords"] = ""
                writer.writerow(row)

    # 2. Read all lines from output CSV
    with open(output_csv, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row")
        fieldnames = reader.fieldnames
        passages = list(reader)

    processor = HebrewTextProcessor(max_concurrent)
    semaphore = asyncio.Semaphore(max_concurrent)

    # 3. Process only rows where translation/summary/keywords are unfilled
    #    We'll do it in chunks so partial results are saved incrementally.
    idx = 0
    total = len(passages)

    while idx < total:
        chunk_tasks = []
        chunk_indices = []

        # Build a chunk up to max_concurrent lines that need processing
        while len(chunk_tasks) < max_concurrent and idx < total:
            row = passages[idx]
            if (
                (not row["translation"])
                or (not row["summary"])
                or (not row["keywords"])
            ):
                # This row is missing at least one field
                async def handle_row(r=row, i=idx):
                    async with semaphore:
                        updated = await processor.process_passage(r)
                        return (i, updated)

                chunk_tasks.append(handle_row())
                chunk_indices.append(idx)
            idx += 1

        # If no tasks found, we're done
        if not chunk_tasks:
            break

        # Gather results for this chunk
        results = await asyncio.gather(*chunk_tasks)

        # Update and write incremental results to CSV
        for row_index, updated_passage in results:
            passages[row_index] = updated_passage

        # Write the entire CSV to persist partial progress
        with open(output_csv, "w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(passages)

    print(f"Incremental enrichment complete. Results saved to {output_csv}")


if __name__ == "__main__":
    asyncio.run(main())
