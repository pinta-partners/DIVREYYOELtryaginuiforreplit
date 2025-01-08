import asyncio
import csv
import sys
from typing import List, Dict
from litellm import acompletion

# Example how to run:
# python data/preprocessing/enrich_csv_with_translation_and_keywords.py data/divrey_yoel_vayechi.csv data/divrey_yoel_vayechi_enriched.csv 15


class HebrewTextProcessor:

    def __init__(self, max_concurrent: int):
        self.max_concurrent = max_concurrent  # Bounded parallelism

    async def call_litellm(self, prompt: str) -> str:
        """Make an asynchronous call to LiteLLM with a prompt."""
        try:
            messages = [{"role": 'user', "content": prompt}]
            response = await acompletion(model="claude-3-5-sonnet-20241022",
                                         messages=messages,
                                         max_tokens=1500,
                                         temperature=0,
                                         num_retries=3)
            assistant_reply = response.choices[0].message.content.strip()
            return assistant_reply
        except Exception as e:
            print(f"API error: {e}")
            return "API call failed"

    async def translate_text(self, hebrew_text: str) -> str:
        """Translate Hebrew text to English using the exact prompt."""
        prompt = f"""Please translate the following Hebrew text into English. This is from Sefer Divrey Yoel:

        Hebrew text: {hebrew_text}

        Provide the translation, preserving Hasidic concepts and terminology."""
        return await self.call_litellm(prompt)

    async def generate_summary(self, hebrew_text: str) -> str:
        """Generate a summary of the Hebrew text using the exact prompt."""
        prompt = f"""Please read and summarize the following Hebrew passage from Sefer Divrey Yoel.
        Focus on the key Hasidic concepts and theological insights:

        Hebrew text: {hebrew_text}

        Provide a clear 3-4 sentence summary that captures the theological depth."""
        return await self.call_litellm(prompt)

    async def generate_keywords(self, hebrew_text: str) -> str:
        """Extract key Hebrew/Jewish theological terms using the exact prompt."""
        prompt = f"""Please extract 10 key Hebrew/Jewish theological terms from this Hebrew passage:

        Hebrew text: {hebrew_text}

        List exactly 10 terms, one per line, focusing on Hasidic and Kabbalistic concepts."""
        return await self.call_litellm(prompt)

    async def process_passage(self, passage: Dict[str, str]) -> Dict[str, str]:
        """Enrich a single passage."""
        hebrew_text = passage['passage_content']
        print(
            f"Processing: {passage['book_name']} - {passage['parsha_name']} - Torah #{passage['dvar_torah_id']} - Passage #{passage['passage_id']}"
        )

        # Process asynchronously
        translation_task = asyncio.create_task(
            self.translate_text(hebrew_text))
        summary_task = asyncio.create_task(self.generate_summary(hebrew_text))
        keywords_task = asyncio.create_task(
            self.generate_keywords(hebrew_text))

        translation, summary, keywords = await asyncio.gather(
            translation_task, summary_task, keywords_task)

        # Enrich the passage
        passage['translation'] = translation
        passage['summary'] = summary
        passage['keywords'] = keywords
        return passage

    async def process_all_passages(
            self, passages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Enrich all passages."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_with_semaphore(passage):
            async with semaphore:
                return await self.process_passage(passage)

        tasks = [process_with_semaphore(passage) for passage in passages]
        return await asyncio.gather(*tasks)


async def main():

    # Check command-line arguments for input and output file paths
    if len(sys.argv) != 4:
        print(
            "Usage: python enrich_csv.py <input_csv> <output_csv> <max_concurrent>"
        )
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    max_concurrent = int(sys.argv[3])

    # Load passages from CSV
    passages = []
    with open(input_csv, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            # Initialize missing fields
            row['translation'] = ""
            row['summary'] = ""
            row['keywords'] = ""
            passages.append(row)

    # Process all passages
    processor = HebrewTextProcessor(max_concurrent)
    enriched_passages = await processor.process_all_passages(passages)

    # Write enriched passages to CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        fieldnames = list(enriched_passages[0].keys())
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_passages)

    print(f"Enriched output saved to {output_csv}")


if __name__ == "__main__":
    asyncio.run(main())
