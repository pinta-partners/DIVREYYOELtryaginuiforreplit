"""
Copyright © 2024 Erick Aleman
Contact: Erick@EACognitive.com

This file is part of the DMSRS implementation.
Contains utility functions and standard implementations.
See LICENSE.md for terms of use.
"""

# step_1.py

import json
import os
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import uuid

from litellm import completion as litellm_completion
# from openai import OpenAI
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.traceback import install

# Install rich traceback handler
install(show_locals=True)

# Initialize Rich console
console = Console()

# Configure logging with Rich handler
logging.basicConfig(level=logging.DEBUG,
                    format="%(message)s",
                    handlers=[RichHandler(rich_tracebacks=True, markup=True)])
logger = logging.getLogger("step_1_script")

# Global lock for thread-safe file writing
lock = Lock()

# Centralized storage folder for saving results
CENTRALIZED_FOLDER = Path("data/answers")
CENTRALIZED_FOLDER.mkdir(parents=True, exist_ok=True)

# Rate limit controls - adjusted for GPT-4o-mini
CHUNK_SIZE = 6800  # tokens per chunk
MAX_WORKERS = 35  # Number of workers for ThreadPoolExecutor


def log_json_path(json_path: Path):
    """Logs the path of the generated JSON file for later steps."""
    manifest_path = Path("data/manifest.json")
    manifest = []

    # Load existing manifest if it exists
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

    # Check for duplicate paths
    if str(json_path) not in [entry["path"] for entry in manifest]:
        manifest.append({
            "timestamp": datetime.now().isoformat(),
            "path": str(json_path)
        })

    # Save the updated manifest
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    logger.info(f"[green]Logged JSON path: {json_path}[/green]")


def chunk_text(text: str,
               max_tokens: int,
               overlap_tokens: int = 1300) -> List[str]:
    """Chunk the text into parts with overlapping content."""
    logger.info(
        "[cyan]Chunking text into manageable parts with overlap...[/cyan]")
    words = text.split()
    chunks = []
    tokens_per_word = 1.33  # Approximate token count per word
    max_words = int(max_tokens / tokens_per_word)
    overlap_words = int(overlap_tokens / tokens_per_word)

    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = words[start:end]

        # Add overlap to the next chunk if not at the end of the text
        if end < len(words):
            chunk += words[end - overlap_words:end]

        chunks.append(' '.join(chunk))
        start += max_words - overlap_words  # Move start to overlap

    logger.info(
        f"[green]Text split into {len(chunks)} chunks with overlap[/green]")
    return chunks


def process_single_chunk(chunk, question, chunk_index) -> Dict:
    """Process a single chunk and return the response along with the metadata."""
    logger.info(f"[blue]Processing chunk {chunk_index}[/blue]")

    try:
        # Define the messages (prompt)
        system_message = (
            "You are a highly knowledgeable scholar and expert in the teachings of the Divrey Yoel. "
            "Your task is to analyze the provided text to identify passages that best reflect the teachings or themes "
            "of the Divrey Yoel in relation to the given question.")
        user_message = f"""TASK: Identify a passage from the Divrey Yoel that provides meaningful insight into the following question:
Question: {question}
Text from the Divrey Yoel to analyze:
{chunk}
"""
        response_instructions = f"""RESPONSE INSTRUCTIONS:
- Provide ONLY the passage reference in this format: "Divrey Yoel, Parshas [Name], Torah #[X], Passage #[Y]".

- If no passage aligns with the question, respond with "No relevant match found".
"""

        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            },
            {
                "role": "user",
                "content": response_instructions
            },
        ]

        # AI model settings
        model_settings = {
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 100,
            "top_p": 1,
            "stream": False,
        }

        # Send the request to the OpenAI API
        completion = litellm_completion(
            # model="anthropic/claude-3-sonnet-20240229",
            model=model_settings["model"],
            messages=messages,
            temperature=model_settings["temperature"],
            max_tokens=model_settings["max_tokens"],
            top_p=model_settings["top_p"],
            stream=model_settings["stream"])
        # completion = client.chat.completions.create(
        #     model=model_settings["model"],
        #     messages=messages,
        #     temperature=model_settings["temperature"],
        #     max_tokens=model_settings["max_tokens"],
        #     top_p=model_settings["top_p"],
        #     stream=model_settings["stream"],
        # )

        assistant_reply = completion.choices[0].message.content.strip()

        if not assistant_reply:
            raise ValueError("Empty response content from API")

        logger.info(
            f"[green]Received response for chunk {chunk_index}[/green]")

        return {
            "response": assistant_reply,
        }

    except Exception as e:
        logger.error(f"[red]Error processing chunk {chunk_index}: {e}[/red]")
        return {"response": f"Error: {e}"}


def save_to_question_folder(question: str, raw_answers: List[Dict],
                            question_id: str):
    """Save the question and cleaned answers in a centralized folder."""
    cleaned_passages = []
    for answer in raw_answers:
        if "response" in answer and answer["response"]:
            cleaned_passages.extend(
                [line for line in answer["response"].splitlines() if line])

    # Create step_1 folder within the question's primary folder
    question_folder = CENTRALIZED_FOLDER / question_id / "step_1"
    question_folder.mkdir(parents=True, exist_ok=True)

    data = {
        "question_id": question_id,
        "question": question,
        "answer": {
            "relevant_passages": cleaned_passages
        },
        "timestamp": datetime.now().isoformat(),
    }

    output_file = question_folder / "response.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    logger.info(f"[green]Saved cleaned response to {output_file}[/green]")
    log_json_path(output_file)
    return question_folder


def main(question=None):
    try:
        # Load environment variables and initialize OpenAI client
        # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        console.print(
            Panel.fit(
                "[yellow]GPT-4o-mini Chat Completion Script - Step 1[/yellow]")
        )

        # Only prompt for question if not provided
        if question is None:
            question = console.input(
                "[bold cyan]Please enter your question: [/bold cyan]").strip()

        if not question:
            raise ValueError("No question provided")

        question_id = str(uuid.uuid4())

        input_path = Path("guider/processed_output.txt")
        if not input_path.exists():
            raise FileNotFoundError("Input file not found.")

        with input_path.open("r", encoding="utf-8") as file:
            user_input = file.read().strip()

        if not user_input:
            raise ValueError("Input file is empty.")

        chunks = chunk_text(user_input, CHUNK_SIZE)

        raw_answers = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_chunk = {
                executor.submit(process_single_chunk, chunk, question, i + 1):
                i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_chunk):
                try:
                    raw_answers.append(future.result())
                except Exception as e:
                    logger.error(f"[red]Error processing a chunk: {e}[/red]")

        save_to_question_folder(question, raw_answers, question_id)

    except Exception as e:
        logger.error(f"[red]An error occurred: {e}[/red]")
        console.print(
            Panel.fit(f"[red]Error: {e}[/red]",
                      title="Error Details",
                      border_style="red"))
        raise  # Re-raise the exception to be caught by the main script


if __name__ == "__main__":
    main()
