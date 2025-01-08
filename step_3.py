# step_3.py
"""
Proprietary and Confidential
Copyright © 2024 Erick Aleman
Contact: Erick@EACognitive.com

This file is part of the DMSRS implementation.
Contains utility functions and standard implementations.
See LICENSE.md for terms of use.
"""

import json
import os
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import re

# Anthropic Messages API
from anthropic import Anthropic

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.traceback import install

# Install rich traceback handler
install(show_locals=True)

# Initialize Rich console
console = Console()

# Configure logging with Rich handler
logging.basicConfig(level=logging.DEBUG,
                    format="%(message)s",
                    handlers=[RichHandler(rich_tracebacks=True, markup=True)])
logger = logging.getLogger("step_3_script")

# Global lock for thread-safe file writing
lock = Lock()

# Rate limit controls
PASSAGES_PER_CALL = 1  # Number of passages to process per API call
MAX_WORKERS = 19  # Number of concurrent workers
TARGET_PASSAGES = 15  # Desired number of passages to narrow down to
MINIMUM_SCORE_THRESHOLD = 7.0  # Must meet/exceed 7.0 average score


def get_latest_question_id() -> str:
    """Get the most recent question ID from the manifest file."""
    manifest_path = Path("data/manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Manifest file not found. Please run step 1 first.")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not manifest:
        raise ValueError("Manifest is empty. Please run step 1 first.")

    latest_entry = sorted(manifest, key=lambda x: x["timestamp"])[-1]
    response_path = Path(latest_entry["path"])
    return response_path.parent.parent.name


def prepare_passage_batches(passages: List[Dict]) -> List[List[Dict]]:
    """Split passages into smaller batches for the LLM to handle."""
    logger.info("[cyan]Preparing passage batches...[/cyan]")
    batches = [
        passages[i:i + PASSAGES_PER_CALL]
        for i in range(0, len(passages), PASSAGES_PER_CALL)
    ]
    logger.info(
        f"[green]Created {len(batches)} batches from {len(passages)} passages[/green]"
    )
    return batches


def strip_code_fences(text: str) -> str:
    """Remove triple backtick fences, if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.startswith('json'):
            text = text[4:]
        if text.startswith('\n'):
            text = text[1:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def clean_hebrew_text(text: str) -> str:
    """Clean Hebrew text while preserving paragraph breaks and proper spacing."""
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        lines = paragraph.split('\n')
        joined = ' '.join(line.strip() for line in lines)
        joined = ' '.join(joined.split())
        if joined:
            cleaned_paragraphs.append(joined)
    return '\n\n'.join(cleaned_paragraphs).strip()


def process_single_batch(client: Anthropic, batch: List[Dict], question: str,
                         batch_index: int) -> Dict:
    """
    Send a single batch of passages to Claude (via the Messages API) for scoring.
    Returns {"batch_index": int, "response": [...]} or {"error": "..."}.
    """
    try:
        logger.info(f"[blue]Processing batch {batch_index + 1}[/blue]")

        # Process single passage (PASSAGES_PER_CALL=1)
        passage = batch[0] if batch else {}
        reference = passage.get("reference", "Unknown Reference")
        passage_text = passage.get("passage", "")

        system_prompt = (
            "You are a knowledgeable evaluator of Torah content, particularly familiar with the sefer Divrey yoel. "
            "Your task is to score the relevance of retrieved Divrei Torah passages against a user's query "
            "on a scale of 1-10.\n\n"
            "Guidelines:\n"
            "- 10: Perfect match, exactly what they're looking for\n"
            "- 7-9: Good match, addresses their main point\n"
            "- 4-6: Somewhat related\n"
            "- 1-3: Barely relevant\n\n"
            "Consider what the user intended to ask, and if it actually answers what they're asking.\n\n"
            "Please reason out loud in your answer.")

        # Send request using messages API with system as top-level parameter
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            system=system_prompt,
            messages=[{
                "role":
                "user",
                "content":
                (f"Given the question below and the passage, please think step by step in Hebrew or English, "
                 f"explaining how or why this passage addresses the user's question. "
                 f"At the very end, include a single line:\nFinal Score: X\n"
                 f"Where X is an integer from 1 to 10.\n\n"
                 f"Question: {question}\n\n"
                 f"Passage (Reference: {reference}):\n{passage_text}\n")
            }],
            max_tokens=1000,
            temperature=0.1)

        response_content = response.content[0].text.strip()
        if not response_content:
            raise ValueError("Empty completion response")

        logger.debug(
            f"RAW model response (batch {batch_index+1}):\n{response_content}")

        # Extract final score from the text
        match = re.search(r"[Ff]inal\s*[Ss]core:\s*(\d+)", response_content)
        if match:
            numeric_score = int(match.group(1))
            if not 1 <= numeric_score <= 10:
                raise ValueError(f"Invalid score value: {numeric_score}")
        else:
            numeric_score = 1  # fallback if not found
            logger.warning(
                f"[yellow]No score found in response for batch {batch_index + 1}, using fallback score of 1[/yellow]"
            )

        return {
            "batch_index":
            batch_index,
            "response": [{
                "reference": reference,
                "score": numeric_score,
                "raw_text": response_content
            }]
        }

    except Exception as e:
        logger.error(
            f"[red]Error processing batch {batch_index + 1}: {e}[/red]")
        return {"batch_index": batch_index, "error": str(e)}


def save_final_results(question_id: str, question: str,
                       selected_passages: List[Dict],
                       all_responses: List[Dict]) -> Path:
    """
    Save the final selected passages and all batch responses to JSON.
    """
    try:
        step_3_folder = Path("data/answers") / question_id / "step_3"
        step_3_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Number of selected passages: {len(selected_passages)}")

        # Clean up the selected passages
        cleaned_passages = []
        for psg in selected_passages:
            # Must have these 6 fields or skip
            if not all(k in psg for k in [
                    "section", "topic", "torah_number", "passage_number",
                    "passage", "english_translation"
            ]):
                logger.warning(f"Missing required fields in passage: {psg}")
                continue

            # Tidy up text
            cleaned_passage = {
                "section": psg["section"].strip(),
                "topic": psg["topic"].strip(),
                "torah_number": psg["torah_number"].strip(),
                "passage_number": psg["passage_number"].strip(),
                "passage": clean_hebrew_text(psg["passage"]),
                "english_translation": psg["english_translation"].strip(),
                "average_score": psg.get("average_score", 0),
                "reference": psg.get("reference", "")
            }
            cleaned_passages.append(cleaned_passage)
        average_score = sum(
            psg["average_score"]
            for psg in cleaned_passages) / len(cleaned_passages)

        final_output = {
            "question_id": question_id,
            "question": question,
            "selected_passages": cleaned_passages,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "total_passages_processed": len(all_responses),
                "average_score": average_score,
                "selection_criteria":
                f"Average score >= {MINIMUM_SCORE_THRESHOLD}",
                "scoring_version": "3.0 (Anthropic Messages API)"
            }
        }

        selections_file = step_3_folder / "final_selections.json"
        with lock:
            with selections_file.open("w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=4)

        # Also save all_responses for debugging
        debug_output = {
            "question_id": question_id,
            "question": question,
            "all_responses": all_responses,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "total_batches":
                len(all_responses),
                "successful_batches":
                sum(1 for r in all_responses if "error" not in r)
            }
        }
        debug_file = step_3_folder / "all_responses.json"
        with lock:
            with debug_file.open("w", encoding="utf-8") as f:
                json.dump(debug_output, f, ensure_ascii=False, indent=4)

        logger.info(f"[green]Saved final results to {selections_file}[/green]")
        return selections_file

    except Exception as e:
        logger.error(f"[red]Error saving final results: {e}[/red]")
        raise


def main(question_id=None):
    try:
        console.print(
            Panel.fit("[yellow]Step 3: Final Passage Selection[/yellow]"))

        # Load environment / Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "[red]ANTHROPIC_API_KEY not found in environment variables[/red]"
            )
        client = Anthropic(api_key=api_key)

        # If question_id not given, get from manifest
        question_id = question_id or get_latest_question_id()
        logger.info(f"[cyan]Processing question ID: {question_id}[/cyan]")

        # Load step_2 results
        step_2_folder = Path("data/answers") / question_id / "step_2"
        queried_results_path = step_2_folder / "queried_results.json"
        if not queried_results_path.exists():
            raise FileNotFoundError(
                f"[red]No queried_results.json found at {queried_results_path}[/red]"
            )

        with queried_results_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        question = data.get("question")
        if not question:
            raise ValueError(
                "[red]No question found in queried_results.json[/red]")

        matched_passages = data.get("matched_passages", [])
        if not matched_passages:
            raise ValueError("[red]No matched passages found[/red]")

        # Build references from torah_number & passage_number
        original_passages = {}
        for psg in matched_passages:
            section = psg.get("section", "Unknown").strip()
            topic = psg.get("topic", "Unknown").strip()
            torah_num = psg.get("torah_number", "0").strip()
            pass_num = psg.get("passage_number", "0").strip()

            # Create a reference for Claude
            reference = f"Divrey Yoel, Parshas {topic}, Torah #{torah_num}, Passage #{pass_num}"
            psg["reference"] = reference

            # Ensure these are in the final dictionary
            psg["torah_number"] = torah_num
            psg["passage_number"] = pass_num

            # Store for later retrieval
            original_passages[reference] = psg

        # Batch and process concurrency
        batches = prepare_passage_batches(matched_passages)
        logger.info(f"[cyan]Processing {len(batches)} batches...[/cyan]")

        all_responses = []
        with Progress(SpinnerColumn(),
                      TextColumn("[progress.description]{task.description}"),
                      console=console) as progress:
            task_id = progress.add_task("[cyan]Processing batches...",
                                        total=len(batches))

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_map = {
                    executor.submit(process_single_batch, client, batch, question, i):
                    i
                    for i, batch in enumerate(batches)
                }
                for future in as_completed(future_map):
                    try:
                        result = future.result()
                        all_responses.append(result)
                        progress.advance(task_id)
                    except Exception as e:
                        logger.error(f"[red]Error processing batch: {e}[/red]")

        # Collect and average scores
        passage_scores = {}
        for response in all_responses:
            if "response" in response and "error" not in response:
                for item in response["response"]:
                    ref = item.get("reference")
                    sc = item.get("score")
                    if ref and sc is not None:
                        try:
                            numeric_score = float(sc)
                            passage_scores.setdefault(ref,
                                                      []).append(numeric_score)
                        except ValueError:
                            logger.error(
                                f"[red]Invalid score for {ref}: {sc}[/red]")

        averaged_scores = []
        for ref, scores in passage_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                averaged_scores.append({
                    "reference": ref,
                    "average_score": avg_score
                })

        # Filter by threshold
        filtered = [
            x for x in averaged_scores
            if x["average_score"] >= MINIMUM_SCORE_THRESHOLD
        ]
        filtered.sort(key=lambda x: x["average_score"], reverse=True)

        top_passages = filtered[:TARGET_PASSAGES]
        logger.info(
            f"[green]Selected {len(top_passages)} passages >= {MINIMUM_SCORE_THRESHOLD}[/green]"
        )

        # Rebuild final data
        selected_passage_data = []
        for scored in top_passages:
            ref = scored["reference"]
            if ref in original_passages:
                passage_data = original_passages[ref].copy()
                passage_data["average_score"] = scored["average_score"]
                selected_passage_data.append(passage_data)
                logger.info(
                    f"[cyan]Added passage {ref} => score {scored['average_score']}[/cyan]"
                )
            else:
                logger.warning(
                    f"[yellow]No original data found for {ref}[/yellow]")

        # Save or error
        if selected_passage_data:
            output_file = save_final_results(question_id, question,
                                             selected_passage_data,
                                             all_responses)
            console.print(
                Panel.fit(
                    f"[green]Done! Saved results to {output_file}[/green]"))
        else:
            raise ValueError(
                f"[red]No passages were selected with average score >= {MINIMUM_SCORE_THRESHOLD}[/red]"
            )

    except Exception as e:
        logger.error(f"[red]An error occurred: {e}[/red]")
        console.print(
            Panel.fit(f"[red]Error: {e}[/red]",
                      title="Error Details",
                      border_style="red"))
        exit(1)


if __name__ == "__main__":
    main()
