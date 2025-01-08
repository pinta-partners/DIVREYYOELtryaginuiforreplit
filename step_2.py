"""
Copyright © 2024 Erick Aleman
Contact: Erick@EACognitive.com

This file is part of the DMSRS implementation.
Contains utility functions and standard implementations.
See LICENSE.md for terms of use.
"""

# step_2.py

import json
import pandas as pd
from pathlib import Path
import re
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.traceback import install
import logging

# Install rich traceback handler
install(show_locals=True)

# Initialize Rich console
console = Console()

# Configure logging with Rich handler
logging.basicConfig(level=logging.DEBUG,
                    format="%(message)s",
                    handlers=[RichHandler(rich_tracebacks=True, markup=True)])
logger = logging.getLogger("step_2_script")


def get_latest_question_id():
    """Get the most recent question ID from the manifest file."""
    manifest_path = Path("data/manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Manifest file not found. Please run step 1 first.")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not manifest:
        raise ValueError("Manifest is empty. Please run step 1 first.")

    # Sort by timestamp and get the most recent entry
    latest_entry = sorted(manifest, key=lambda x: x["timestamp"])[-1]
    response_path = Path(latest_entry["path"])

    return response_path.parent.parent.name


def retry_search(section, topic, torah_number, passage_number,
                 csv_data):  # <-- CHANGED
    """Enhanced retry search with multiple fallback strategies."""
    try:
        # Strategy 1: Exact match but case-insensitive
        matched_row = csv_data[
            (csv_data["section"].str.strip().str.lower() == section.lower())
            & (csv_data["topic"].str.strip().str.lower() == topic.lower()) &
            (csv_data["torah_number"].astype(str) == torah_number)
            &  # <-- CHANGED
            (csv_data["passage_number"].astype(str) == passage_number
             )  # <-- CHANGED
        ]
        if not matched_row.empty:
            logger.info(
                f"[green]Strategy 1 (Exact match) successful for Section: '{section}', Topic: '{topic}', Torah #: '{torah_number}', Passage #: '{passage_number}'[/green]"
            )
            return matched_row.iloc[0]

        # Strategy 2: Try matching with comma-separated section/topic
        if "," in topic:
            try:
                section_part, topic_part = topic.split(",", 1)
                combined_section = f"{section}, {section_part}".strip()
                matched_row = csv_data[
                    (csv_data["section"].str.strip().str.lower() ==
                     combined_section.lower())
                    & (csv_data["topic"].str.strip().str.lower() ==
                       topic_part.strip().lower()) &
                    (csv_data["torah_number"].astype(str) == torah_number)
                    &  # <-- CHANGED
                    (csv_data["passage_number"].astype(str) == passage_number
                     )  # <-- CHANGED
                ]
                if not matched_row.empty:
                    logger.info(
                        f"[green]Strategy 2 (Comma-split) successful for Section: '{section}', Topic: '{topic}', Torah #: '{torah_number}', Passage #: '{passage_number}'[/green]"
                    )
                    return matched_row.iloc[0]
            except Exception as e:
                logger.warning(
                    f"[yellow]Strategy 2 failed with error: {e}[/yellow]")

        # Strategy 3: Try matching with section as topic and vice versa
        try:
            matched_row = csv_data[(
                (csv_data["section"].str.strip().str.lower() == topic.lower())
                &
                (csv_data["topic"].str.strip().str.lower() == section.lower())
                | (csv_data["section"].str.strip().str.lower().str.contains(
                    section.lower(), na=False)) & (csv_data["topic"].str.strip(
                    ).str.lower().str.contains(topic.lower(), na=False)))
                                   & (csv_data["torah_number"].astype(str) ==
                                      torah_number)  # <-- CHANGED
                                   & (csv_data["passage_number"].astype(str)
                                      == passage_number)  # <-- CHANGED
                                   ]
            if not matched_row.empty:
                logger.info(
                    f"[green]Strategy 3 (Cross-match) successful for Section: '{section}', Topic: '{topic}', Torah #: '{torah_number}', Passage #: '{passage_number}'[/green]"
                )
                return matched_row.iloc[0]
        except Exception as e:
            logger.warning(
                f"[yellow]Strategy 3 failed with error: {e}[/yellow]")

        # Strategy 4: Fuzzy matching with strict torah#/passage# validation
        try:
            # Extract key terms from section and topic
            search_terms = set(
                (section + " " + topic).lower().replace(",", " ").split())
            potential_matches = []

            for _, row in csv_data.iterrows():
                # Strict matching on torah_number and passage_number
                if str(row["torah_number"]) != torah_number or str(
                        row["passage_number"]) != passage_number:
                    continue

                csv_text = (str(row["section"]) + " " +
                            str(row["topic"])).lower()
                # Count how many search terms are found in the CSV text
                matches = sum(1 for term in search_terms if term in csv_text)
                matching_ratio = matches / len(
                    search_terms) if search_terms else 0

                # Only consider it a match if we have a high confidence (75% or more terms match)
                if matching_ratio >= 0.75:
                    potential_matches.append((matching_ratio, row))

            # Sort by matching ratio and get the best match
            if potential_matches:
                best_match = sorted(potential_matches,
                                    key=lambda x: x[0],
                                    reverse=True)[0]
                logger.info(
                    f"[green]Strategy 4 (Fuzzy) successful with {best_match[0]*100:.1f}% confidence for Section: '{section}', Topic: '{topic}', Torah #: '{torah_number}', Passage #: '{passage_number}'[/green]"
                )
                return best_match[1]

        except Exception as e:
            logger.warning(
                f"[yellow]Strategy 4 failed with error: {e}[/yellow]")

        logger.warning(
            f"[yellow]All retry strategies failed for Section: '{section}', Topic: '{topic}', Torah #: '{torah_number}', Passage #: '{passage_number}'.[/yellow]"
        )
        return None

    except Exception as e:
        logger.error(
            f"[red]Retry failed for Section: '{section}', Topic: '{topic}', Torah #: '{torah_number}', Passage #: '{passage_number}'. Error: {e}[/red]"
        )
        return None


def process_response_file_with_csv(input_json_path: Path, csv_file_path: Path,
                                   output_json_path: Path):
    """Process the response.json file, search the CSV file for each passage, retry on failure, and save the results."""
    try:
        # Load the JSON data
        logger.info(
            f"[cyan]Loading input JSON file from {input_json_path}...[/cyan]")
        with input_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate JSON structure
        if "answer" not in data or "relevant_passages" not in data["answer"]:
            raise ValueError(
                "Invalid JSON structure. 'answer.relevant_passages' not found."
            )

        passages = data["answer"]["relevant_passages"]
        if not passages:
            raise ValueError(
                "[ERROR] No relevant passages found in the JSON file.")

        # Load and preprocess the CSV data
        logger.info(f"[cyan]Loading CSV file from {csv_file_path}...[/cyan]")
        csv_data = pd.read_csv(csv_file_path)

        # Standardize column names for matching
        csv_data.rename(
            columns={
                "section":
                "section",  # <-- CHANGED (only if CSV has lowercase column headers)
                "topic": "topic",  # <-- CHANGED
                "torah #": "torah_number",  # <-- CHANGED
                "passage #": "passage_number",  # <-- CHANGED
                "hebrew_text": "passage",  # <-- CHANGED
                "translation": "english_translation",  # <-- CHANGED
            },
            inplace=True,
        )

        # Preprocess CSV data
        for col in ["section", "topic", "torah_number",
                    "passage_number"]:  # <-- CHANGED
            csv_data[col] = csv_data[col].astype(str).str.strip()

        # Ensure required columns exist
        required_columns = [
            "section",
            "topic",
            "torah_number",  # <-- CHANGED
            "passage_number",  # <-- CHANGED
            "passage",
            "english_translation",
        ]
        for col in required_columns:
            if col not in csv_data.columns:
                raise ValueError(
                    f"[ERROR] Missing required column '{col}' in the CSV file."
                )

        # Initialize lists to hold queried results and errors
        matched_passages = []
        errors = []

        # Regex pattern for the new text file format
        pattern = re.compile(
            r"^Divrey Yoel,\s*Parshas\s+(?P<topic>[^,]+),\s*Torah\s*#(?P<torah_number>\d+),\s*Passage\s*#(?P<passage_number>\d+)$"
        )  # <-- CHANGED

        for passage in passages:
            if not isinstance(passage, str):
                errors.append({
                    "error": "Unsupported passage structure",
                    "original": passage
                })
                continue

            if "No relevant match found." in passage:
                continue

            try:
                # Parse the passage using regex
                match = pattern.match(passage)
                if not match:
                    raise ValueError(
                        f"Passage does not match the expected format: '{passage}'"
                    )

                # Assign text-file pieces to variables
                section = "Torah"  # <-- CHANGED
                topic = match.group("topic").strip()  # <-- CHANGED
                torah_number = match.group(
                    "torah_number").strip()  # <-- CHANGED
                passage_number = match.group(
                    "passage_number").strip()  # <-- CHANGED

                logger.debug(
                    f"[blue]Parsed Section: '{section}', Topic: '{topic}', "
                    f"Torah #: '{torah_number}', Passage #: '{passage_number}'[/blue]"
                )

                # Search in the CSV
                matched_row = csv_data[(csv_data["section"].str.strip(
                ).str.lower() == section.lower()) & (
                    csv_data["topic"].str.strip().str.lower() == topic.lower())
                                       & (csv_data["torah_number"].astype(str)
                                          == torah_number) &  # <-- CHANGED
                                       (csv_data["passage_number"].astype(str)
                                        == passage_number)  # <-- CHANGED
                                       ]

                # Retry search if no match
                if matched_row.empty:
                    logger.warning(
                        f"[yellow]No match found for Section: '{section}', "
                        f"Topic: '{topic}', Torah #: '{torah_number}', Passage #: '{passage_number}'. Retrying...[/yellow]"
                    )
                    matched_row = retry_search(section, topic, torah_number,
                                               passage_number,
                                               csv_data)  # <-- CHANGED

                if matched_row is not None and not matched_row.empty:
                    # Convert row to dictionary for JSON serialization
                    matched_row = matched_row if isinstance(
                        matched_row, pd.Series) else matched_row.iloc[0]
                    matched_row = matched_row.astype(str).fillna("")
                    matched_passages.append({
                        "original":
                        passage,
                        "section":
                        matched_row.get("section", ""),
                        "topic":
                        matched_row.get("topic", ""),
                        "torah_number":
                        matched_row.get("torah_number", ""),  # <-- CHANGED
                        "passage_number":
                        matched_row.get("passage_number", ""),  # <-- CHANGED
                        "passage":
                        matched_row.get("passage", ""),
                        "english_translation":
                        matched_row.get("english_translation", ""),
                    })
                else:
                    errors.append({
                        "original":
                        passage,
                        "error":
                        (f"No match found for Section: '{section}', Topic: '{topic}', "
                         f"Torah #: '{torah_number}', Passage #: '{passage_number}'"
                         )
                    })

            except Exception as e:
                logger.error(
                    f"[red]Failed to process passage '{passage}'. Error: {e}[/red]"
                )
                errors.append({
                    "original": passage,
                    "error": f"Processing error: {e}"
                })

        # Extract metadata from input JSON
        question_id = data.get("question_id", "")
        question = data.get("question", "")
        timestamp = datetime.now().isoformat()

        # Compile final output with metadata and matched passages
        final_output = {
            "question_id": question_id,
            "question": question,
            "matched_passages": matched_passages,
            "errors": errors,
            "timestamp": timestamp,
        }

        # Save queried results
        with output_json_path.open("w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)

        logger.info(
            f"[green]Queried results saved to {output_json_path}[/green]")
        return str(output_json_path)

    except Exception as e:
        logger.error(f"[red]Failed to process file: {e}[/red]")
        raise


def main(question_id=None):
    try:
        console.print(
            Panel.fit(
                "[yellow]Step 2: Processing Latest Question Results[/yellow]"))

        # Get the latest question ID from the manifest
        question_id = get_latest_question_id()
        logger.info(
            f"[cyan]Processing latest question ID: {question_id}[/cyan]")

        # Define paths
        question_folder = Path("data/answers") / question_id
        step_1_folder = question_folder / "step_1"
        input_json_path = step_1_folder / "response.json"

        # Define the output folder for step_2 and ensure it exists
        step_2_folder = question_folder / "step_2"
        step_2_folder.mkdir(parents=True, exist_ok=True)
        output_json_path = step_2_folder / "queried_results.json"

        # Define the CSV file path
        csv_file_path = Path("data/dataset.csv")

        # Verify all required files exist
        if not step_1_folder.exists():
            raise FileNotFoundError(
                f"Step 1 folder not found at '{step_1_folder}'")
        if not input_json_path.exists():
            raise FileNotFoundError(
                f"Response.json not found at '{input_json_path}'")
        if not csv_file_path.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found at '{csv_file_path}'")

        # Process the response.json and save queried_results.json
        process_response_file_with_csv(input_json_path, csv_file_path,
                                       output_json_path)
        console.print(
            Panel.fit("[green]Processing completed successfully![/green]"))

    except Exception as e:
        logger.error(f"[red]An error occurred: {e}[/red]")
        console.print(
            Panel.fit(f"[red]Error: {e}[/red]",
                      title="Error Details",
                      border_style="red"))
        exit(1)


if __name__ == "__main__":
    main()
