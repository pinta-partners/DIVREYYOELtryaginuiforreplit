"""
Proprietary and Confidential
Copyright © 2024 Erick Aleman
Contact: Erick@EACognitive.com

This file contains proprietary implementation of:
1. Hebrew Analysis System
   - Context-aware translation mechanisms
   - Relevance mapping
   - Explanation generation
   - Cultural context preservation system

Protected under LICENSE.md
Version: 1.0

Any unauthorized copying, modification, distribution, or use of this file,
via any medium, is strictly prohibited.
"""

# step_4.py

import json
import os
import time
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
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
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger("step_4_script")

# Global lock for thread-safe file writing
lock = Lock()

# Processing settings
MAX_WORKERS = 10  # Process 10 passages concurrently


def get_latest_question_id() -> str:
    """Get the most recent question ID from the manifest file."""
    manifest_path = Path("data/manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError("Manifest file not found. Please run previous steps first.")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not manifest:
        raise ValueError("Manifest is empty. Please run previous steps first.")

    latest_entry = sorted(manifest, key=lambda x: x["timestamp"])[-1]
    response_path = Path(latest_entry["path"])
    return response_path.parent.parent.name


def get_completion(client: OpenAI, system_message: str, user_message: str) -> str:
    """Get completion from OpenAI API with error handling."""
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.15,
            max_tokens=1500,
            presence_penalty=0,
            frequency_penalty=0,
            stream=False
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise


def extract_relevant_sentences(client: OpenAI, passage: Dict, question: str) -> str:
    """Extract relevant sentences from passage."""
    try:
        system_message = """אתה מומחה בזיהוי טקסטים מדויקים מתוך כתבי הדברי יואל.
משימתך היא רק אחת: להעתיק באופן מדויק משפטים מהטקסט שעונים באופן ישיר על השאלה.

כללים:
- העתק אך ורק משפטים שלמים
- העתק מילה במילה בדיוק כפי שמופיע בטקסט
- אל תוסיף שום מילת הסבר או קישור
- אל תשנה את סדר המילים
- אל תשנה נקודות, פסיקים או רווחים
- אל תוסיף ניקוד
- אם אין משפטים שעונים ישירות, ציין "אין משפטים ישירים"

אזהרה: אל תוסיף שום דבר מעבר למשפטים עצמם."""

        user_message = f"""שאלה: {question}

טקסט לניתוח:
{passage.get('passage', '')}

העתק רק את המשפטים הרלוונטיים:"""

        return get_completion(client, system_message, user_message)

    except Exception as e:
        logger.error(f"[red]Error extracting sentences: {str(e)}[/red]")
        return f"Error: {str(e)}"


def generate_explanation(client: OpenAI, passage: Dict, relevant_sentences: str, question: str) -> str:
    """Generate explanation using passage and extracted sentences."""
    try:
        system_message = """הנך נדרש לבאר בלשון רבנית מסורתית כיצד דברי הדברי יואל עונים על השאלה שנשאלה.

יש להשיב בלשון רבנית בלבד (כמו בתשובות האחרונים), ולא בעברית מודרנית.

עליך לכתוב משפט אחד בלבד המתחיל ב"ביאור העניין הוא" המסביר כיצד הקטע והמשפטים המצוטטים עונים על השאלה.

דוגמא לסגנון הנדרש:
ביאור העניין הוא שבשעת התפילה, מחשבות זרות עלולות לחדור ללב המתפלל, וכך נראה כאילו פיו ולבו אינם שווים, אך באמת זוהי מלחמה רוחנית נגד כוחות המנסים לבלבל את כוונתו הטהורה.

הנחיות:
- הסבר כיצד הקטע עונה על השאלה
- השתמש בלשון רבנית מסורתית בלבד
- משפט אחד קצר ותמציתי
- התייחס למשפטים שצוטטו מהקטע
- פתח ב"ביאור העניין הוא\""""

        user_message = f"""שאלה: {question}

הטקסט המלא:
{passage.get('passage', '')}

המשפטים המצוטטים:
{relevant_sentences}

הסבר כיצד טקסט זה והמשפטים המצוטטים עונים על השאלה:"""

        return get_completion(client, system_message, user_message)

    except Exception as e:
        logger.error(f"[red]Error generating explanation: {str(e)}[/red]")
        return f"Error: {str(e)}"


def process_passage(
    client: OpenAI,
    passage: Dict,
    question: str,
    passage_index: int
) -> Dict:
    """Process a single passage with two API calls."""
    try:
        logger.info(f"[blue]Processing passage {passage_index + 1}[/blue]")

        # Build a 'source' string with torah_number / passage_number
        # to avoid KeyError on 'number'
        source = (
            f"Divrey Yoel, {passage.get('section','?')}, {passage.get('topic','?')} "
            f"(Torah #{passage.get('torah_number','?')}, Passage #{passage.get('passage_number','?')})"
        )

        # First API call - extract relevant sentences
        relevant_sentences = extract_relevant_sentences(client, passage, question)

        # Second API call - generate explanation
        explanation = generate_explanation(client, passage, relevant_sentences, question)

        return {
            "source": source,
            "relevant_sentences": relevant_sentences,
            "passage": passage.get('passage',''),
            "explanation": explanation
        }

    except Exception as e:
        logger.error(f"[red]Error processing passage {passage_index + 1}: {str(e)}[/red]")
        return {
            "source": f"Divrey Yoel, {passage.get('section','?')}, {passage.get('topic','?')} "
                      f"(Torah #{passage.get('torah_number','?')}, Passage #{passage.get('passage_number','?')})",
            "relevant_sentences": f"Error: {str(e)}",
            "passage": passage.get('passage',''),
            "explanation": f"Error: {str(e)}"
        }


def save_results(question: str, results: List[Dict], original_data: Dict) -> Path:
    """Save the passages with their analyses in descending score order."""
    try:
        # Instead of using (section, topic, number), we reference the final 'reference'
        # from step_3 or the newly built 'source' in step_4
        # so let's build a dictionary that maps reference -> average_score
        score_mapping = {
            p.get('reference', ''): float(p.get('average_score', 0))
            for p in original_data["selected_passages"]
        }

        # Sort the results in descending order of the mapped score
        # If we can't find a reference in score_mapping, default to 0
        sorted_results = sorted(
            results,
            key=lambda x: score_mapping.get(x["source"], 0),
            reverse=True
        )

        output_data = {
            "question": question,
            "analyzed_passages": sorted_results
        }

        step_4_folder = Path("data/answers") / get_latest_question_id() / "step_4"
        step_4_folder.mkdir(parents=True, exist_ok=True)

        output_file = step_4_folder / "passage_analysis.json"
        with lock:
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)

        logger.info(f"[green]Saved analysis to {output_file}[/green]")
        return output_file

    except Exception as e:
        logger.error(f"[red]Error saving results: {str(e)}[/red]")
        raise


def main(question_id=None):
    try:
        console.print(Panel.fit("[yellow]Step 4: Generating Passage Analysis[/yellow]"))

        # Load environment variables and initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("[red]OPENAI_API_KEY not found in environment variables[/red]")

        client = OpenAI(api_key=api_key)

        # Get latest question ID if not provided
        question_id = question_id or get_latest_question_id()
        logger.info(f"[cyan]Processing question ID: {question_id}[/cyan]")

        # Load results from step 3
        step_3_folder = Path("data/answers") / question_id / "step_3"
        final_selections_path = step_3_folder / "final_selections.json"
        if not final_selections_path.exists():
            raise FileNotFoundError(f"[red]Final selections not found at {final_selections_path}[/red]")

        with final_selections_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        question = data.get("question")
        if not question:
            raise ValueError("[red]Question not found in final selections[/red]")

        selected_passages = data.get("selected_passages", [])
        if not selected_passages:
            raise ValueError("[red]No selected passages found in final selections[/red]")

        # Process all passages
        all_results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Processing passages...", total=len(selected_passages))

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_passage = {
                    executor.submit(
                        process_passage, 
                        client, 
                        passage, 
                        question, 
                        idx
                    ): idx
                    for idx, passage in enumerate(selected_passages)
                }

                for future in as_completed(future_to_passage):
                    try:
                        result = future.result()
                        if result:
                            all_results.append(result)
                        progress.advance(task)
                    except Exception as e:
                        logger.error(f"[red]Error processing passage: {e}[/red]")

        if not all_results:
            raise ValueError("[red]No results were generated[/red]")

        # Save final results
        output_file = save_results(question, all_results, data)
        console.print(Panel.fit(f"[green]Successfully processed and saved analysis to {output_file}[/green]"))

    except Exception as e:
        logger.error(f"[red]An error occurred: {e}[/red]")
        console.print(
            Panel.fit(f"[red]Error: {e}[/red]", title="Error Details", border_style="red")
        )
        exit(1)


if __name__ == "__main__":
    main()
