"""
Proprietary and Confidential
Copyright © 2024 Erick Aleman
Contact: Erick@EACognitive.com

This file contains proprietary implementation of:
1. Dynamic Multi-Stage Retrieval System (DMSRS)
   - Progressive refinement methodology
   - Multi-level filtering system
   - Dynamic passage evaluation algorithms
2. Advanced Scoring Algorithm
   - Contextual relevance scoring system
   - Multi-factor evaluation metrics
   - Dynamic weighting mechanisms

Protected under LICENSE.md
Version: 1.0

Any unauthorized copying, modification, distribution, or use of this file,
via any medium, is strictly prohibited.
"""

#step_3.py
import json
import os
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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
MINIMUM_SCORE_THRESHOLD = 7.0  # Minimum average score to include a passage


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
    """Prepare batches of passages for processing."""
    logger.info("[cyan]Preparing passage batches...[/cyan]")

    # Split passages into batches
    batches = [
        passages[i:i + PASSAGES_PER_CALL]
        for i in range(0, len(passages), PASSAGES_PER_CALL)
    ]

    logger.info(
        f"[green]Created {len(batches)} batches from {len(passages)} passages[/green]"
    )
    return batches


def strip_code_fences(text: str) -> str:
    """Remove code fences from the text."""
    text = text.strip()
    if text.startswith("```"):
        # Remove ``` and optional language identifier
        text = text[3:]
        if text.startswith('json'):
            text = text[4:]
        # Remove leading newline if present
        if text.startswith('\n'):
            text = text[1:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def clean_hebrew_text(text: str) -> str:
    """Clean Hebrew text while preserving paragraph breaks and proper spacing.

    Args:
        text (str): Raw Hebrew text input

    Returns:
        str: Cleaned text with preserved paragraph structure
    """

    # Split into paragraphs (preserve empty lines as paragraph breaks)
    paragraphs = text.split('\n\n')

    # Clean each paragraph individually
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        # Clean up whitespace within each paragraph
        # Replace multiple spaces/newlines with a single space
        cleaned_paragraph = ' '.join(line.strip()
                                     for line in paragraph.split('\n'))
        cleaned_paragraph = ' '.join(cleaned_paragraph.split())
        if cleaned_paragraph:  # Only add non-empty paragraphs
            cleaned_paragraphs.append(cleaned_paragraph)

    # Join paragraphs with double newlines
    return '\n\n'.join(cleaned_paragraphs).strip()


def process_single_batch(client: OpenAI, batch: List[Dict], question: str,
                         batch_index: int) -> Dict:
    """Process a single batch of passages and return the response."""
    try:
        logger.info(f"[blue]Processing batch {batch_index + 1}[/blue]")

        system_message = (
            # "You are a scholar and expert in the teachings of Divrey Yoel,"
            # "with deep understanding of rabbinic Hebrew and Hasidic texts."
            # "Your task is to analyze the Hebrew articles and rate each on a scale of 1-10"
            # "according to the following criteria:"
            "אתה מלומד בקי ומומחה בתורת הדברי יואל, "
            "עם הבנה עמוקה בעברית רבנית וטקסטים חסידיים. "
            "משימתך היא לנתח את המאמרים בעברית ולדרג כל אחד בסולם מ-1 עד 10 לפי הקריטריונים הבאים:"
            "\n\n"
            # "Core Relevance Scoring:"
            "ניקוד רלוונטיות מרכזית:\n"
            # "1. Fundamentals of Livelihood and Protection (0-5 points):"
            "1. יסודות פרנסה ושמירה (0-5 נקודות):\n"
            # "- Does the article explain fundamentals of livelihood and divine providence? (0-2)"
            "- האם המאמר מסביר יסודות בענייני פרנסה והשגחה? (0-2)\n"
            # "   * 2 points only for articles explaining essential fundamentals like the Manna, Joseph, or sources of abundance"
            # "   * 1 point for general explanations"
            # "   * 0 points for superficial references"
            "   * 2 נקודות רק למאמרים המבארים יסודות מהותיים כמו המן, יוסף, או מקורות השפע\n"
            "   * 1 נקודה להסברים כלליים\n"
            "   * 0 נקודות להתייחסויות שטחיות\n"
            # "- Does it explain the essence of protection in livelihood? (0-2)"
            "- האם הוא מסביר את מהות השמירה בפרנסה? (0-2)\n"
            # "   * 2 points only for deep and detailed explanation of protection methods"
            # "   * 1 point for basic coverage"
            # "   * 0 points for superficial coverage"
            "   * 2 נקודות רק להסבר מעמיק ומפורט על דרכי השמירה\n"
            "   * 1 נקודה להתייחסות בסיסית\n"
            "   * 0 נקודות להתייחסות שטחית\n"
            # "- Does it detail the spiritual dangers in matters of livelihood? (0-1)"
            "- האם הוא מפרט את הסכנות הרוחניות בענייני פרנסה? (0-1)\n"
            # "   * 1 point only for detailed explanation of dangers"
            # "   * 0 points for general references"
            "   * 1 נקודה רק להסבר מפורט של הסכנות\n"
            "   * 0 נקודות להתייחסות כללית\n"
            "\n"
            # "2. Practical Application (0-3 points):"
            "2. יישום מעשי (0-3 נקודות):\n"
            # "- Does the article provide practical examples? (0-2)"
            "- האם המאמר מביא דוגמאות מעשיות? (0-2)\n"
            # "   * 2 points only for detailed practical examples from Torah or Chazal"
            # "   * 1 point for general examples"
            # "   * 0 points for mere mentions"
            "   * 2 נקודות רק לדוגמאות מפורטות ומעשיות מהתורה או מחז\"ל\n"
            "   * 1 נקודה לדוגמאות כלליות\n"
            "   * 0 נקודות לאזכורים בלבד\n"
            # "- Does it show how to implement the principles? (0-1)"
            "- האם הוא מראה איך ליישם את העקרונות? (0-1)\n"
            # "   * 1 point only for clear practical instructions"
            # "   * 0 points for general advice"
            "   * 1 נקודה רק להוראות מעשיות ברורות\n"
            "   * 0 נקודות לעצות כלליות\n"
            "\n"
            # "3. Spiritual Depth (0-2 points):"
            "3. עומק רוחני (0-2 נקודות):\n"
            # "- Does the article connect to deep spiritual ideas? (0-1)"
            "- האם המאמר מקשר לרעיונות רוחניים עמוקים? (0-1)\n"
            # "   * 1 point only for deep and detailed connections"
            "   * 1 נקודה רק לקישורים מעמיקים ומפורטים\n"
            # "- Does it show the connection between livelihood and divine service? (0-1)"
            "- האם הוא מראה את הקשר בין פרנסה לעבודת ה'? (0-1)\n"
            # "   * 1 point only for detailed and deep explanations"
            "   * 1 נקודה רק להסברים מפורטים ועמוקים\n"
            "\n"
            # "Bonus (maximum +1 point):"
            "בונוס (מקסימום +1 נקודה):\n"
            # "- +1 point only for foundational articles that form an essential basis for understanding the topic"
            # "or articles that connect the topic to fundamental ideas in Tiferet Shlomo's teachings"
            "- +1 נקודה רק למאמרים יסודיים המהווים בסיס מהותי להבנת הנושא\n"
            "או למאמרים המקשרים את הנושא לרעיונות יסוד בתורת הדברי יואל\n"
            "\n"
            # "Final Scoring Guide - Be Very Strict:"
            "מדריך ניקוד סופי - יש להקפיד מאוד:\n"
            # "10: Extremely rare - only for foundational article perfectly explaining the topic's roots"
            "10: נדיר ביותר - רק למאמר יסודי ומקיף המסביר את שורשי הנושא באופן מושלם\n"
            # "9: Very rare - only for especially deep article with detailed practical application"
            "9: נדיר מאוד - רק למאמר מעמיק במיוחד עם יישום מעשי מפורט\n"
            # "7-8: For good article covering important aspects deeply"
            "7-8: למאמר טוב המכסה היבטים חשובים בצורה מעמיקה\n"
            # "5-6: For article with good insights but not comprehensive"
            "5-6: למאמר עם תובנות טובות אך לא מקיף\n"
            # "3-4: For article with partial relevance to topic"
            "3-4: למאמר עם קשר חלקי לנושא\n"
            # "1-2: For article with minimal relevance"
            "1-2: למאמר עם קשר מינימלי\n"
            "\n"
            # "Important Notes:"
            "הערות חשובות:\n"
            # "- Be very strict in giving high scores"
            "- יש להיות קפדן מאוד במתן ציונים גבוהים\n"
            # "- Scores of 9-10 should be extremely rare"
            "- ציונים של 9-10 צריכים להיות נדירים ביותר\n"
            # "- Verify each point is given only when article truly meets exact criteria"
            "- יש לוודא שכל נקודה ניתנת רק כשהמאמר באמת עומד בקריטריונים המדויקים\n"
            # "- Don't give high scores just for mentioning important concepts - detailed explanation required"
            "- אין לתת ניקוד גבוה רק בגלל אזכור של מושגים חשובים - נדרשת הסברה מעמיקה"
        )

        # Initialize passages_text with source and passage
        passages_text = ""
        for passage in batch:
            reference = passage.get('reference', 'Unknown Reference')
            passage_text = passage.get('passage', '')
            passages_text += f"מקור: {reference}\nמאמר: {passage_text}\n\n"

        user_message = f"""משימה: נתח את המאמר מהדברי יואל המובא להלן, ותן לו ציון בין 1 ל-10 על פי מידת הרלוונטיות והתובנות שהוא מספק לשאלה הנתונה. יש להיות קפדן מאוד במתן ציונים גבוהים.

שאלה: {question}

המאמר לניתוח:
{passages_text}

הוראות:
- יש לציין את המקור בתור 'reference' ואת הציון בתור 'score'
- התוצאה צריכה להיות בפורמט JSON כך:
  [{{"reference": "מקור המאמר", "score": הציון}}]
- אין לכלול הסברים או טקסט נוסף"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_message
                },
            ]

            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=1000,
                presence_penalty=-0.1,  # Discourage repetitive high scores
                frequency_penalty=0.2,  # Encourage varied scoring
                top_p=0.8,  # More focused responses
                stream=False)

            if not completion or not completion.choices:
                raise ValueError("Empty completion response")

            response_content = completion.choices[0].message.content.strip()

            print(f"RAW model response:\n{response_content}")

            
            if not response_content:
                raise ValueError("Empty response content")

            # Remove code fences if present
            response_content = strip_code_fences(response_content)

            # Parse the response as JSON
            try:
                response_data = json.loads(response_content)
                if not isinstance(response_data, list):
                    response_data = [response_data]

                # Validate response data
                for item in response_data:
                    if 'reference' not in item or 'score' not in item:
                        raise ValueError(f"Invalid response format: {item}")

                logger.info(
                    f"[green]Successfully processed batch {batch_index + 1}[/green] {passage_text}"
                )
                return {"batch_index": batch_index, "response": response_data}

            except json.JSONDecodeError as e:
                logger.error(
                    f"[red]Error parsing JSON response: {e}[/red]\nResponse Content:\n{response_content}"
                )
                return {
                    "batch_index": batch_index,
                    "error": f"JSON Parsing Error: {e}"
                }

        except Exception as api_error:
            logger.error(
                f"[red]API error processing batch {batch_index + 1}: {str(api_error)}[/red]"
            )
            return {
                "batch_index": batch_index,
                "error": f"API Error: {str(api_error)}"
            }

    except Exception as e:
        logger.error(
            f"[red]Error processing batch {batch_index + 1}: {str(e)}[/red]")
        return {"batch_index": batch_index, "error": str(e)}


def process_single_batch(client: OpenAI, batch: List[Dict], question: str,
                         batch_index: int) -> Dict:
    """Process a single batch of passages and return the response.

    Args:
        client (OpenAI): The OpenAI client instance
        batch (List[Dict]): List of passages to process
        question (str): The specific question being analyzed
        batch_index (int): Index of current batch

    Returns:
        Dict: Processing results with batch_index and either response data or error
    """
    try:
        logger.info(f"[blue]Processing batch {batch_index + 1}[/blue]")

        # Ensure question is properly formatted
        formatted_question = question.strip()
        if not formatted_question:
            raise ValueError("Question cannot be empty")

        system_message = (
            "אתה מלומד בקי ומומחה בתורת הדברי יואל, "
            "עם הבנה עמוקה בעברית רבנית וטקסטים חסידיים. "
            "\n\n"
            f"השאלה שעליה אנו מחפשים תשובה היא:\n{formatted_question}\n\n"  # Question prominently displayed
            "משימתך היא לנתח את המאמרים בעברית ולדרג אותם לפי מידת הרלוונטיות שלהם לשאלה זו. "
            "עליך להיות קפדן מאוד ולתת ציון גבוה רק למאמרים שעונים על השאלה באופן מהותי ומעמיק. "
            "דרג כל מאמר בסולם מ-1 עד 10 לפי הקריטריונים הבאים:"
            "\n\n"
            "ניקוד רלוונטיות מרכזית:\n"
            "1. יסודות פרנסה ושמירה (0-5 נקודות):\n"
            "- האם המאמר מסביר יסודות הקשורים ישירות לשאלה? (0-2)\n"
            "   * 2 נקודות רק למאמרים המבארים יסודות מהותיים הקשורים לשאלה\n"
            "   * 1 נקודה להסברים כלליים\n"
            "   * 0 נקודות להתייחסויות שטחיות\n"
            "- האם הוא מסביר את העניין בצורה מעמיקה? (0-2)\n"
            "   * 2 נקודות רק להסבר מעמיק ומפורט\n"
            "   * 1 נקודה להתייחסות בסיסית\n"
            "   * 0 נקודות להתייחסות שטחית\n"
            "- האם הוא מפרט את ההיבטים החשובים? (0-1)\n"
            "   * 1 נקודה רק להסבר מפורט\n"
            "   * 0 נקודות להתייחסות כללית\n"
            "\n"
            "2. יישום מעשי (0-3 נקודות):\n"
            "- האם המאמר מביא דוגמאות מעשיות הקשורות לשאלה? (0-2)\n"
            "   * 2 נקודות רק לדוגמאות מפורטות ומעשיות מהתורה או מחז\"ל\n"
            "   * 1 נקודה לדוגמאות כלליות\n"
            "   * 0 נקודות לאזכורים בלבד\n"
            "- האם הוא מראה איך ליישם את העקרונות? (0-1)\n"
            "   * 1 נקודה רק להוראות מעשיות ברורות\n"
            "   * 0 נקודות לעצות כלליות\n"
            "\n"
            "3. עומק רוחני (0-2 נקודות):\n"
            "- האם המאמר מקשר לרעיונות רוחניים עמוקים הקשורים לשאלה? (0-1)\n"
            "   * 1 נקודה רק לקישורים מעמיקים ומפורטים\n"
            "- האם הוא מראה את הקשר לעבודת ה'? (0-1)\n"
            "   * 1 נקודה רק להסברים מפורטים ועמוקים\n"
            "\n"
            "בונוס (מקסימום +1 נקודה):\n"
            "- +1 נקודה רק למאמרים יסודיים המהווים בסיס מהותי להבנת השאלה\n"
            "או למאמרים המקשרים את השאלה לרעיונות יסוד בתורת הדברי יואל\n"
            "\n"
            "מדריך ניקוד סופי - יש להקפיד מאוד:\n"
            "10: נדיר ביותר - רק למאמר שעונה על השאלה באופן מושלם ומקיף\n"
            "9: נדיר מאוד - רק למאמר שעונה על השאלה בצורה מעמיקה במיוחד\n"
            "7-8: למאמר שעונה על השאלה בצורה טובה ומעמיקה\n"
            "5-6: למאמר שעונה על השאלה חלקית\n"
            "3-4: למאמר שקשור לשאלה באופן עקיף\n"
            "1-2: למאמר שקשור לשאלה באופן מינימלי\n"
            "\n"
            "הערות חשובות:\n"
            "- יש להתמקד בקשר הישיר לשאלה הנתונה\n"
            "- יש להיות קפדן מאוד במתן ציונים גבוהים\n"
            "- ציונים של 9-10 צריכים להיות נדירים ביותר\n"
            "- יש לוודא שכל נקודה ניתנת רק כשהמאמר באמת עונה על השאלה\n"
            "- אין לתת ניקוד גבוה רק בגלל אזכור של מושגים - נדרשת הסברה מעמיקה"
        )

        # Initialize passages_text with source and passage
        passages_text = ""
        for passage in batch:
            reference = passage.get('reference', 'Unknown Reference')
            passage_text = passage.get('passage', '')
            passages_text += f"מקור: {reference}\nמאמר: {passage_text}\n\n"

        # User message with prominent question placement
        user_message = f"""משימה: נתח את המאמר מהדברי יואל המובא להלן בהקשר של השאלה הבאה:

שאלה: {formatted_question}

עליך לתת ציון בין 1 ל-10 המשקף עד כמה המאמר עונה על השאלה הנ"ל.
יש להיות קפדן מאוד במתן ציונים ולתת ציון גבוה רק למאמרים שעונים על השאלה באופן מהותי.

המאמר לניתוח:
{passages_text}

הוראות:
- יש לציין את המקור בתור 'reference' ואת הציון בתור 'score'
- התוצאה צריכה להיות בפורמט JSON כך:
  [{{"reference": "מקור המאמר", "score": הציון}}]
- אין לכלול הסברים או טקסט נוסף"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_message
                },
            ]

            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=1000,
                presence_penalty=-0.1,  # Discourage repetitive high scores
                frequency_penalty=0.2,  # Encourage varied scoring
                top_p=0.8,  # More focused responses
                stream=False)

            if not completion or not completion.choices:
                raise ValueError("Empty completion response")

            response_content = completion.choices[0].message.content.strip()

            if not response_content:
                raise ValueError("Empty response content")

            # Remove code fences if present
            response_content = strip_code_fences(response_content)

            # Parse the response as JSON
            try:
                response_data = json.loads(response_content)
                if not isinstance(response_data, list):
                    response_data = [response_data]

                # Validate response data
                for item in response_data:
                    if 'reference' not in item or 'score' not in item:
                        raise ValueError(f"Invalid response format: {item}")

                logger.info(
                    f"[green]Successfully processed batch {batch_index + 1}[/green]"
                )
                return {"batch_index": batch_index, "response": response_data}

            except json.JSONDecodeError as e:
                logger.error(
                    f"[red]Error parsing JSON response: {e}[/red]\nResponse Content:\n{response_content}"
                )
                return {
                    "batch_index": batch_index,
                    "error": f"JSON Parsing Error: {e}"
                }

        except Exception as api_error:
            logger.error(
                f"[red]API error processing batch {batch_index + 1}: {str(api_error)}[/red]"
            )
            return {
                "batch_index": batch_index,
                "error": f"API Error: {str(api_error)}"
            }

    except Exception as e:
        logger.error(
            f"[red]Error processing batch {batch_index + 1}: {str(e)}[/red]")
        return {"batch_index": batch_index, "error": str(e)}


def save_final_results(question_id: str, question: str,
                       selected_passages: List[Dict],
                       all_responses: List[Dict]) -> Path:
    """Save the final selected passages and all responses to JSON files.

    Args:
        question_id (str): Identifier for the question
        question (str): The actual question text
        selected_passages (List[Dict]): List of selected passages with their metadata
        all_responses (List[Dict]): List of all batch processing responses

    Returns:
        Path: Path to the saved selections file
    """
    try:
        step_3_folder = Path("data/answers") / question_id / "step_3"
        step_3_folder.mkdir(parents=True, exist_ok=True)

        # Debug logging to see what we're getting
        logger.info(f"Number of selected passages: {len(selected_passages)}")

        # Process selected_passages to include only specified fields and clean the Hebrew text
        cleaned_selected_passages = []
        for passage in selected_passages:
            # Ensure we have all required fields
            if not all(key in passage for key in [
                    'section', 'topic', 'number', 'passage',
                    'english_translation'
            ]):
                logger.warning(
                    f"Missing required fields in passage: {passage}")
                continue

            cleaned_passage = {
                'section':
                passage['section'].strip(),
                'topic':
                passage['topic'].strip(),
                'number':
                passage['number'].strip(),
                'passage':
                clean_hebrew_text(passage['passage']),
                'english_translation':
                passage['english_translation'].strip()
                if passage.get('english_translation') else '',
                'average_score':
                passage.get('average_score', 0),
                'reference':
                passage.get('reference', '')  # Add reference if available
            }
            cleaned_selected_passages.append(cleaned_passage)

        # Debug logging
        logger.info(
            f"Number of cleaned passages: {len(cleaned_selected_passages)}")

        # Save final selections with more detailed data
        final_output = {
            "question_id": question_id,
            "question": question,
            "selected_passages": cleaned_selected_passages,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "total_passages_processed": len(all_responses),
                "selection_criteria":
                f"Passages with average score above {MINIMUM_SCORE_THRESHOLD}",
                "scoring_version": "3.0"
            }
        }

        selections_file = step_3_folder / "final_selections.json"
        with lock:  # Using the global lock for thread-safe file writing
            with selections_file.open("w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=4)

        # Save all responses for debugging with more detail
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
        with lock:  # Using the global lock for thread-safe file writing
            with debug_file.open("w", encoding="utf-8") as f:
                json.dump(debug_output, f, ensure_ascii=False, indent=4)

        logger.info(f"[green]Saved final results to {selections_file}[/green]")
        return selections_file

    except Exception as e:
        logger.error(f"[red]Error saving final results: {str(e)}[/red]")
        raise


def main(question_id=None):
    try:
        console.print(
            Panel.fit("[yellow]Step 3: Final Passage Selection[/yellow]"))

        # Load environment variables and initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "[red]OPENAI_API_KEY not found in environment variables[/red]")

        client = OpenAI(api_key=api_key)

        # Get latest question ID
        question_id = get_latest_question_id()
        logger.info(f"[cyan]Processing question ID: {question_id}[/cyan]")

        # Load queried results from step 2
        step_2_folder = Path("data/answers") / question_id / "step_2"
        queried_results_path = step_2_folder / "queried_results.json"

        if not queried_results_path.exists():
            raise FileNotFoundError(
                f"[red]Queried results not found at {queried_results_path}[/red]"
            )

        with queried_results_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        question = data.get("question")
        if not question:
            raise ValueError(
                "[red]Question not found in queried results[/red]")

        matched_passages = data.get("matched_passages", [])
        if not matched_passages:
            raise ValueError(
                "[red]No matched passages found in queried results[/red]")

        # Create mapping of original passages
        original_passages = {}
        for passage in matched_passages:
            section = passage.get('section', 'Unknown Section').strip()
            topic = passage.get('topic', 'Unknown Topic').strip()
            number = passage.get('number', '0').strip()
            reference = f"Tiferet Shlomo, on {section}, {topic} {number}"
            passage['reference'] = reference
            original_passages[reference] = passage

        # Prepare passage batches
        batches = prepare_passage_batches(matched_passages)
        logger.info(f"[cyan]Processing {len(batches)} batches...[/cyan]")

        all_responses = []

        # Process batches with ThreadPoolExecutor
        with Progress(SpinnerColumn(),
                      TextColumn("[progress.description]{task.description}"),
                      console=console) as progress:
            task = progress.add_task("[cyan]Processing batches...",
                                     total=len(batches))

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_batch = {
                    executor.submit(process_single_batch, client, batch, question, i):
                    i
                    for i, batch in enumerate(batches)
                }

                for future in as_completed(future_to_batch):
                    try:
                        result = future.result()
                        all_responses.append(result)
                        progress.advance(task)
                    except Exception as e:
                        logger.error(f"[red]Error processing batch: {e}[/red]")

        # Collect and process scores
        passage_scores = {}
        valid_responses = 0
        for response in all_responses:
            if "response" in response and "error" not in response:
                valid_responses += 1
                batch_response = response["response"]
                for item in batch_response:
                    # Look for either 'reference' or 'source' key
                    reference = item.get('reference') or item.get('source')
                    score = item.get('score')

                    if reference and score is not None:
                        try:
                            score = float(score)
                            if reference in passage_scores:
                                passage_scores[reference].append(score)
                            else:
                                passage_scores[reference] = [score]
                        except ValueError:
                            logger.error(
                                f"[red]Invalid score value for reference {reference}: {score}[/red]"
                            )
        # Calculate average scores and prepare final passages
        selected_passage_data = []
        averaged_scores = []

        for reference, scores in passage_scores.items():
            if scores:  # Only process if we have scores
                average_score = sum(scores) / len(scores)
                averaged_scores.append({
                    'reference': reference,
                    'average_score': average_score
                })

        # Filter passages by minimum score threshold
        filtered_passages = [
            p for p in averaged_scores
            if p['average_score'] >= MINIMUM_SCORE_THRESHOLD
        ]

        # Sort the filtered passages by average score in descending order
        filtered_passages.sort(key=lambda x: x['average_score'], reverse=True)

        # Select top passages up to TARGET_PASSAGES limit
        top_passages = filtered_passages[:TARGET_PASSAGES]

        logger.info(
            f"[green]Selected {len(top_passages)} passages with average score above {MINIMUM_SCORE_THRESHOLD}[/green]"
        )

        # Prepare final passage data
        for passage_score in top_passages:
            reference = passage_score['reference']
            if reference in original_passages:
                passage_data = original_passages[reference].copy()
                passage_data['average_score'] = passage_score['average_score']
                selected_passage_data.append(passage_data)
                logger.info(
                    f"[cyan]Added passage {reference} with score {passage_score['average_score']}[/cyan]"
                )
            else:
                logger.warning(
                    f"[yellow]Could not find original passage data for {reference}[/yellow]"
                )

        # Save results
        if selected_passage_data:
            output_file = save_final_results(question_id, question,
                                             selected_passage_data,
                                             all_responses)
            console.print(
                Panel.fit(
                    f"[green]Successfully processed and saved results to {output_file}[/green]"
                ))
        else:
            raise ValueError(
                f"[red]No passages were selected with average score above {MINIMUM_SCORE_THRESHOLD}[/red]"
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
