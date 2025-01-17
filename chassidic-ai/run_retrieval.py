from datetime import datetime
import json
import os

from .retrieval.relevance.initial_relevance import RelevanceChecker, CandidatePassage
from .retrieval.datasources.csv_datasource import DataSource

# Load env vars from .env
from dotenv import load_dotenv

load_dotenv()
print("Loaded environment variables from .env")


async def main():

    run_params = {
        # "model_list": model_list,
        "max_parallelism": 4,
        "max_block_tokens_size": 16000,
        "dataset_file_path": "data/dataset.csv",
        "cutoff_score": 7.0,
    }

    datasource = DataSource(run_params["dataset_file_path"])
    datasource.read_csv()

    relevance_checker = RelevanceChecker(
        max_parallelism=run_params["max_parallelism"],
        max_block_tokens_size=run_params["max_block_tokens_size"],
    )
    question = "חידוש על פרשת ויחי"

    # Convert passages to CandidatePassage objects
    candidate_passages = [
        CandidatePassage(
            dataset="other_books.texts_from_csv",
            compound_id=passage.get_id(),
            book_name=passage.book_name,
            section=passage.section,
            topic=passage.topic,
            torah_num=int(passage.torah_number),
            passage_num=int(passage.passage_number),
            hebrew_text=passage.hebrew_text,
            en_translation=passage.translation,
            he_summary=passage.summary,
            en_summary=passage.summary,
            keywords=passage.keywords.split("\n"),
            relevance_judgments=[],
        )
        for passage in datasource.passages
    ]

    results = await relevance_checker.check_relevance(
        query=question, passages=candidate_passages
    )

    # Sort by relevance score
    results.sort(key=lambda x: x.relevance_judgments[0].score, reverse=True)

    # Cutoff at score
    results = [
        result
        for result in results
        if result.relevance_judgments[0].score >= run_params["cutoff_score"]
    ]

    # Create a new folder with timestamp as the name under runs/
    os.makedirs("runs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = f"runs/{timestamp}"
    os.makedirs(folder, exist_ok=True)

    # Save the question, results and the passages to the folder. One JSON file with everything + one text file with question + one CSV with results.
    with open(f"{folder}/question.txt", "w") as f:
        f.write(question)

    with open(f"{folder}/results.csv", "w") as f:
        f.write(
            "book_name,section,topic,torah_number,passage_number,hebrew_text,score,reason\n"
        )
        for result in results:
            f.write(
                f"{result.book_name},{result.section},{result.topic},{result.torah_num},{result.passage_num},{result.hebrew_text},{result.relevance_judgments[0].score},{result.relevance_judgments[0].reason}\n"
            )

    # Dump the results to a JSON file
    json_obj = {
        "question": question,
        "run_params": run_params,
        "timstamp": timestamp,
        "results": [
            {
                "book_name": result.book_name,
                "section": result.section,
                "topic": result.topic,
                "torah_number": result.torah_num,
                "passage_number": result.passage_num,
                "hebrew_text": result.hebrew_text,
                "score": result.relevance_judgments[0].score,
                "reason": result.relevance_judgments[0].reason,
            }
            for result in results
        ],
    }
    with open(f"{folder}/results.json", "w") as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=4)

    for result in results:
        # Print the id (including book, topic, section, etc), passage and the score
        print(
            f"{result.book_name}, {result.section}, {result.topic}, {result.torah_num}, {result.passage_num}"
        )
        print(result.hebrew_text)
        print(result.relevance_judgments[0].score)
        print(result.relevance_judgments[0].reason)
        # Separator
        print("=" * 80)


# run main
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
