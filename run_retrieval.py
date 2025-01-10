from datetime import datetime
import json
import os
import sys

import instructor
import litellm
from retrieval.initial_relevance import RelevanceChecker
from retrieval.datasource import DataSource

# Load env vars from .env
from dotenv import load_dotenv

load_dotenv()
print("Loaded environment variables from .env")


model_list = [
    {
        "model_name": "gpt-4o-mini",
    }
]

aclient: instructor.AsyncInstructor = instructor.from_openai(
    client=litellm.AsyncOpenAI(),
    mode=instructor.Mode.JSON,
)


async def main():

    run_params = {
        "model_list": model_list,
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

    results = await relevance_checker.check_relevance(
        aclient=aclient, query=question, passages=datasource.passages
    )

    # Sort by relevance score
    results.sort(key=lambda x: x.score, reverse=True)

    # Cutoff at score
    results = [
        result for result in results if result.score >= run_params["cutoff_score"]
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
                f"{result.passage.book_name},{result.passage.section},{result.passage.topic},{result.passage.torah_number},{result.passage.passage_number},{result.passage.hebrew_text},{result.score},{result.reason}\n"
            )

    # Dump the results to a JSON file
    json_obj = {
        "question": question,
        "run_params": run_params,
        "timstamp": timestamp,
        "results": [
            {
                "book_name": result.passage.book_name,
                "section": result.passage.section,
                "topic": result.passage.topic,
                "torah_number": result.passage.torah_number,
                "passage_number": result.passage.passage_number,
                "hebrew_text": result.passage.hebrew_text,
                "score": result.score,
                "reason": result.reason,
            }
            for result in results
        ],
    }
    with open(f"{folder}/results.json", "w") as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=4)

    for result in results:
        # Print the id (including book, topic, section, etc), passage and the score
        print(
            f"{result.passage.book_name}, {result.passage.section}, {result.passage.topic}, {result.passage.torah_number}, {result.passage.passage_number}"
        )
        print(result.passage.hebrew_text)
        print(result.score)
        print(result.reason)
        # Separator
        print("=" * 80)


# run main
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
