import os
import sys
import csv
"""
Usage:
    python data/preprocessing/combine_enriched_csvs.py data/enriched/ data/dataset.csv

This script merges all CSV files within <folder_path> into a single CSV file
(<output_csv>, defaulting to 'dataset.csv' if not provided). It normalizes
the rows according to these final columns:

    book_name, section, topic, torah #, passage #, hebrew_text, translation, summary, keywords

Mappings:
1) If book_name is missing, it defaults to "Divrey Yoel".
2) If section is missing, it defaults to "Torah".
3) topic is taken from 'topic' if it exists, else from 'parsha_name' if it exists.
4) torah # is taken from 'torah #' or 'dvar_torah_id'.
5) passage # is taken from 'passage #' or 'passage_id'.
6) hebrew_text is taken from 'hebrew_text' or 'passage_content'.
"""

FINAL_FIELDS = [
    "book_name", "section", "topic", "torah #", "passage #", "hebrew_text",
    "translation", "summary", "keywords"
]


def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_csv.py <folder_path> [<output_csv>]")
        sys.exit(1)

    folder_path = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "dataset.csv"

    # Validate folder exists
    if not os.path.isdir(folder_path):
        print(
            f"Error: The folder '{folder_path}' does not exist or is not a directory."
        )
        sys.exit(1)

    # Collect all CSV filenames
    csv_files = [
        f for f in os.listdir(folder_path) if f.lower().endswith('.csv')
    ]
    if not csv_files:
        print(f"No CSV files found in the folder '{folder_path}'.")
        sys.exit(0)

    # We'll store normalized rows here
    all_rows = []

    # Process each CSV file
    for filename in csv_files:
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                # Build a new row according to the FINAL_FIELDS
                new_row = {}

                # (1) book_name
                book_name = row.get('book_name', '').strip()
                if not book_name:
                    book_name = "Divrey Yoel"

                # (2) section
                section = row.get('section', '').strip()
                if not section:
                    section = "Torah"

                # (3) topic -> from 'topic' or fallback to 'parsha_name'
                topic = row.get('topic', '').strip()
                if not topic:
                    topic = row.get('parsha_name', '').strip()

                # (4) torah #
                torah_number = row.get('torah #', '').strip()
                if not torah_number:
                    torah_number = row.get('dvar_torah_id', '').strip()

                # (5) passage #
                passage_number = row.get('passage #', '').strip()
                if not passage_number:
                    passage_number = row.get('passage_id', '').strip()

                # (6) hebrew_text
                hebrew_text = row.get('hebrew_text', '').strip()
                if not hebrew_text:
                    hebrew_text = row.get('passage_content', '').strip()

                # translation, summary, keywords
                translation = row.get('translation', '').strip()
                summary = row.get('summary', '').strip()
                keywords = row.get('keywords', '').strip()

                new_row["book_name"] = book_name
                new_row["section"] = section
                new_row["topic"] = topic
                new_row["torah #"] = torah_number
                new_row["passage #"] = passage_number
                new_row["hebrew_text"] = hebrew_text
                new_row["translation"] = translation
                new_row["summary"] = summary
                new_row["keywords"] = keywords

                all_rows.append(new_row)

    # Write combined CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=FINAL_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(
        f"Combined {len(csv_files)} CSV file(s) into '{output_csv}' with {len(all_rows)} total rows."
    )


if __name__ == "__main__":
    main()
