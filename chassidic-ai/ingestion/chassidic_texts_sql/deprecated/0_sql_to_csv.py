import argparse
import re
import csv
import json
import sys


def parse_tags(sql_array):
    """
    Convert Postgres-style text array '{"tag1","tag2"}'
    into a JSON list ["tag1","tag2"].
    """
    inside = sql_array.strip("{}")
    # Split on commas that are not inside quotes:
    tag_parts = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', inside)
    tags_list = []
    for part in tag_parts:
        # Strip quotes, if any
        part = part.strip().strip('"').strip("'")
        if part:
            tags_list.append(part)
    return json.dumps(tags_list, ensure_ascii=False)


def parse_embeddings(embedding_text):
    """
    Parse bracketed float list, e.g. '[-0.0175, 0.024...]' => Python list of floats.
    """
    return json.loads(embedding_text.replace("'", '"'))


def parse_value(s):
    """
    Convert SQL `NULL` => empty string, strip quotes if needed.
    """
    s = s.strip()
    if s.upper() == "NULL":
        return ""
    # Remove leading/trailing single quotes if present
    if s.startswith("'") and s.endswith("'"):
        s = s[1:-1]
    return s


def main():
    parser = argparse.ArgumentParser(
        description="Parse INSERT statements from an SQL file into CSV."
    )
    parser.add_argument("sql_file", help="Path to the SQL file.")
    parser.add_argument(
        "-o",
        "--output",
        default="output.csv",
        help="Output CSV file (default: output.csv)",
    )
    args = parser.parse_args()

    # Read entire SQL text from file
    try:
        with open(args.sql_file, "r", encoding="utf-8") as f:
            sql_text = f.read()
    except OSError as e:
        print(f"Error reading file '{args.sql_file}': {e}", file=sys.stderr)
        sys.exit(1)

    # Regex to match each INSERT that references "chassidus_sentences",
    # capturing all the parenthesized row data up to the final semicolon.
    # This handles multiple statements if present.
    insert_pattern = re.compile(
        r'INSERT\s+INTO\s+"chassidus_sentences".*?VALUES\s+((?:\(.*?\)\s*,?\s*)+);',
        re.IGNORECASE | re.DOTALL,
    )

    # We'll store parsed rows from all insert statements
    all_parsed_rows = []

    # We define CSV columns in the order needed:
    headers = [
        "id",
        "chassidus_text_id",
        "sentence",
        "sentence_number",
        "context",
        "paragraph",
        "source",
        "tags",
        "sefaria_name",
        "created_at",
        "updated_at",
        "translation",
        "translation_version",
        "embedding_large_english_hebrew",
    ]

    # For each matching INSERT block
    for insert_block_match in insert_pattern.finditer(sql_text):
        block_text = insert_block_match.group(
            1
        )  # everything between VALUES and the trailing semicolon

        # Now we get each parenthesized row: (....)
        # We want to capture the text inside parentheses, ignoring trailing commas
        # or whitespace.
        row_pattern = re.compile(r"\((.*?)\)\s*,?\s*", re.DOTALL)
        rows = row_pattern.findall(block_text)

        for row_index, row_content in enumerate(rows, start=1):
            # split by commas not inside quotes
            parts = re.split(r",(?=(?:[^']*'[^']*')*[^']*$)", row_content)
            parts = [parse_value(p) for p in parts]

            # We expect 14 columns:
            if len(parts) != 14:
                # skip or raise an error if needed
                # but we skip gracefully
                continue

            # fix up columns (7 -> tags, 13 -> embeddings) in 0-based
            parts[7] = parse_tags(parts[7])  # tags
            parts[13] = parse_embeddings(parts[13])  # embedding

            all_parsed_rows.append(parts)

            # Print progress every 100 records
            if len(all_parsed_rows) % 100 == 0:
                print(f"Parsed {len(all_parsed_rows)} records so far...")

    # Write out to CSV
    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(all_parsed_rows)

    print(f"Done. Parsed a total of {len(all_parsed_rows)} records.")


if __name__ == "__main__":
    main()
