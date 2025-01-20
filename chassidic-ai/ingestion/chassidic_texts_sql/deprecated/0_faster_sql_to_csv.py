import argparse
import re
import csv
import json
import sys


def parse_tags(sql_array: str) -> str:
    """
    Convert something like {tagA,"tag B"} into a JSON array string ["tagA","tag B"].
    """
    inside = sql_array.strip("{}")
    parts = []
    current = []
    in_quotes = False
    i = 0
    while i < len(inside):
        c = inside[i]
        if c == '"':
            in_quotes = not in_quotes
            i += 1
            continue
        elif c == "," and not in_quotes:
            parts.append("".join(current).strip())
            current = []
            i += 1
            continue
        else:
            current.append(c)
            i += 1
    if current:
        parts.append("".join(current).strip())

    parts_clean = []
    for p in parts:
        p = p.strip().strip('"').strip("'")
        if p:
            parts_clean.append(p)
    return json.dumps(parts_clean, ensure_ascii=False)


def parse_embeddings(embedding_text: str):
    """
    Convert bracketed float list, e.g. '[-0.01,0.5,...]' => Python list of floats.
    """
    # treat as JSON, replacing single quotes if necessary:
    return json.loads(embedding_text.replace("'", '"'))


def parse_value(s: str) -> str:
    """
    Convert SQL NULL => '', remove single quotes if present around entire string.
    """
    s = s.strip()
    if s.upper() == "NULL":
        return ""
    if len(s) >= 2 and s.startswith("'") and s.endswith("'"):
        s = s[1:-1]
    return s


def split_insert_columns(row_content: str) -> list:
    """
    State-machine approach to split a single insert-values row into columns.
    Improves upon the naive version by handling SQL-escaped single quotes: '' => one literal quote.
    """
    columns = []
    current = []
    in_quotes = False
    i = 0
    length = len(row_content)

    while i < length:
        c = row_content[i]

        # Handle an escaped single quote '' => literal quote, don't toggle
        # This is: if we see `'`, and the next char is `'`, treat as one literal `'`.
        if c == "'" and (i + 1 < length) and row_content[i + 1] == "'":
            # It's an escaped quote => just add `'` to current text, skip next
            current.append("'")
            i += 2
            continue

        if c == "'":
            # Toggle in_quotes
            in_quotes = not in_quotes
            current.append(c)
        elif c == "," and not in_quotes:
            # Column boundary
            columns.append("".join(current).strip())
            current = []
        else:
            current.append(c)

        i += 1

    # last column
    if current:
        columns.append("".join(current).strip())
    return columns


def split_values_row(row_text):
    result = []
    current = []
    depth = 0
    in_quotes = False
    escape_next = False

    for char in row_text.strip():
        if escape_next:
            current.append(char)
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            current.append(char)
            continue

        if char == '"' and not escape_next:
            in_quotes = not in_quotes
            current.append(char)
        elif char == "(" and not in_quotes:
            depth += 1
            current.append(char)
        elif char == ")" and not in_quotes:
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0 and not in_quotes:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        result.append("".join(current).strip())
    return result


def extract_insert_blocks(sql_text):
    blocks = []
    current_block = []
    in_insert = False

    for line in sql_text.splitlines():
        line = line.strip()
        if not line or line.startswith("--"):
            continue

        if "INSERT INTO" in line.upper():
            in_insert = True
            current_block = [line]
        elif in_insert:
            current_block.append(line)
            if line.endswith(";"):
                blocks.append(" ".join(current_block))
                in_insert = False

    return blocks


def extract_values(insert_block):
    values_start = insert_block.upper().find("VALUES")
    if values_start == -1:
        return []

    values_text = insert_block[values_start:].strip("VALUES ").strip(";")
    rows = []
    current_row = []
    depth = 0
    in_quotes = False

    for char in values_text:
        if char == '"':
            in_quotes = not in_quotes
        elif char == "(" and not in_quotes:
            depth += 1
            if depth == 1:
                current_row = []
                continue
        elif char == ")" and not in_quotes:
            depth -= 1
            if depth == 0:
                rows.append("".join(current_row))
                continue

        if depth > 0:
            current_row.append(char)

    return rows


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

    # 1) Read entire file
    try:
        with open(args.sql_file, "r", encoding="utf-8") as f:
            sql_text = f.read()
    except OSError as e:
        print(f"Error reading file '{args.sql_file}': {e}", file=sys.stderr)
        sys.exit(1)

    # CSV column order:
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

    parsed_rows = []
    sql_blocks = extract_insert_blocks(sql_text)

    for block in sql_blocks:
        if "chassidus_sentences" not in block:
            continue
        rows = extract_values(block)
        for row in rows:
            values = split_values_row(row)
            if len(values) == 14:  # Expected number of columns
                values = [parse_value(v) for v in values]
                values[7] = parse_tags(values[7])
                values[13] = parse_embeddings(values[13])
                parsed_rows.append(values)

    # Write to CSV
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(parsed_rows)

    # 4) Print total
    print(f"Done. Parsed a total of {len(parsed_rows)} records.")


if __name__ == "__main__":
    main()
