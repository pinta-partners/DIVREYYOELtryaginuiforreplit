import sys
import csv
import argparse
from docx import Document


def extract_passages(doc):
    current_dvar_torah = 0
    current_passage = []
    passages = []
    in_bold = False

    for paragraph in doc.paragraphs:
        if "*" in paragraph.text:
            current_dvar_torah += 1
            continue

        for run in paragraph.runs:
            text = run.text.strip()
            if not text:
                continue

            if run.bold and not in_bold:
                # New passage starts
                if current_passage:
                    passages.append((current_dvar_torah, "".join(current_passage)))
                current_passage = [text]
                in_bold = True
            elif not run.bold:
                current_passage.append(text)
                in_bold = False

    # Add last passage
    if current_passage:
        passages.append((current_dvar_torah, "".join(current_passage)))

    return passages


def main():
    parser = argparse.ArgumentParser(description="Convert DOCX to CSV")
    parser.add_argument(
        "--book_name", help="Book name", required=False, default="Divrey Yoel"
    )
    parser.add_argument(
        "--parsha_name", help="Parsha name", required=False, default="Shemot"
    )
    parser.add_argument(
        "--input_docx_file",
        required=False,
        help="Path to the input DOCX file",
        default="data/raw_input_docx/shemot.docx",
    )
    args = parser.parse_args()

    doc = Document(args.input_docx_file)
    passages = extract_passages(doc)

    # Generate rows with incrementing passage_ids
    rows = [
        [args.book_name, args.parsha_name, dvar_torah_id, i, content]
        for i, (dvar_torah_id, content) in enumerate(passages)
    ]

    writer = csv.writer(sys.stdout)
    writer.writerow(
        ["book_name", "parsha_name", "dvar_torah_id", "passage_id", "passage_content"]
    )
    for row in rows:
        writer.writerow(row)


if __name__ == "__main__":
    main()
