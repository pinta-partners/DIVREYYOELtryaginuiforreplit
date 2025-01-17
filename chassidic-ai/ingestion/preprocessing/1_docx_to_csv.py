import sys
import csv
import argparse
from docx import Document

# Usage example:
# python chassidic-ai/ingestion/preprocessing/1_docx_to_csv.py --book_name "Divrey Yoel" --parsha_name "Shemot" --input_docx_file "data/raw_input_docx/שמות.docx" > data/raw_input_csv/divrey_yoel_shemot.csv
# python chassidic-ai/ingestion/preprocessing/1_docx_to_csv.py --book_name "Divrey Yoel" --parsha_name "Bo" --input_docx_file "data/raw_input_docx/בא.docx" > data/raw_input_csv/divrey_yoel_bo.csv
# python chassidic-ai/ingestion/preprocessing/1_docx_to_csv.py --book_name "Divrey Yoel" --parsha_name "Beshalach" --input_docx_file "data/raw_input_docx/בשלח.docx" > data/raw_input_csv/divrey_yoel_beshalach.csv
# python chassidic-ai/ingestion/preprocessing/1_docx_to_csv.py --book_name "Divrey Yoel" --parsha_name "Va'era" --input_docx_file "data/raw_input_docx/וארא.docx" > data/raw_input_csv/divrey_yoel_vaera.csv


def extract_passages(doc):
    current_dvar_torah = 0
    current_passage = []
    passages = []
    in_bold = False

    for paragraph in doc.paragraphs:
        has_asterisk = "*" in paragraph.text
        if not has_asterisk:  # Skip processing asterisk paragraphs
            for run in paragraph.runs:
                text = run.text.strip(" ")

                if not text:
                    continue

                if run.bold and not in_bold:
                    # New passage starts
                    if current_passage:
                        passages.append((current_dvar_torah, " ".join(current_passage)))
                    current_passage = [text]
                    in_bold = True
                elif not run.bold:
                    current_passage.append(text)
                    in_bold = False

        if has_asterisk and current_passage:  # End current dvar torah
            passages.append((current_dvar_torah, " ".join(current_passage)))
            current_passage = []
            current_dvar_torah += 1

    # Add last passage
    if current_passage:
        passages.append((current_dvar_torah, " ".join(current_passage)))

    return passages


def main():
    parser = argparse.ArgumentParser(description="Convert DOCX to CSV")
    parser.add_argument("--book_name", help="Book name", required=True)
    parser.add_argument("--parsha_name", help="Parsha name", required=True)
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
