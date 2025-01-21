import sys
import csv
from bs4 import BeautifulSoup
import argparse


# Usage: python data/preprocessing/1_docx_to_csv.py -book_name="Divrey Yoel" --parsha_name="Ba" --input_docx_file="data/raw_docx/בא.docx"
def main():
    parser = argparse.ArgumentParser(description="Convert HTML to CSV")
    parser.add_argument(
        "--book_name", help="Book name", required=False, default="Divrey Yoel"
    )
    parser.add_argument(
        "--parsha_name", help="Parsha name", required=False, default="Shemot"
    )
    parser.add_argument(
        "--input_html_file",
        required=False,
        help="Path to the input HTML file",
        default="data/raw_input_html/שמות.html",
    )
    args = parser.parse_args()

    with open(args.input_html_file, encoding="utf-8") as f:
        markup_text = f.read()
        # Replace all `</b> <b>` with space

    # Skip to start of the main content by skipping first four <body> tags
    markup_text = markup_text.split("<body", 4)[-1]

    # Unify bold tags
    markup_text = markup_text.replace("</b><b>", " ")
    markup_text = markup_text.replace("</b> <b>", " ")
    markup_text = markup_text.replace("</b> <br/><b>", " ")
    # Split by <b> tags
    markup_text_passages = markup_text.split("<b>")
    # Remove all other tags
    markup_text_passages = [
        BeautifulSoup(p, "html.parser").get_text() for p in markup_text_passages
    ]
    # Fill in row structure (passage_id is incremental)
    rows = [
        [args.book_name, args.parsha_name, i, i, p]
        for i, p in enumerate(markup_text_passages)
    ]

    writer = csv.writer(sys.stdout)
    writer.writerow(
        ["book_name", "parsha_name", "dvar_torah_id", "passage_id", "passage_content"]
    )
    for row in rows:
        writer.writerow(row)


if __name__ == "__main__":
    main()
