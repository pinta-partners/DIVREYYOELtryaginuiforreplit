import csv
import sys
"""
Usage:
    python data/preprocessing/csv_to_text.py data/dataset.csv guider/datas
et.txt

This script reads an enriched CSV (with columns:
    book_name,section,topic,torah #,passage #,hebrew_text,translation,summary,keywords
 )


and creates a single text file. For each row, it writes:

==================================================
book_name, parsha_name, Torah #X, Passage #Y

Original Hebrew:
<passage_content>

**Translation:**
<translation>

**Summary:**
<summary>

**Keywords:**
1. ...
2. ...
...
10. ...
"""


def main():
    if len(sys.argv) != 3:
        print("Usage: python create_text_file.py <input_csv> <output_txt>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_txt = sys.argv[2]

    with open(input_csv, 'r', encoding='utf-8') as infile, \
         open(output_txt, 'w', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)

        for row in reader:
            # Write a separator
            outfile.write(
                "==================================================\n")
            outfile.write(
                f"{row['book_name']}, {row['topic']}, "
                f"Torah #{row['torah #']}, Passage #{row['passage #']}\n\n"
            )

            # Original Hebrew
            outfile.write("Original Hebrew:\n")
            outfile.write(f"{row['hebrew_text']}\n\n")

            # Translation
            outfile.write("**Translation:**\n")
            outfile.write(f"{row['translation']}\n\n")

            # Summary
            outfile.write("**Summary:**\n")
            outfile.write(f"{row['summary']}\n\n")

            # Keywords
            outfile.write("**Keywords:**\n")
            # The keywords might be newline-separated; handle them accordingly
            keywords_lines = row['keywords'].strip().split('\n')
            for i, keyword in enumerate(keywords_lines, start=1):
                outfile.write(f"{i}. {keyword.strip()}\n")
            outfile.write("\n")

    print(f"Text file created: {output_txt}")


if __name__ == "__main__":
    main()
