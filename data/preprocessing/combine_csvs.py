import os
import sys
import csv
"""
Usage:
    python combine_csv.py <folder_path> [<output_csv>]

    - folder_path: The folder containing the CSV files to combine.
    - output_csv:  (Optional) The name/path of the output CSV file. Defaults to "dataset.csv" if not provided.
"""


def main():
    if len(sys.argv) < 2:
        print("Usage: python combine_csv.py <folder_path> [<output_csv>]")
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

    # Sort filenames if desired (optional):
    # csv_files.sort()

    # Initialize data storage
    all_rows = []
    master_fieldnames = None

    # Process each CSV file
    for filename in csv_files:
        filepath = os.path.join(folder_path, filename)

        with open(filepath, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            current_fieldnames = reader.fieldnames

            # If first CSV, record fieldnames
            if master_fieldnames is None:
                master_fieldnames = current_fieldnames
            else:
                # Check if fieldnames match
                if current_fieldnames != master_fieldnames:
                    print(
                        f"Error: The CSV file '{filename}' has a different set of columns."
                    )
                    print(f"Expected: {master_fieldnames}")
                    print(f"Found:    {current_fieldnames}")
                    sys.exit(1)

            # Append all rows from this file
            for row in reader:
                all_rows.append(row)

    # Write combined CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=master_fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Combined {len(csv_files)} CSV file(s) into '{output_csv}'.")


if __name__ == "__main__":
    main()
