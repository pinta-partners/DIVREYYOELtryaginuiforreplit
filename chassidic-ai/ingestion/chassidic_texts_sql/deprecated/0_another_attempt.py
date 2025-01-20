import argparse
import csv
import re
import os


def parse_sql_to_csv(input_file, output_file):
    try:
        with open(input_file, "r", encoding="utf-8") as sql_file:
            content = sql_file.read()

        # Extracting the table name and headers from the first CREATE TABLE statement
        create_table_match = re.search(
            r"CREATE TABLE\s+`?(\w+)`?\s*\((.*?)\)\s*;", content, re.S
        )
        if not create_table_match:
            raise ValueError("CREATE TABLE statement not found in the SQL file.")

        table_name = create_table_match.group(1)
        columns = [
            col.strip().split()[0].strip("`")
            for col in create_table_match.group(2).split(",")
        ]

        # Writing headers to CSV
        with open(output_file, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(columns)

            # Parsing INSERT INTO statements and extracting rows
            insert_statements = re.findall(
                rf"INSERT INTO `?{table_name}`?\s+VALUES\s*(\(.*?\));", content, re.S
            )

            total_rows = 0
            for statement in insert_statements:
                # Splitting multiple rows in a single statement
                rows = re.findall(r"\((.*?)\)", statement)
                for row in rows:
                    # Safely parsing individual row values
                    values = re.findall(r'(?:"(.*?)"|\'?(.*?)\'|([^,]+))(?:,|$)', row)
                    parsed_values = [v[0] or v[1] or v[2] for v in values]
                    writer.writerow(parsed_values)
                    total_rows += 1

                    if total_rows % 100 == 0:
                        print(f"Processed {total_rows} rows...")

        print(f"Finished processing. Total rows: {total_rows}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse an SQL dump file and export rows to a CSV."
    )
    parser.add_argument("input_file", help="Path to the SQL dump file.")
    parser.add_argument("output_file", help="Path to the output CSV file.")

    args = parser.parse_args()
    parse_sql_to_csv(args.input_file, args.output_file)
