import csv
from pydantic import BaseModel


class Passage(BaseModel):
    book_name: str
    section: str
    topic: str
    torah_number: str
    passage_number: str
    hebrew_text: str
    translation: str
    summary: str
    keywords: str

    def get_id(self):
        return f"{self.book_name}, {self.section}, {self.topic}, {self.torah_number}, {self.passage_number}"


class DataSource:
    def __init__(self, file_path):
        self.file_path = file_path
        self.passages: list[Passage] = []

    def read_csv(self):
        with open(self.file_path, mode="r", encoding="utf-8") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                passage = Passage(
                    book_name=row["book_name"],
                    section=row["section"],
                    topic=row["topic"],
                    torah_number=row["torah #"],
                    passage_number=row["passage #"],
                    hebrew_text=row["hebrew_text"],
                    translation=row["translation"],
                    summary=row["summary"],
                    keywords=row["keywords"],
                )
                self.passages.append(passage)


# Example usage:
# datasource = DataSource('/path/to/your/csvfile.csv')
# datasource.read_csv()
# print(datasource.passages)
