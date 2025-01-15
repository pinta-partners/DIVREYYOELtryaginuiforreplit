import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output
from PIL import Image
import re
import json
import os

# NOTE: This script is not currently used in the pipeline.
# NOTE: This script is not yet functional.

# Set Tesseract configuration for Hebrew
pytesseract.pytesseract.tesseract_cmd = (
    # r"/usr/local/bin/tesseract"  # Update with your tesseract path
    r"/opt/homebrew/bin/tesseract"
)

os.environ["TESSDATA_PREFIX"] = "/opt/homebrew/share/tessdata/"

tesseract_config = "--psm 6 -c preserve_interword_spaces=1 -l heb_rashi"


# Helper function to perform OCR on an image
def perform_ocr(page_image):
    text_data = pytesseract.image_to_data(
        page_image, output_type=Output.DICT, config=tesseract_config
    )
    return text_data


# Function to parse the text and extract metadata
def parse_text(page_text, page_number):
    paragraphs = []
    section_number = 0
    paragraph_number = 0
    current_section = []

    for line in page_text.splitlines():
        # Skip empty lines
        if not line.strip():
            continue

        # Check for asterisks indicating new section
        if "*" in line:
            if current_section:
                paragraphs.append(
                    {"Section_Number": section_number, "Paragraphs": current_section}
                )
            section_number += 1
            paragraph_number = 0
            current_section = []
            continue

        # Identify bold start (simulate detection via capitalization or manual tagging)
        bold_start_match = re.match(r"^\s*(\S+)", line)
        if bold_start_match:
            paragraph_number += 1
            current_section.append(
                {"Paragraph_Number": paragraph_number, "Text": line.strip()}
            )
        else:
            # Append text to the last paragraph
            if current_section:
                current_section[-1]["Text"] += " " + line.strip()
            else:
                # Handle edge cases
                paragraph_number += 1
                current_section.append(
                    {"Paragraph_Number": paragraph_number, "Text": line.strip()}
                )

    # Append the last section
    if current_section:
        paragraphs.append(
            {"Section_Number": section_number, "Paragraphs": current_section}
        )

    return paragraphs


# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path, start_page, end_page):
    structured_data = []
    with fitz.open(pdf_path) as pdf:
        for page_number in range(start_page, end_page + 1):
            print(f"Processing page {page_number}...")
            page = pdf[page_number - 1]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Perform OCR
            text_data = perform_ocr(img)
            page_text = " ".join(text_data["text"]).replace("\n", " ")

            # Parse text
            page_metadata = parse_text(page_text, page_number)
            structured_data.extend(page_metadata)

    return structured_data


# Main script
if __name__ == "__main__":
    pdf_file = (
        "data/raw_pdf/Hebrewbooks_org_21045.pdf"  # Replace with the actual file path
    )
    output_json = "parsed_text.json"
    start_page = 10
    # end_page = 663
    end_page = 25

    # Extract text and structure it
    structured_data = extract_text_from_pdf(pdf_file, start_page, end_page)

    # Save to JSON
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(structured_data, json_file, ensure_ascii=False, indent=4)

    print(f"Extraction completed. Data saved to {output_json}")
