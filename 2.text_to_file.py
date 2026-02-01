import fitz  # PyMuPDF
import os

PDF_PATH = "Documents/attention-is-all-you-need.pdf"          # path to your PDF
OUTPUT_DIR = "Text"         # where text files will be saved

# create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

doc = fitz.open(PDF_PATH)

print(f"Total pages: {len(doc)}")

for page_number, page in enumerate(doc):
    text = page.get_text()

    page_no = page_number + 1 
    file_path = os.path.join(OUTPUT_DIR, f"page_{page_no}.txt")
    
    with open(file_path , "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"saved {file_path}")