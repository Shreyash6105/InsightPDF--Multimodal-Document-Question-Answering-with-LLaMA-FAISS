import camelot
import os

PDF_PATH = "Documents/attention-is-all-you-need.pdf"          # path to your PDF
OUTPUT_DIR = "tables"       # where tables will be saved

os.makedirs(OUTPUT_DIR, exist_ok=True)

# read all tables from all pages
tables = camelot.read_pdf(PDF_PATH, pages="all")

print(f"Total tables found: {len(tables)}")

for i, table in enumerate(tables):
    table_number = i+1
    file_path = os.path.join(OUTPUT_DIR, f"table_{table_number}.csv")
    table.to_csv(file_path, index=False , encoding="utf-8")
    print(f"Saved table: {file_path}")
