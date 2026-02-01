#pip install pymupdf
import fitz #python name of PyMuPDF 
pdf_path = 'Documents/attention-is-all-you-need.pdf'

doc = fitz.open(pdf_path)
#doc is not text
#doc is a Document object

print(f"Total pages: {len(doc)}")

#iterating through pages
for page_number, page in enumerate(doc):
    text = page.get_text()
    print("_"*40)
    print(f"Page {page_number}")
    print(f"Text length: {len(text)}")
    print(text[:300])  # preview first 300 characters