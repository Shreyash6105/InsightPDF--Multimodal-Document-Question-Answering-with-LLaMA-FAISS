import fitz  # PyMuPDF
import os

PDF_PATH = "Documents/attention-is-all-you-need.pdf"          # path to your PDF
OUTPUT_DIR = "images"         # where text files will be saved

# create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

doc = fitz.open(PDF_PATH)

print(f"Total pages: {len(doc)}")

for page_number, page in enumerate(doc):
    images = page.get_images(full=True)
    
    if not images:
        continue

    for img_index , img in enumerate(images):
        reference = img[0]
        
        image_dict = doc.extract_image(reference)
        
        image_bytes = image_dict["image"]
        image_extension = image_dict["ext"]  #can be jpg,png,etc
        
        image_no = img_index + 1 
        page_no = page_number + 1
        
        image_name = f"page_{page_no}_image_{image_no}.{image_extension}"
        image_path = os.path.join(OUTPUT_DIR , image_name)
        
        #wb refers to write binary
        with open(image_path , "wb") as f:
            f.write(image_bytes)
            
        print(f"saved image : {image_path}")