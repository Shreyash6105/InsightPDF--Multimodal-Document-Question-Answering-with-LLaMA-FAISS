import faiss
import pickle
#pickle is python module for Converting Python objects (lists, dicts, custom classes) into a byte stream and Loading that byte stream back into Python objects.
import numpy as np
import pandas as pd
import sys
import os

# Add project root to sys.path as folder "embeddings" is sister folder of "vectorstore"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embeddings.ollama_embeddings import create_embedding
from llm.llama_client import call_llama

#example inputs
# summaries = [
#     "This page introduces the report and outlines key objectives.",
#     "This table compares sales metrics across different regions.",
#     "This image shows a bar chart of projected market growth."
# ]

# doc_ids = [
#     "report1::page_1::text::0",
#     "report1::page_3::table::0",
#     "report1::page_4::image::0"
# ]


#basically use "5.summarization_layer.py"
def summarize_text_page(text):
    prompt = f"""
    Summarize the following PDF page.
    Focus on key facts, definitions, and findings.
    Do not add explanations or opinions.

    Text:
    {text}
    """
    return call_llama(prompt)

#build summaries and doc_ids from extracted text
summaries = []
doc_ids = []

TEXT_DIR = "Text"

for filename in sorted(os.listdir(TEXT_DIR)):
    if not filename.endswith(".txt"):
        continue

    page_num = filename.replace("page_", "").replace(".txt", "")
    path = os.path.join(TEXT_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        page_text = f.read().strip()

    if not page_text:
        continue

    summary = summarize_text_page(page_text)

    summaries.append(summary)
    doc_ids.append(f"report1::page_{page_num}::text::0")




def summarize_table(table_text):
    prompt = f"""
    Describe what the following table represents.
    Mention key trends, comparisons, and notable values.

    Table:
    {table_text}
    """
    return call_llama(prompt)

TABLE_DIR = "tables"

for i, filename in enumerate(sorted(os.listdir(TABLE_DIR))):
    if not filename.endswith(".csv"):
        continue

    path = os.path.join(TABLE_DIR, filename)
    table_text = pd.read_csv(path).to_string()

    summary = summarize_table(table_text)

    summaries.append(summary)
    doc_ids.append(f"report1::table::table_{i}")


def summarize_image(image_name, page_num):
    prompt = f"""
    You are summarizing an image extracted from a PDF.

    Context:
    - Page number: {page_num}
    - Image file name: {image_name}

    Based on this context, describe what this image is likely to represent
    (e.g., chart, diagram, illustration) and its possible purpose in the document.
    """
    return call_llama(prompt)

IMAGE_DIR = "images"

for filename in sorted(os.listdir(IMAGE_DIR)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    # filename: page_4_image_0.png
    parts = filename.replace(".png", "").split("_")
    page_num = parts[1]
    image_index = parts[-1]

    summary = summarize_image(filename, page_num)

    summaries.append(summary)
    doc_ids.append(f"report1::page_{page_num}::image::{image_index}")




embeddings = create_embedding(summaries)

embedding_matrix = np.array(embeddings).astype("float32") #convert list of embeddings in numpy array as FAISS requires vectors in float32 format


#create and store FAISS index
dimension = embedding_matrix.shape[1] #no. of feature in each embeddings
index = faiss.IndexFlatL2(dimension)  #IndexFlatL2 is - FAISS index type that uses L2 (Euclidean) distance to measure similarity.
index.add(embedding_matrix)


#save index and metadata
faiss.write_index(index, "vectorstore/index.faiss")

with open("vectorstore/doc_ids.pkl", "wb") as f:
    pickle.dump(doc_ids, f)
    #saves doc_id alongside the index

print("FAISS index built successfully using Ollama embeddings")

#this will create "doc_ids.pkl" and "index.faiss" inside folder "vectorstore"
