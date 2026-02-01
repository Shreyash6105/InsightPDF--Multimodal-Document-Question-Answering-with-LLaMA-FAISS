# Multimodal Chat with PDFs (Text, Tables, Images)

## Purpose

The purpose of this project is to build a **multimodal document intelligence system** that allows querying a PDF using its **text, images, and tables**, while ensuring answers are **grounded strictly in the document**.

This project demonstrates how a real-world **Retrieval-Augmented Generation (RAG)** pipeline works using **open-source LLMs and tools only**.

---

## What This Project Does

- Takes a PDF as input
- Extracts:
  - Text (page-wise)
  - Images (page-wise)
  - Tables
- Saves extracted data to disk
- Uses an LLM (LLaMA via Ollama) to generate summaries
- Converts summaries into embeddings using `bge-m3`
- Stores embeddings in FAISS
- Allows users to ask questions
- Retrieves relevant text, images, and tables
- Generates grounded answers with sources

---

## Folder Structure

Documents/sample.pdf

Text/ # Created automatically

images/ # Created automatically

tables/ # Created automatically

llm/llama_client.py

embeddings/ollama_embeddings.py

vectorstore/build_index.py (index.faiss and doc_ids.pkl)

query/query.py

1.text_extraction.py

2.text_to_file.py

3.image_extraction_storage.py

4.table_extraction_storage.py

5.summarization_layer.py  # Reference only (do not run)

README.md



-----

## Execution Flow:-

Follow the steps **in order**.

---

## Step 0: Add PDF

Place the PDF inside the `Documents/` folder.
`Documents/sample.pdf`

---

## Step 1: Test Text Extraction

Used only to test and understand text extraction.
`python 1_text_extraction.py`

- Prints extracted text to console
- Does not save files

---

## Step 2: Save Extracted Text

Saves page-wise text into the `Text/` folder.
`python 2_text_to_file.py`

---

## Step 3: Extract and Save Images

Extracts images from the PDF and saves them.
`python 3_image_extraction_storage.py`

---

## Step 4: Extract and Save Tables

Extracts tables and saves them as CSV files.
`python 4_table_extraction_storage.py`

---

## Step 5: LLM Setup (Manual)

Create a folder named `llm` and add LLM calling code.
`llm/llama_client.py`

Used to:
- Call LLaMA via Ollama
- Generate summaries
- Generate final answers

`5_summarization_layer.py` is **only for reference** and should **not** be executed.

---

## Step 6: Embeddings Setup

Create the `embeddings` folder.
`embeddings/ollama_embeddings.py`

- Contains `create_embedding()` function
- Uses `bge-m3` model via Ollama

---

## Step 7: Build Vector Store

Generates embeddings and builds FAISS index.
`python vectorstore/build_index.py`

Creates:
`vectorstore/index.faiss`
`vectorstore/doc_ids.pkl`

These files can be deleted and rebuilt if data changes.

---

## Step 8: Query the Document

Main entry point for asking questions.
`python query/query.py`

- Accepts a question
- Retrieves relevant text, tables, and images
- Generates a grounded answer
- Prints sources

---


---

## Out-of-Document Questions

If a question cannot be answered from the PDF, the system responds:


---

## Technologies Used

- Python
- PyMuPDF
- Camelot
- Ollama (LLaMA, bge-m3)
- FAISS
- Pandas

---

## Author
#### Shreyash More 
