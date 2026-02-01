import faiss
import pickle
import numpy as np
import pandas as pd

import sys
import os
# Add project root to sys.path as folder "embeddings" and "llm" is sister folder of "query"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embeddings.ollama_embeddings import create_embedding
from llm.llama_client import call_llama

# load FAISS index
index = faiss.read_index("vectorstore/index.faiss")

# load doc_id mapping
with open("vectorstore/doc_ids.pkl", "rb") as f:
    doc_ids = pickle.load(f)


def embed_query(question: str):
    embedding = create_embedding([question])
    return np.array(embedding).astype("float32")


#finds top k most relevant documents
# def retrieve_doc_ids(question, top_k=5):
#     query_vector = embed_query(question)
#     distances, indices = index.search(query_vector, top_k)

#     results = []
#     for idx in indices[0]:
#         results.append(doc_ids[idx])

#     return results
#update
def retrieve_doc_ids(question, top_k=10):
    query_vector = embed_query(question)
    distances, indices = index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        did = doc_ids[idx]
        dtype = did.split("::")[2]   # text / table / image
        results.append((did, dtype))

    return results





def load_content(doc_id):
    parts = doc_id.split("::")

    dtype = parts[2]

    if dtype == "text":
        page_num = parts[1].replace("page_", "")
        path = f"Text/page_{page_num}.txt"

        if not os.path.exists(path):
            return f"[ERROR: TEXT FILE NOT FOUND: {path}]"

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if dtype == "table":
        table_index = parts[3]
        path = f"tables/table_{table_index}.csv"

        if not os.path.exists(path):
            return f"[ERROR: TABLE FILE NOT FOUND: {path}]"

        return pd.read_csv(path).to_string()

    if dtype == "image":
        page_num = parts[1].replace("page_", "")
        image_index = parts[3]
        path = f"images/page_{page_num}_image_{image_index}.png"

        if not os.path.exists(path):
            return f"[ERROR: IMAGE FILE NOT FOUND: {path}]"

        return f"[IMAGE FILE: {path}]"
    
    return "[WARNING: Unsupported or missing content]"



#prompting llm
def build_prompt(question, retrieved_contents):
    # context = "\n\n".join(retrieved_contents[:4])
    clean_contents = [c for c in retrieved_contents if isinstance(c, str) and c.strip()]
    context = "\n\n".join(clean_contents[:4])

    prompt = f"""
    You are a helpful AI assistant answering questions about a PDF document.

    Use the provided context to answer the question as best as possible.
    The context may be partial or summarized.

    Rules:
    - Base your answer ONLY on the given context
    - If the context is partially relevant, infer carefully
    - Do NOT say "I don't know" unless the context is completely irrelevant
    - Be concise and factual

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    return prompt



# def ask(question):
#     retrieved_ids = retrieve_doc_ids(question)
#     contents = [load_content(did) for did in retrieved_ids]

#     prompt = build_prompt(question, contents)
#     answer = call_llama(prompt)

#     return answer, retrieved_ids

#update
def ask(question):
    retrieved = retrieve_doc_ids(question)

    selected = []
    seen = set()

    for did, dtype in retrieved:
        if dtype not in seen:
            selected.append(did)
            seen.add(dtype)

        # we want at most 1 text + 1 table + 1 image
        if len(seen) >= 3:
            break

    # fallback: ensure at least one text
    if not any("::text::" in d for d in selected):
        for did, dtype in retrieved:
            if dtype == "text":
                selected.append(did)
                break

    contents = []

    for did in selected:
        content = load_content(did)
        if content and isinstance(content, str):
            contents.append(content)

    prompt = build_prompt(question, contents)
    answer = call_llama(prompt)

    return answer, selected



#asking actual queries
# if __name__ == "__main__":
#     questions = [
#         "What is this document about?",
#         "What does the market analysis section discuss?",
#         "What metrics are shown in the tables?",
#         "Is there any chart related to sales or growth?",
#         "What do the text and visuals together indicate?"
#     ]

#     for q in questions:
#         print("\nQUESTION:", q)
#         answer, sources = ask(q)
#         print("ANSWER:\n", answer)
#         print("SOURCES:")
#         for s in sources:
#             print(" -", s)


#for file named attention-is-all-you-need
if __name__ == "__main__":
    questions = [
        "What are the two main components of the Transformer architecture? ",
        "How many layers does the base Transformer model use in both encoder and decoder? ",
        "What is the dimensionality (dmodel) used in the base Transformer model? ",

        "What is the formula for Scaled Dot-Product Attention? ",
        "Why do the authors scale the dot products by 1/√dk in their attention mechanism? ",
        "How many attention heads does the Transformer use, and what is the dimension of each head? ",

        "According to Table 1, what are the main advantages of self-attention layers compared to recurrent and convolutional layers in terms of computational complexity and parallelization? "
    ]

    for q in questions:
        print("\nQUESTION:", q)
        answer, sources = ask(q)
        print("ANSWER:\n", answer)
        print("SOURCES:")
        for s in sources:
            print(" -", s)
