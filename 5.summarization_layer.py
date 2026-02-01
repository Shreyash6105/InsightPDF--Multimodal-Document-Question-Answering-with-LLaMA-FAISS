from llm.llama_client import call_llama

def summarize_text(page_text: str) -> str:
    prompt = f"""
    You are summarizing a PDF page.

    Focus on:
    - key facts
    - definitions
    - important findings

    Do not add explanations or opinions.

    Text:
    {page_text}
    """
    return call_llama(prompt)


def summarize_table(table_text: str) -> str:
    prompt = f"""
    You are summarizing a table extracted from a PDF.

    Describe:
    - what the table represents
    - key trends or comparisons
    - notable values

    Table:
    {table_text}
    """
    return call_llama(prompt)


def summarize_image(image_description: str) -> str:
    prompt = f"""
    You are describing an image from a PDF.

    Include:
    - what the image shows
    - charts, labels, or objects
    - what it represents conceptually

    Image description:
    {image_description}
    """
    return call_llama(prompt)


#example for page
# with open("Text/page_3.txt", "r", encoding="utf-8") as f:
#     page_text = f.read()
# summary = summarize_text(page_text)
# print(summary)


#example for image
#for now image description is manual, later can be automated using vision models


#example for table
# import pandas as pd
# df = pd.read_csv("tables/table_1.csv")
# table_text = df.to_string()
# summary = summarize_table(table_text)
# print(summary)
