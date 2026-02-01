import requests

#same code i used in one of my previous project named : RAG-Based_AI_Teaching_Assistant
def create_embedding(text_list):
    embeddings = []

    for text in text_list:
        if text is None or not str(text).strip():
            text = "empty content"

        r = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "bge-m3",
                "prompt": text
            }
        )

        data = r.json()

        if "embedding" not in data:
            r = requests.post(
                "http://localhost:11434/api/embeddings",
                json={
                    "model": "bge-m3",
                    "prompt": "placeholder text"
                }
            )
            data = r.json()

        embeddings.append(data["embedding"])

    return embeddings
