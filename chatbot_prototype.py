import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from google import genai

os.environ["GEMINI_API_KEY"] = "AIzaSyARWeK-HH5BV2ZNPEbOrmmazZSz_-YgQAI"

try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")

EXCEL_PATH = "ATL.xlsx"
FAISS_INDEX_PATH = "fbr_faiss.index"
TEXTS_PATH = "fbr_texts.npy"
BATCH_SIZE = 64
TOP_K = 5
MODEL_NAME = "gemini-2.5-flash"

if os.path.exists(TEXTS_PATH) and os.path.exists(FAISS_INDEX_PATH):
    print("Loading existing FAISS index and texts...")
    texts = np.load(TEXTS_PATH, allow_pickle=True).tolist()
    index = faiss.read_index(FAISS_INDEX_PATH)
else:
    print("Reading Excel and generating embeddings...")
    try:
        df = pd.read_excel(EXCEL_PATH)
    except FileNotFoundError:
        print(f"Error: The file {EXCEL_PATH} was not found. Please ensure it exists.")
        exit()

    df["full_info"] = df.apply(
        lambda r: f"NTN={r.get('NTN', 'N/A')} | Name={r.get('Name', 'N/A')} | Status={r.get('Status', 'N/A')} | Income={r.get('Income', 'N/A')}",
        axis=1
    )
    texts = df["full_info"].tolist()

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batches"):
        batch = texts[i:i+BATCH_SIZE]
        emb = embedder.encode(batch)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    np.save(TEXTS_PATH, texts)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("FAISS index and embeddings saved.")

chat_history = []

print("\n" + "="*50)
print("FBR Accounting Chatbot (Hybrid) Ready!")
print(f"Using Model: {MODEL_NAME}")
print("Type 'quit' to exit")
print("="*50 + "\n")

embedder = SentenceTransformer("all-MiniLM-L6-v2") 

while True:
    query = input("You: ")
    if query.lower() == "quit":
        break

    q_vec = embedder.encode([query]).astype("float32")
    distances, ids = index.search(q_vec, TOP_K)
    retrieved_text = "\n".join([texts[i] for i in ids[0]])

    prompt = f"""
You are a tax and accounting assistant.

RULES:
1. You may use general tax knowledge (such as how income tax is calculated).
2. Also use the retrieved data below if relevant (NTN, Name, Business Name, STR number, Income, etc).
3. If both are available (user provides income AND Excel row has tax info), combine both logically.
4. If required info is missing, ask the user for clarification.
5. Be accurate, clear, and step-by-step in calculations.

RETRIEVED DATA (from Excel):
{retrieved_text}

USER QUESTION:
{query}

ANSWER:
"""


    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0 
            )
        )
        answer = response.text
    except Exception as e:
        answer = f"Error getting response from Gemini: {e}"

    print("\nBot:", answer)
    chat_history.append((query, answer))