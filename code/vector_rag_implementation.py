import os
import faiss
import pickle
import numpy as np
from PyPDF2 import PdfReader
import google.generativeai as genai

import os
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# CONFIG
# -------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")
API_KEY = os.getenv("API_KEY")

VECTOR_DIM = os.getenv("VECTOR_DIM")
FAISS_DB_PATH = os.getenv("FAISS_DB_PATH")
TEXT_STORE_PATH = os.getenv("TEXT_STORE_PATH")

genai.configure(api_key=API_KEY)

# -------------------------
# PDF READING
# -------------------------
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# -------------------------
# SPLIT TEXT INTO CHUNKS
# -------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# -------------------------
# GEMINI EMBEDDINGS
# -------------------------
def get_embedding(text):
    response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text
    )
    return response['embedding']

# -------------------------
# CREATE FAISS INDEX
# -------------------------
def create_faiss_index(chunks):
    # Get dimension dynamically from the first chunk
    first_embedding = np.array(get_embedding(chunks[0]), dtype='float32')
    VECTOR_DIM = first_embedding.shape[0]
    print(f"Embedding dimension detected: {VECTOR_DIM}")

    index = faiss.IndexFlatL2(VECTOR_DIM)
    texts = []

    for chunk in chunks:
        embedding = np.array(get_embedding(chunk), dtype='float32')
        if embedding.shape[0] != VECTOR_DIM:
            raise ValueError(f"Embedding dimension {embedding.shape[0]} != {VECTOR_DIM}")
        index.add(embedding.reshape(1, VECTOR_DIM))
        texts.append(chunk)

    os.makedirs(os.path.dirname(FAISS_DB_PATH), exist_ok=True)

    # Save text store
    with open(TEXT_STORE_PATH, "wb") as f:
        pickle.dump(texts, f)
    # Save FAISS index
    faiss.write_index(index, FAISS_DB_PATH)
    return index, VECTOR_DIM

# -------------------------
# LOAD FAISS INDEX & TEXTS
# -------------------------
def load_faiss_index():
    index = faiss.read_index(FAISS_DB_PATH)
    with open(TEXT_STORE_PATH, "rb") as f:
        texts = pickle.load(f)
    return index, texts

# -------------------------
# QUERY RAG SYSTEM
# -------------------------
def query_rag(question, top_k=3):
    index, texts = load_faiss_index()
    q_emb = np.array(get_embedding(question), dtype='float32').reshape(1, index.d)
    D, I = index.search(q_emb, top_k)
    context = "\n\n".join([texts[i] for i in I[0]])

    model = genai.GenerativeModel(CHAT_MODEL)
    chat = model.start_chat(history=[])

    prompt=f"Answer the question based on the context below. Make the anser short, simple and precise.:\n\nContext:\n{context}\n\nQuestion: {question}"
    
    response = chat.send_message(prompt)
    return response.text