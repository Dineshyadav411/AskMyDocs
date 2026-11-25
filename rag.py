import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

class RAGStore:
    def __init__(self):
        self.chunks = []      # list of dicts: {"text":..., "page":...}
        self.embeddings = None

    def add_pdf(self, file_bytes: bytes):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        new_chunks = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if not text:
                continue
            # simple chunking by ~500 chars
            for i in range(0, len(text), 500):
                new_chunks.append({"text": text[i:i+500], "page": page_num})
        self.chunks.extend(new_chunks)
        texts = [c["text"] for c in self.chunks]
        self.embeddings = model.encode(texts, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 3):
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        q = model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q          # cosine similarity
        idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"text": self.chunks[i]["text"], "page": self.chunks[i]["page"],
             "score": float(scores[i])}
            for i in idx
        ]