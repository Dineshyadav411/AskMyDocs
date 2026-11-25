from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rag import RAGStore

app = FastAPI(title="AskMyDocs")
store = RAGStore()

class Query(BaseModel):
    question: str
    top_k: int = 3

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    store.add_pdf(data)
    return {"status": "ok", "chunks": len(store.chunks)}

@app.post("/ask")
def ask(q: Query):
    results = store.search(q.question, q.top_k)
    # simple extractive answer: return top chunks with citations
    answer = "\n\n".join(
        f"[Page {r['page']}] {r['text']}" for r in results
    ) if results else "No relevant content found."
    return {"answer": answer, "sources": results}