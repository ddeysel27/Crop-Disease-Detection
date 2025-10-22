# Build a FAISS index from local PDFs or text for treatment/prevention retrieval.
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss, json, os

def load_docs(text_dir="app/kb_texts"):
    texts = []
    for p in Path(text_dir).glob("*.txt"):
        texts.append((p.name, p.read_text(encoding="utf-8")))
    return texts

def split_into_chunks(text, chunk_size=700, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start+chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0: start = 0
        if start >= len(text): break
    return chunks

def main():
    kb_dir = Path("app/kb_texts"); kb_dir.mkdir(exist_ok=True, parents=True)
    texts = load_docs(kb_dir)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    corpus = []
    meta = []
    for fname, full in texts:
        for i, chunk in enumerate(split_into_chunks(full)):
            corpus.append(chunk)
            meta.append({"source": fname, "chunk": i})
    embs = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, "app/kb.index")
    Path("app/kb_meta.json").write_text(json.dumps({"meta": meta, "corpus": corpus}, indent=2))
    print("Wrote app/kb.index and app/kb_meta.json")

if __name__ == "__main__":
    main()
