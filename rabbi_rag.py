import os
import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


def fetch_book(book: str):
    """Fetch English text of a book from Sefaria"""
    url = f"https://www.sefaria.org/api/texts/{book}?context=0&commentary=0&pad=0&lang=en"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    texts = []
    if isinstance(data.get('text'), list):
        for i, chapter in enumerate(data['text']):
            if isinstance(chapter, list):
                for j, verse in enumerate(chapter):
                    ref = f"{book} {i+1}:{j+1}"
                    texts.append((ref, verse))
            else:
                ref = f"{book} {i+1}"
                texts.append((ref, chapter))
    else:
        texts.append((book, data.get('text', '')))
    return texts


class RabbiRAG:
    """Minimal Retrieval-Augmented Generation using Sefaria"""
    def __init__(self, books, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.books = books
        self.model = SentenceTransformer(model_name)
        self.texts = []
        self.refs = []
        self.index = None

    def build(self):
        """Build the FAISS index from the selected books"""
        for book in self.books:
            passages = fetch_book(book)
            for ref, text in passages:
                self.refs.append(ref)
                self.texts.append(text)
        embeddings = self.model.encode(self.texts, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype('float32'))

    def search(self, query, top_k=5):
        """Search for relevant passages"""
        q_emb = self.model.encode([query])
        D, I = self.index.search(np.array(q_emb).astype('float32'), top_k)
        results = []
        for idx in I[0]:
            ref = self.refs[idx]
            results.append({
                'ref': ref,
                'text': self.texts[idx],
                'link': f"https://www.sefaria.org/{ref.replace(' ', '.')}"
            })
        return results

    def build_context(self, query, top_k=5):
        """Return formatted context for LLM"""
        results = self.search(query, top_k)
        context = "\n".join(f"{r['ref']}: {r['text']}" for r in results)
        return context, results

    def answer(self, query, llm, top_k=5):
        """Generate answer from LLM with citations"""
        context, results = self.build_context(query, top_k)
        prompt = (
            "You are an expert on Jewish philosophical and ethical works, specifically "
            "Derech Hashem (The Way of God) by Rabbi Moshe Chaim Luzzatto (Ramchal) and "
            "The Beginning of Wisdom. Answer the question using the provided passages from these works. "
            "Maintain the clear, structured approach characteristic of these systematic guides to "
            "understanding God, creation, providence, human purpose, and ethical living. "
            "If relevant, note which work and which section the answer comes from "
            "(e.g., Derekh Hashem on Creation/Providence/Divine Service, or The Beginning of Wisdom). "
            "Provide hyperlinks to sources.\n"
            f"Context:\n{context}\nQuestion: {query}\nAnswer:"
        )
        completion = llm(prompt)
        return completion, results
