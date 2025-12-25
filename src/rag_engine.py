import os
import re
import faiss
import pickle
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from src.config import Config


class ClinicalKnowledgeBase:
    """
    Core RAG Engine handling document ingestion, embedding, vector storage,
    and retrieval logic for the Clinical Assistant.
    """

    def __init__(self):
        print(f"Loading Embedding Model: {Config.EMBEDDING_MODEL}...")
        self.encoder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.index = None
        # Stores metadata: [{'text': "...", 'page': 1, 'source': "guidelines.pdf"}, ...]
        self.chunks = []

    # --- Core Logic: Ingestion ---
    def load_and_process_pdfs(self, pdf_paths: list[str]):
        """Ingests PDFs, tracks page numbers & filenames, chunks them, and builds index."""
        new_chunks_data = []

        for path in pdf_paths:
            # Clean filename logic (Removes Streamlit's temp prefix)
            raw_filename = os.path.basename(path)
            if "_" in raw_filename:
                filename = raw_filename.split("_", 1)[1]
            else:
                filename = raw_filename

            print(f"Processing: {filename}")

            try:
                reader = PdfReader(path)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    page_num = i + 1

                    # Sliding Window Chunking (Prevents data loss)
                    text_segments = self._sliding_window_chunking(page_text)

                    for segment in text_segments:
                        new_chunks_data.append({
                            "text": segment,
                            "page": page_num,
                            "source": filename
                        })
            except Exception as e:
                print(f"Error reading {filename}: {e}")

        if not new_chunks_data:
            return "No relevant clinical text found in documents."

        # Embed and Build Index
        print(f"Embedding {len(new_chunks_data)} chunks with metadata...")
        text_only = [item['text'] for item in new_chunks_data]

        embeddings = self.encoder.encode(
            text_only,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype("float32"))
        self.chunks = new_chunks_data
        print("✅ Knowledge Base Built Successfully.")

    # --- Core Logic: Retrieval ---
    def search(self, query: str) -> str:
        """Retrieves chunks and formats them with source citation metadata."""
        if not self.index:
            return ""

        if self._is_unsafe_query(query):
            return "GUARDRAIL: This query asks for subjective comparison. Please ask for specific guidelines."

        # Vector Search
        query_vec = self.encoder.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vec, k=Config.RETRIEVAL_K)

        # Format Context
        context_parts = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                item = self.chunks[idx]
                # Injection of Metadata for LLM Citation
                formatted_chunk = f"[Source: '{item['source']}', Page: {item['page']}]\n{item['text']}"
                context_parts.append(formatted_chunk)

        return "\n\n".join(context_parts)

    # --- Persistence: Save/Load ---
    def save_index(self, folder_path="storage"):
        """Persists the FAISS index and metadata to disk."""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.index:
            faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))

        with open(os.path.join(folder_path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"✅ Index saved to {folder_path}")

    def load_index(self, folder_path="storage"):
        """Loads the FAISS index and metadata from disk."""
        if not os.path.exists(os.path.join(folder_path, "index.faiss")):
            return False

        self.index = faiss.read_index(os.path.join(folder_path, "index.faiss"))

        with open(os.path.join(folder_path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)

        print(f"✅ Index loaded from {folder_path}")
        return True

    # --- Helper Methods ---
    def _sliding_window_chunking(self, text: str, window_size=1000, overlap=200) -> list[str]:
        """Splits text into overlapping windows to ensure boundary context is kept."""
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + window_size
            chunk = text[start:end]
            if len(chunk) > 50:
                chunks.append(chunk)
            start += (window_size - overlap)
        return chunks

    def _is_unsafe_query(self, query: str) -> bool:
        q = query.lower()
        return any(re.search(p, q) for p in Config.UNSAFE_PATTERNS)