from sentence_transformers import SentenceTransformer

# Small, fast, very common baseline model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None

def load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings locally using HuggingFace.
    """
    model = load_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.tolist()
