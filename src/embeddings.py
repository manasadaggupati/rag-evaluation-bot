from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(texts: list[str]):
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )