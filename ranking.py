import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def rank_similar_datasets(
    embeddings: np.ndarray,
    names: list,
    query_name: str,
    metric: str = "cosine"
):
    idx = names.index(query_name)
    query_emb = embeddings[idx].reshape(1, -1)

    if metric == "cosine":
        scores = cosine_similarity(query_emb, embeddings)[0]
    else:
        raise ValueError("Métrica não suportada")

    ranking = sorted(
        zip(names, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranking
