"""
ranking.py — Ranking de similaridade entre datasets

Correções aplicadas:
  - Suporte a métricas 'cosine' E 'euclidean'
    (euclidean é mais coerente com a contrastive loss usada no treino)
  - Distâncias euclidianas são convertidas em similaridade via exp(-d)
    para manter escala comparável com cosine [0, 1]
  - Função retorna também o array de scores para uso externo
  - Parâmetros lidos de CFG.clustering por padrão
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import List, Tuple, Optional

from config import CFG


# ============================================================================
# RANKING DE UM DATASET CONTRA TODOS
# ============================================================================

def rank_similar_datasets(
    embeddings:  np.ndarray,
    names:       List[str],
    query_name:  str,
    metric:      Optional[str] = None,
    top_k:       Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    Ordena todos os datasets por similaridade em relação a um dataset query.

    Args:
        embeddings:  Array (n_datasets, embedding_dim)
        names:       Lista de nomes (mesma ordem que embeddings)
        query_name:  Nome do dataset de referência
        metric:      'cosine' ou 'euclidean'
                     Se None, usa CFG.clustering.ranking_metric
        top_k:       Retorna apenas os top_k resultados
                     Se None, usa CFG.clustering.ranking_top_k

    Returns:
        Lista de (nome, score) ordenada do mais para o menos similar.
        Score está sempre em (0, 1]:
          - cosine   : similaridade cosseno direta
          - euclidean: exp(−distância)
    """
    metric = metric or CFG.clustering.ranking_metric
    top_k  = top_k  or CFG.clustering.ranking_top_k

    if query_name not in names:
        raise ValueError(
            f"Dataset '{query_name}' não encontrado. "
            f"Disponíveis: {names}"
        )

    idx       = names.index(query_name)
    query_emb = embeddings[idx].reshape(1, -1)

    scores = _compute_scores(query_emb, embeddings, metric)

    ranking = sorted(
        zip(names, scores.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    return ranking[:top_k]


# ============================================================================
# MATRIZ DE RANKING COMPLETA
# ============================================================================

def build_similarity_matrix(
    embeddings: np.ndarray,
    names:      List[str],
    metric:     Optional[str] = None,
) -> np.ndarray:
    """
    Constrói a matriz de similaridade (n × n) entre todos os datasets.

    Args:
        embeddings: Array (n_datasets, embedding_dim)
        names:      Lista de nomes (usada apenas para print)
        metric:     'cosine' ou 'euclidean'

    Returns:
        Array (n, n) com scores em (0, 1]
    """
    metric = metric or CFG.clustering.ranking_metric

    if metric == 'cosine':
        sim_matrix = cosine_similarity(embeddings)
        # Garante [0, 1] (cosine pode dar -1 a 1)
        sim_matrix = (sim_matrix + 1.0) / 2.0
    elif metric == 'euclidean':
        dist_matrix = euclidean_distances(embeddings)
        sim_matrix  = np.exp(-dist_matrix)
    else:
        raise ValueError(f"Métrica '{metric}' não suportada. Use 'cosine' ou 'euclidean'.")

    print(f"  Matriz de similaridade ({metric}): shape {sim_matrix.shape}")
    return sim_matrix


# ============================================================================
# RANKING CRUZADO (todos vs todos)
# ============================================================================

def rank_all_datasets(
    embeddings: np.ndarray,
    names:      List[str],
    metric:     Optional[str] = None,
    top_k:      Optional[int] = None,
) -> dict:
    """
    Gera o ranking de similaridade para cada dataset.

    Returns:
        Dicionário {nome: [(nome_vizinho, score), ...]}
    """
    metric = metric or CFG.clustering.ranking_metric
    top_k  = top_k  or CFG.clustering.ranking_top_k

    rankings = {}
    for name in names:
        rankings[name] = rank_similar_datasets(embeddings, names, name, metric, top_k)
    return rankings


# ============================================================================
# HELPER INTERNO
# ============================================================================

def _compute_scores(
    query_emb: np.ndarray,
    all_emb:   np.ndarray,
    metric:    str
) -> np.ndarray:
    """
    Calcula scores de similaridade entre query e todos os embeddings.
    Sempre retorna valores em (0, 1].
    """
    if metric == 'cosine':
        scores = cosine_similarity(query_emb, all_emb)[0]
        return (scores + 1.0) / 2.0    # remapeia [-1, 1] → [0, 1]

    elif metric == 'euclidean':
        dists  = euclidean_distances(query_emb, all_emb)[0]
        return np.exp(-dists)           # exp(-d) ∈ (0, 1]

    else:
        raise ValueError(
            f"Métrica '{metric}' não suportada. Use 'cosine' ou 'euclidean'."
        )


# ============================================================================
# TESTES
# ============================================================================

if __name__ == "__main__":
    from config import CFG

    print("\n" + "=" * 70)
    print("TESTE DO RANKING")
    print("=" * 70)

    np.random.seed(42)
    n   = 8
    dim = CFG.siamese.embedding_dim

    # Cria 3 "clusters" sintéticos
    embs  = np.vstack([
        np.random.randn(3, dim) * 0.05 + 1.0,   # cluster A
        np.random.randn(3, dim) * 0.05 + 3.0,   # cluster B
        np.random.randn(2, dim) * 0.05 + 6.0,   # cluster C
    ]).astype(np.float32)
    names = [f"DS_{i}" for i in range(n)]

    for metric in ('cosine', 'euclidean'):
        print(f"\n  Métrica: {metric}")
        ranking = rank_similar_datasets(embs, names, "DS_0", metric=metric, top_k=5)
        for i, (name, score) in enumerate(ranking, 1):
            tag = " ← query" if name == "DS_0" else ""
            print(f"    {i}. {name}: {score:.4f}{tag}")

    # Matriz completa
    print("\n  Matriz de similaridade (cosine):")
    sim = build_similarity_matrix(embs, names, metric='cosine')
    print(f"    shape: {sim.shape}")
    print(f"    diagonal (esperado ≈ 1.0): {np.diag(sim).round(4)}")

    print("\n✓ TODOS OS TESTES DE RANKING CONCLUÍDOS\n")