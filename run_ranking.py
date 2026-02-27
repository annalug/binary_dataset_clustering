"""
run_ranking.py — Pipeline principal

Correções aplicadas:
  - Verifica se existe modelo treinado ANTES de gerar embeddings;
    se não existir, executa o pipeline de treinamento completo
  - dataset_type_labels derivados de CFG.pairs.dataset_type_map
    (semântica de tipo de dataset, não malware/benigno)
  - Todos os caminhos de saída via CFG.paths
  - Parâmetros de clustering, ranking e plot via CFG
"""

import os
import numpy as np
import pandas as pd

from config import CFG, print_config
from dataset_loader import load_datasets_from_folder
from standardizer import DatasetStandardizer
from siamese import SiameseNet
from pair_generator import build_dataset_type_labels, create_balanced_pairs
from ranking import rank_similar_datasets, rank_all_datasets, build_similarity_matrix
from similarity_clustering import SimilarityBasedClustering, create_clusters_from_ranking
from plot_clusters import generate_cluster_plots


# ============================================================================
# UTILITÁRIOS
# ============================================================================

def _model_exists() -> bool:
    """Verifica se há modelo siamês treinado salvo."""
    encoder_path = f"{CFG.paths.model_file(CFG.siamese.model_prefix)}_encoder.keras"
    return os.path.exists(encoder_path)


def _save_csv(df: pd.DataFrame, filename: str):
    path = CFG.paths.output_file(filename)
    df.to_csv(path, index=False)
    print(f"  ✓ CSV salvo em: {path}")
    return path


def _save_txt(content: str, filename: str):
    path = CFG.paths.output_file(filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  ✓ TXT salvo em: {path}")
    return path


# ============================================================================
# ETAPA 1 — TREINAMENTO
# ============================================================================

def run_training(datasets_std: np.ndarray, names: list):
    """
    Treina o modelo siamês com pares baseados em TIPO DE DATASET.

    O CFG.pairs.dataset_type_map deve ser preenchido pelo usuário
    para indicar quais datasets são estruturalmente similares.
    """
    print("\n" + "=" * 70)
    print("ETAPA 1 — TREINAMENTO DO MODELO SIAMÊS")
    print("=" * 70)

    # Labels de tipo de dataset (NÃO labels malware/benigno)
    ds_labels = build_dataset_type_labels(names, CFG.pairs.dataset_type_map)
    print(f"\n  Labels de tipo de dataset:")
    for name, lbl in zip(names, ds_labels):
        print(f"    {name:40s} → tipo {lbl}")

    # Gera pares balanceados por tipo
    pairs_left, pairs_right, sim_labels = create_balanced_pairs(
        datasets_std,
        ds_labels,
        pairs_per_type=CFG.pairs.pairs_per_class,
        random_state=CFG.pairs.random_state,
    )

    # Treina
    model = SiameseNet()
    model.train(pairs_left, pairs_right, sim_labels)
    model.save()

    return model


# ============================================================================
# ETAPA 2 — EMBEDDINGS
# ============================================================================

def run_embeddings(model: SiameseNet, datasets_std: np.ndarray) -> np.ndarray:
    """Extrai embeddings de todos os datasets."""
    print("\n" + "=" * 70)
    print("ETAPA 2 — EXTRAÇÃO DE EMBEDDINGS")
    print("=" * 70)

    embeddings = np.array([
        model.get_embedding(d) for d in datasets_std
    ])
    print(f"\n  ✓ Embeddings: {embeddings.shape}")
    return embeddings


# ============================================================================
# ETAPA 3 — RANKING
# ============================================================================

def run_ranking(embeddings: np.ndarray, names: list):
    """Gera e exibe ranking de similaridade para todos os datasets."""
    print("\n" + "=" * 70)
    print("ETAPA 3 — RANKING DE SIMILARIDADE")
    print("=" * 70)

    metric   = CFG.clustering.ranking_metric
    top_k    = CFG.clustering.ranking_top_k
    all_rank = rank_all_datasets(embeddings, names, metric=metric, top_k=top_k)

    # Exibe ranking do primeiro dataset como exemplo
    query = names[0]
    print(f"\n  Ranking para '{query}' (métrica: {metric}):\n")
    for i, (name, score) in enumerate(all_rank[query], 1):
        tag = " ← query" if name == query else ""
        print(f"    {i:2d}. {name:40s} {score:.4f}{tag}")

    # Salva rankings em CSV
    rows = []
    for src, rank_list in all_rank.items():
        for rank_pos, (tgt, score) in enumerate(rank_list, 1):
            rows.append({
                'query':    src,
                'rank':     rank_pos,
                'target':   tgt,
                'score':    round(score, 6),
                'metric':   metric,
            })
    _save_csv(pd.DataFrame(rows), "ranking_results.csv")

    # Salva matriz de similaridade
    sim_matrix = build_similarity_matrix(embeddings, names, metric=metric)
    sim_df = pd.DataFrame(sim_matrix, index=names, columns=names)
    _save_csv(sim_df, "similarity_matrix.csv")

    return all_rank, sim_matrix


# ============================================================================
# ETAPA 4 — CLUSTERING
# ============================================================================

def run_clustering(embeddings: np.ndarray, names: list):
    """Aplica clustering hierárquico e por threshold."""
    print("\n" + "=" * 70)
    print("ETAPA 4 — CLUSTERING")
    print("=" * 70)

    cc = CFG.clustering
    n_clusters = min(cc.n_clusters, max(2, len(names) // 2))

    # --- Hierárquico ---
    print("\n  Método: Clustering Hierárquico")
    clusterer = SimilarityBasedClustering(
        method=cc.method,
        n_clusters=n_clusters,
        linkage=cc.linkage,
    )
    cluster_labels = clusterer.fit_predict(embeddings, names)

    df_hier = pd.DataFrame({'dataset': names, 'cluster': cluster_labels})
    _save_csv(df_hier, "clustering_hierarchical.csv")

    # --- Threshold ---
    print(f"\n  Método: Threshold (≥ {cc.similarity_threshold})")
    threshold_clusters = create_clusters_from_ranking(
        embeddings, names,
        threshold=cc.similarity_threshold,
        min_cluster_size=cc.min_cluster_size,
    )

    print(f"\n  Clusters por threshold:")
    rows_thr = []
    for cid, cnames in threshold_clusters.items():
        label = "Outliers" if cid == -1 else f"Cluster {cid}"
        print(f"    {label} ({len(cnames)}): {', '.join(cnames)}")
        for cname in cnames:
            rows_thr.append({'dataset': cname, 'cluster': cid})
    _save_csv(pd.DataFrame(rows_thr), "clustering_threshold.csv")

    return cluster_labels, threshold_clusters


# ============================================================================
# ETAPA 5 — RELATÓRIO TEXTUAL
# ============================================================================

def run_report(
    names:          list,
    embeddings:     np.ndarray,
    cluster_labels: np.ndarray,
    sim_matrix:     np.ndarray,
):
    """Gera relatório textual consolidado."""
    print("\n" + "=" * 70)
    print("ETAPA 5 — RELATÓRIO")
    print("=" * 70)

    from sklearn.metrics import silhouette_score, davies_bouldin_score

    valid      = cluster_labels != -1
    silhouette = davies_bouldin = float('nan')
    n_valid    = int(np.sum(valid))

    if n_valid > 1 and len(np.unique(cluster_labels[valid])) > 1:
        try:
            silhouette    = silhouette_score(embeddings[valid], cluster_labels[valid])
            davies_bouldin = davies_bouldin_score(embeddings[valid], cluster_labels[valid])
        except Exception:
            pass

    unique_labels = np.unique(cluster_labels)
    n_clusters    = len(unique_labels) - (1 if -1 in unique_labels else 0)
    lines = []
    sep   = "=" * 70

    lines.append(sep)
    lines.append("RELATÓRIO DE CLUSTERING DE DATASETS")
    lines.append(sep)
    lines.append("")
    lines.append("RESUMO:")
    lines.append(f"  Total de datasets    : {len(names)}")
    lines.append(f"  Número de clusters   : {n_clusters}")
    lines.append(f"  Outliers             : {int(np.sum(cluster_labels == -1))}")
    lines.append(f"  Embedding dim        : {embeddings.shape[1]}")
    lines.append(f"  Métrica de ranking   : {CFG.clustering.ranking_metric}")
    lines.append("")
    lines.append("MÉTRICAS DE QUALIDADE:")
    lines.append(f"  Silhouette Score     : {silhouette:.4f}")
    lines.append(f"  Davies-Bouldin Index : {davies_bouldin:.4f}")
    lines.append("")

    lines.append("DISTRIBUIÇÃO POR CLUSTER:")
    for lbl in sorted(unique_labels):
        count   = int(np.sum(cluster_labels == lbl))
        pct     = count / len(names) * 100
        header  = "OUTLIERS" if lbl == -1 else f"CLUSTER {lbl}"
        members = [names[i] for i, l in enumerate(cluster_labels) if l == lbl]
        lines.append(f"\n  [{header}] {count} datasets ({pct:.1f}%)")
        for m in members:
            lines.append(f"    • {m}")

    if n_clusters > 1:
        lines.append("\n" + sep)
        lines.append("SIMILARIDADE INTER-CLUSTER")
        lines.append(sep)
        for i, li in enumerate(unique_labels):
            if li == -1:
                continue
            for j, lj in enumerate(unique_labels):
                if i >= j or lj == -1:
                    continue
                inter = sim_matrix[cluster_labels == li][:, cluster_labels == lj]
                lines.append(f"\n  Cluster {li} ↔ Cluster {lj}:")
                lines.append(f"    Média  : {np.mean(inter):.4f}")
                lines.append(f"    Mínima : {np.min(inter):.4f}")
                lines.append(f"    Máxima : {np.max(inter):.4f}")

    _save_txt("\n".join(lines), "clustering_report.txt")


# ============================================================================
# ETAPA 6 — GRÁFICOS
# ============================================================================

def run_plots(embeddings: np.ndarray, names: list, cluster_labels: np.ndarray):
    """Gera todos os gráficos via plot_clusters."""
    print("\n" + "=" * 70)
    print("ETAPA 6 — GRÁFICOS")
    print("=" * 70)

    results = generate_cluster_plots(
        embeddings,
        names,
        cluster_labels,
        prefix=CFG.plot.plot_prefix,
    )
    return results


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    print_config()
    CFG.paths.ensure_all()

    # 1. Carrega CSVs
    print("\n📂 Carregando datasets...")
    datasets, names, _ = load_datasets_from_folder(
        CFG.paths.data, class_column=CFG.class_column
    )
    # NOTA: `_` são os labels malware/benigno dentro de cada dataset —
    # não usados aqui. Os labels de TIPO DE DATASET vêm de CFG.pairs.dataset_type_map.
    print(f"  ✓ {len(datasets)} datasets carregados")

    # 2. Padroniza
    print("\n🔧 Padronizando...")
    standardizer  = DatasetStandardizer()
    datasets_std  = standardizer.fit_transform_batch(datasets)

    # 3. Modelo siamês — treina se necessário
    model = SiameseNet()
    if _model_exists():
        print("\n🧠 Modelo encontrado — carregando pesos...")
        model.load()
    else:
        print("\n🧠 Modelo não encontrado — iniciando treinamento...")
        if not CFG.pairs.dataset_type_map:
            print(
                "\n  ⚠ CFG.pairs.dataset_type_map está vazio!\n"
                "    Preencha-o em config.py antes de treinar.\n"
                "    Exemplo:\n"
                "      CFG.pairs.dataset_type_map = {\n"
                "          'balanced_adroit':   0,\n"
                "          'balanced_drebin215': 1,\n"
                "          ...\n"
                "      }\n"
                "    Usando embeddings não treinados para demonstração."
            )
        else:
            model = run_training(datasets_std, names)

    # 4. Embeddings
    embeddings = run_embeddings(model, datasets_std)

    # 5. Ranking
    all_rank, sim_matrix = run_ranking(embeddings, names)

    # 6. Clustering
    cluster_labels, _ = run_clustering(embeddings, names)

    # 7. Relatório
    run_report(names, embeddings, cluster_labels, sim_matrix)

    # 8. Gráficos
    run_plots(embeddings, names, cluster_labels)

    # Resumo final
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETO")
    print("=" * 70)
    n_clusters = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"  Datasets processados : {len(names)}")
    print(f"  Clusters encontrados : {n_clusters}")
    print(f"  Outliers             : {int(np.sum(cluster_labels == -1))}")
    print(f"  Embedding dim        : {embeddings.shape[1]}")

    print(f"\n  Saídas:")
    print(f"    Relatórios → {CFG.paths.output}/")
    print(f"    Gráficos   → {CFG.paths.img}/")
    print(f"    Modelos    → {CFG.paths.models}/")

    # Verifica arquivos gerados
    print("\n  Arquivos:")
    expected = [
        CFG.paths.output_file("clustering_hierarchical.csv"),
        CFG.paths.output_file("clustering_threshold.csv"),
        CFG.paths.output_file("ranking_results.csv"),
        CFG.paths.output_file("similarity_matrix.csv"),
        CFG.paths.output_file("clustering_report.txt"),
        CFG.paths.img_file(f"{CFG.plot.plot_prefix}_visualization.jpg"),
        CFG.paths.img_file(f"{CFG.plot.plot_prefix}_dendrogram.jpg"),
    ]
    for f in expected:
        exists = os.path.exists(f)
        size   = f"{os.path.getsize(f) / 1024:.1f} KB" if exists else "—"
        status = "✓" if exists else "⚠"
        print(f"    {status} {f}  ({size})")

    return embeddings, names, cluster_labels


if __name__ == "__main__":
    main()