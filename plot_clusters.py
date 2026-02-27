"""
Geração de gráficos de clustering em JPG
Visualização dos resultados do clustering

Outputs:
  - Gráficos (JPG)       → img/
  - Relatórios (CSV/TXT) → output/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# DIRETÓRIOS DE SAÍDA
# ============================================================================

IMG_DIR = "img"
OUTPUT_DIR = "output"


def _ensure_dirs():
    """Cria os diretórios de saída se não existirem."""
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _img_path(filename: str) -> str:
    return os.path.join(IMG_DIR, filename)


def _output_path(filename: str) -> str:
    return os.path.join(OUTPUT_DIR, filename)


# ============================================================================
# REDUÇÃO DIMENSIONAL (helper compartilhado)
# ============================================================================

def _reduce_to_2d(embeddings: np.ndarray, method: str = "tsne") -> np.ndarray:
    """Reduz embeddings para 2D usando t-SNE ou PCA como fallback."""
    if embeddings.shape[1] <= 2:
        return embeddings

    if method == "tsne":
        try:
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(embeddings) - 1),
                max_iter=1000,
                init='pca'
            )
            return tsne.fit_transform(embeddings)
        except Exception as e:
            print(f"  ⚠ t-SNE falhou ({e}), usando PCA como fallback...")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(embeddings)


# ============================================================================
# GRÁFICO PRINCIPAL: VISUALIZAÇÃO DE CLUSTERS
# ============================================================================

def plot_cluster_visualization(
    embeddings: np.ndarray,
    names: List[str],
    labels: np.ndarray,
    output_filename: str = "cluster_visualization.jpg",
    title: str = "Clustering de Datasets"
) -> str:
    """
    Gera visualização completa dos clusters (t-SNE + histograma + boxplot + heatmap).
    Salva em img/.
    """
    _ensure_dirs()
    out = _img_path(output_filename)

    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    fig = plt.figure(figsize=(20, 12))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    embeddings_2d = _reduce_to_2d(embeddings)

    # 1. t-SNE
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        is_out = label == -1
        ax1.scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1],
            c=['gray'] if is_out else [colors[i % len(colors)]],
            marker='x' if is_out else 'o',
            s=80 if is_out else 100,
            alpha=0.6 if is_out else 0.75,
            label='Outliers' if is_out else f'Cluster {label}',
            edgecolors='gray' if is_out else 'white', linewidth=1
        )
        if not is_out and np.sum(mask) <= 15:
            for idx in np.where(mask)[0]:
                ax1.annotate(names[idx], (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                             fontsize=8, alpha=0.85, ha='center', va='bottom')

    ax1.set_title(f'{title}\nVisualização t-SNE 2D', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Dimensão t-SNE 1', fontsize=12)
    ax1.set_ylabel('Dimensão t-SNE 2', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 2. Histograma por cluster
    ax2 = plt.subplot2grid((3, 3), (0, 2))
    counts = [np.sum(labels == l) for l in sorted(unique_labels)]
    bar_names = ['Outliers' if l == -1 else f'C{l}' for l in sorted(unique_labels)]
    bars = ax2.bar(bar_names, counts, color=colors[:len(counts)], alpha=0.75)
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 str(count), ha='center', va='bottom', fontsize=10)
    ax2.set_title('Distribuição por Cluster', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Nº de Datasets')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Boxplot de distâncias intra-cluster
    ax3 = plt.subplot2grid((3, 3), (1, 2))
    intra_distances, box_labels = [], []
    for label in sorted(unique_labels):
        if label == -1:
            continue
        mask = labels == label
        if np.sum(mask) > 1:
            pts = embeddings[mask]
            centroid = np.mean(pts, axis=0)
            intra_distances.append(np.linalg.norm(pts - centroid, axis=1))
            box_labels.append(f'C{label}')

    if intra_distances:
        bp = ax3.boxplot(intra_distances, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(intra_distances)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax3.set_title('Distâncias Intra-cluster', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Distância do Centroide')
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'Dados insuficientes\npara boxplot',
                 ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Distâncias Intra-cluster', fontsize=14, fontweight='bold')

    # 4. Heatmap de similaridade
    ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    sorted_idx = np.argsort(labels)
    sorted_emb = embeddings[sorted_idx]
    sorted_names = [names[i] for i in sorted_idx]
    sorted_labels = labels[sorted_idx]
    sim_matrix = cosine_similarity(sorted_emb)
    im = ax4.imshow(sim_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)

    cur = 0
    for lbl in np.unique(sorted_labels):
        sz = int(np.sum(sorted_labels == lbl))
        ax4.axhline(cur - 0.5, color='white', linewidth=2, alpha=0.8)
        ax4.axvline(cur - 0.5, color='white', linewidth=2, alpha=0.8)
        cur += sz
    ax4.axhline(cur - 0.5, color='white', linewidth=2, alpha=0.8)
    ax4.axvline(cur - 0.5, color='white', linewidth=2, alpha=0.8)

    interval = max(1, len(sorted_names) // 20)
    ticks = np.arange(0, len(sorted_names), interval)
    fsize = 8 if len(sorted_names) <= 20 else 6
    ax4.set_xticks(ticks)
    ax4.set_yticks(ticks)
    ax4.set_xticklabels([sorted_names[i] for i in ticks], rotation=45, ha='right', fontsize=fsize)
    ax4.set_yticklabels([sorted_names[i] for i in ticks], fontsize=fsize)
    ax4.set_title('Matriz de Similaridade (ordenada por cluster)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax4, shrink=0.8, label='Similaridade Cosseno')

    # 5. Estatísticas no rodapé
    from sklearn.metrics import silhouette_score
    valid = labels != -1
    silhouette = np.nan
    if np.sum(valid) > 1 and len(np.unique(labels[valid])) > 1:
        try:
            silhouette = silhouette_score(embeddings[valid], labels[valid])
        except Exception:
            pass

    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    fig.text(0.01, 0.01,
             f"Estatísticas:\n"
             f"• Total datasets: {len(embeddings)}\n"
             f"• Clusters: {n_clusters}\n"
             f"• Outliers: {int(np.sum(labels == -1))}\n"
             f"• Silhouette Score: {silhouette:.3f}\n"
             f"• Embedding dim: {embeddings.shape[1]}",
             fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Visualização salva em: {out}")
    return out


# ============================================================================
# GRÁFICO 2: DENDROGRAMA
# ============================================================================

def plot_dendrogram(
    embeddings: np.ndarray,
    names: List[str],
    output_filename: str = "dendrogram.jpg",
    method: str = 'ward'
) -> Optional[str]:
    """
    Gera dendrograma hierárquico. Salva em img/.
    """
    _ensure_dirs()
    from scipy.cluster.hierarchy import dendrogram, linkage

    try:
        Z = linkage(embeddings, method=method, metric='euclidean')
    except Exception as e:
        print(f"  ⚠ Erro ao calcular linkage: {e}")
        return None

    out = _img_path(output_filename)
    plt.figure(figsize=(15, 8))
    dendrogram(
        Z, labels=names, leaf_rotation=90, leaf_font_size=10,
        show_contracted=True, truncate_mode='lastp', p=min(30, len(names))
    )
    plt.title(f'Dendrograma Hierárquico (método: {method})', fontsize=16, fontweight='bold')
    plt.xlabel('Datasets', fontsize=12)
    plt.ylabel('Distância', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Dendrograma salvo em: {out}")
    return out


# ============================================================================
# RELATÓRIOS TEXTUAIS
# ============================================================================

def export_cluster_report(
    embeddings: np.ndarray,
    names: List[str],
    labels: np.ndarray,
    csv_filename: str = "cluster_report.csv",
    txt_filename: str = "cluster_analysis.txt"
) -> Tuple[str, str]:
    """
    Exporta relatório CSV e análise textual detalhada. Salva em output/.
    """
    _ensure_dirs()
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    csv_path = _output_path(csv_filename)
    txt_path = _output_path(txt_filename)

    # CSV
    df = pd.DataFrame({'dataset': names, 'cluster': labels, 'is_outlier': labels == -1})
    if len(embeddings) > 2:
        try:
            coords = _reduce_to_2d(embeddings)
            df['tsne_x'] = coords[:, 0]
            df['tsne_y'] = coords[:, 1]
        except Exception:
            pass
    df.to_csv(csv_path, index=False)

    # Métricas
    valid = labels != -1
    silhouette = davies_bouldin = np.nan
    if np.sum(valid) > 1 and len(np.unique(labels[valid])) > 1:
        try:
            silhouette = silhouette_score(embeddings[valid], labels[valid])
            davies_bouldin = davies_bouldin_score(embeddings[valid], labels[valid])
        except Exception:
            pass

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # TXT
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RELATÓRIO DE CLUSTERING DE DATASETS\n")
        f.write("=" * 70 + "\n\n")

        f.write("RESUMO GERAL:\n")
        f.write(f"  • Total de datasets  : {len(names)}\n")
        f.write(f"  • Número de clusters : {n_clusters}\n")
        f.write(f"  • Outliers           : {int(np.sum(labels == -1))}\n")
        f.write(f"  • Embedding dim      : {embeddings.shape[1]}\n\n")

        f.write("MÉTRICAS DE QUALIDADE:\n")
        f.write(f"  • Silhouette Score    : {silhouette:.4f}\n")
        f.write(f"  • Davies-Bouldin Index: {davies_bouldin:.4f}\n\n")
        f.write("  Interpretação:\n")
        f.write("    Silhouette  → mais próximo de 1 = melhor separação\n")
        f.write("    Davies-Bouldin → menor valor = clusters mais compactos\n\n")

        f.write("DISTRIBUIÇÃO DETALHADA:\n")
        for label in sorted(unique_labels):
            count = int(np.sum(labels == label))
            pct = count / len(names) * 100
            header = "OUTLIERS" if label == -1 else f"CLUSTER {label}"
            f.write(f"\n  [{header}] — {count} datasets ({pct:.1f}%)\n")
            members = [names[i] for i, l in enumerate(labels) if l == label]
            for m in members[:10]:
                f.write(f"    • {m}\n")
            if len(members) > 10:
                f.write(f"    • ... e mais {len(members) - 10} datasets\n")

        if n_clusters > 1:
            f.write("\n" + "=" * 70 + "\n")
            f.write("ANÁLISE DE SIMILARIDADE INTER-CLUSTER\n")
            f.write("=" * 70 + "\n\n")
            sim_mat = cosine_similarity(embeddings)
            for i, li in enumerate(unique_labels):
                if li == -1:
                    continue
                for j, lj in enumerate(unique_labels):
                    if i >= j or lj == -1:
                        continue
                    inter = sim_mat[labels == li][:, labels == lj]
                    f.write(f"  Cluster {li} ↔ Cluster {lj}:\n")
                    f.write(f"    • Média   : {np.mean(inter):.4f}\n")
                    f.write(f"    • Mínima  : {np.min(inter):.4f}\n")
                    f.write(f"    • Máxima  : {np.max(inter):.4f}\n\n")

    print(f"  ✓ Relatório CSV salvo em: {csv_path}")
    print(f"  ✓ Análise TXT  salva em: {txt_path}")
    return csv_path, txt_path


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def generate_cluster_plots(
    embeddings: np.ndarray,
    names: List[str],
    labels: np.ndarray,
    prefix: str = "cluster_results"
) -> Dict[str, Optional[str]]:
    """
    Gera todos os gráficos e relatórios de clustering.

    Saídas:
      img/    → visualização JPG + dendrograma JPG
      output/ → relatório CSV + análise TXT

    Returns:
        Dicionário com os caminhos de cada arquivo gerado
    """
    _ensure_dirs()
    print("\n" + "=" * 70)
    print("GERANDO GRÁFICOS E RELATÓRIOS")
    print(f"  Imagens    → {IMG_DIR}/")
    print(f"  Relatórios → {OUTPUT_DIR}/")
    print("=" * 70)

    results: Dict[str, Optional[str]] = {
        'visualization': None,
        'dendrogram': None,
        'csv_report': None,
        'text_report': None,
    }

    try:
        results['visualization'] = plot_cluster_visualization(
            embeddings, names, labels,
            output_filename=f"{prefix}_visualization.jpg"
        )
    except Exception as e:
        print(f"  ⚠ Visualização falhou: {e}")

    try:
        results['dendrogram'] = plot_dendrogram(
            embeddings, names,
            output_filename=f"{prefix}_dendrogram.jpg"
        )
    except Exception as e:
        print(f"  ⚠ Dendrograma falhou: {e}")

    try:
        csv_path, txt_path = export_cluster_report(
            embeddings, names, labels,
            csv_filename=f"{prefix}_report.csv",
            txt_filename=f"{prefix}_analysis.txt"
        )
        results['csv_report'] = csv_path
        results['text_report'] = txt_path
    except Exception as e:
        print(f"  ⚠ Relatório falhou: {e}")

    print("\n" + "=" * 70)
    print("✅ GERAÇÃO CONCLUÍDA")
    print("=" * 70)
    print("\nArquivos gerados:")
    for key, path in results.items():
        status = f"✓  {path}" if path else "✗  não gerado"
        print(f"  {key:20s}: {status}")

    return results


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    dim = 128
    centers = [np.ones(dim) * v for v in [0.0, 1.5, 3.0]]
    embeddings, names, true_labels = [], [], []
    for i, c in enumerate(centers):
        for j in range(4):
            embeddings.append(c + np.random.randn(dim) * 0.2)
            names.append(f"Dataset_C{i}_{j}")
            true_labels.append(i)

    generate_cluster_plots(np.array(embeddings), names, np.array(true_labels), prefix="test")