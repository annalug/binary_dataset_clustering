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
import matplotlib.patches as mpatches
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

IMG_DIR    = "img"
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
# PALETA DARK  (usada em todos os plots)
# ============================================================================

_BG     = "#080b14"   # fundo da figura
_BG2    = "#0d1120"   # fundo de eixos / painéis
_BORDER = "#1e2d4a"   # bordas / spines
_TEXT   = "#e2e8f0"   # títulos e labels principais
_MUTED  = "#64748b"   # labels secundários / gridlines

# Cores por cluster — ponto, borda e label
_C_POINT = ["#4fc3f7", "#ef5350", "#66bb6a", "#ffa726", "#ab47bc",
            "#26c6da", "#ff7043", "#9ccc65", "#ec407a", "#42a5f5"]
_C_EDGE  = ["#0288d1", "#c62828", "#2e7d32", "#e65100", "#6a1b9a",
            "#00838f", "#bf360c", "#558b2f", "#880e4f", "#1565c0"]
_C_LABEL = ["#e1f5fe", "#ffcdd2", "#c8e6c9", "#ffe0b2", "#f3e5f5",
            "#e0f7fa", "#fbe9e7", "#f1f8e9", "#fce4ec", "#e3f2fd"]

_OUTLIER_PT = "#78909c"
_OUTLIER_ED = "#455a64"


def _pt(lbl):  return _OUTLIER_PT if lbl == -1 else _C_POINT[lbl % len(_C_POINT)]
def _ed(lbl):  return _OUTLIER_ED if lbl == -1 else _C_EDGE [lbl % len(_C_EDGE)]
def _lb(lbl):  return _MUTED      if lbl == -1 else _C_LABEL[lbl % len(_C_LABEL)]


def _style_ax(ax, title="", xlabel="", ylabel="", grid_axis="both"):
    """Aplica tema escuro consistente a um eixo."""
    ax.set_facecolor(_BG2)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.set_xlabel(xlabel, color=_MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=_MUTED, fontsize=10)
    if title:
        ax.set_title(title, color=_TEXT, fontsize=12, fontweight="bold", pad=10)
    if grid_axis:
        ax.grid(True, alpha=0.07, color="white", linestyle="--", axis=grid_axis)


# ============================================================================
# REDUÇÃO DIMENSIONAL  (helper compartilhado)
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
                init="pca",
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
    embeddings:      np.ndarray,
    names:           List[str],
    labels:          np.ndarray,
    output_filename: str = "cluster_visualization.jpg",
    title:           str = "Clustering de Datasets",
) -> str:
    """
    Gera visualização completa dos clusters (t-SNE + histograma + boxplot + heatmap).
    Salva em img/.
    """
    _ensure_dirs()
    out = _img_path(output_filename)

    unique_labels = np.unique(labels)
    n_clusters    = len(unique_labels) - (1 if -1 in unique_labels else 0)
    embeddings_2d = _reduce_to_2d(embeddings)

    fig = plt.figure(figsize=(20, 13), facecolor=_BG)
    gs  = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # ── 1. t-SNE ─────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.set_facecolor(_BG)
    for spine in ax1.spines.values():
        spine.set_edgecolor(_BORDER)
    ax1.tick_params(colors=_MUTED, labelsize=8)
    ax1.grid(True, alpha=0.06, color="white", linestyle="--")

    for lbl in unique_labels:
        mask   = labels == lbl
        is_out = lbl == -1
        pc, ec = _pt(lbl), _ed(lbl)
        lname  = "Outliers" if is_out else f"Cluster {lbl}"

        # Halo suave (glow)
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=ec, s=120, alpha=0.10, edgecolors="none", zorder=2)
        # Ponto principal
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=pc, s=55, alpha=0.88,
                    marker="x" if is_out else "o",
                    edgecolors=ec, linewidths=0.7,
                    label=lname, zorder=3)

        # Anotações nos datasets individuais (≤ 15 pontos por cluster)
        if not is_out and np.sum(mask) <= 15:
            for idx in np.where(mask)[0]:
                ax1.annotate(
                    names[idx],
                    (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                    xytext=(0, 9), textcoords="offset points",
                    fontsize=7.5, fontweight="bold", ha="center",
                    color=_lb(lbl),
                    bbox=dict(facecolor=_BG2, alpha=0.80,
                              edgecolor=ec, linewidth=0.7,
                              boxstyle="round,pad=0.25"),
                    zorder=5,
                )

    ax1.set_title(f"{title} — t-SNE 2D", color=_TEXT,
                  fontsize=14, fontweight="bold")
    ax1.set_xlabel("Dimensão t-SNE 1", color=_MUTED, fontsize=10)
    ax1.set_ylabel("Dimensão t-SNE 2", color=_MUTED, fontsize=10)
    legend = ax1.legend(
        title="Clusters", fontsize=9, title_fontsize=9,
        facecolor=_BG2, edgecolor=_BORDER, labelcolor=_TEXT,
        loc="upper right",
    )
    legend.get_title().set_color(_MUTED)

    # ── 2. Histograma por cluster ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    _style_ax(ax2, "Distribuição por Cluster", "Cluster", "Nº de Datasets",
              grid_axis="y")
    counts     = [int(np.sum(labels == l)) for l in sorted(unique_labels)]
    bar_names  = ["Out" if l == -1 else f"C{l}" for l in sorted(unique_labels)]
    bar_colors = [_pt(l) for l in sorted(unique_labels)]
    edge_cols  = [_ed(l) for l in sorted(unique_labels)]
    bars = ax2.bar(bar_names, counts, color=bar_colors, alpha=0.85,
                   edgecolor=edge_cols, linewidth=0.9)
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.06,
                 str(count), ha="center", va="bottom",
                 fontsize=10, color=_TEXT, fontweight="bold")

    # ── 3. Boxplot intra-cluster ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    _style_ax(ax3, "Distâncias Intra-cluster", "Cluster", "Dist. do Centroide",
              grid_axis="y")
    intra_distances, box_labels, box_colors, box_edges = [], [], [], []
    for lbl in sorted(unique_labels):
        if lbl == -1:
            continue
        mask = labels == lbl
        if np.sum(mask) > 1:
            pts      = embeddings[mask]
            centroid = np.mean(pts, axis=0)
            intra_distances.append(np.linalg.norm(pts - centroid, axis=1))
            box_labels.append(f"C{lbl}")
            box_colors.append(_pt(lbl))
            box_edges.append(_ed(lbl))

    if intra_distances:
        bp = ax3.boxplot(
            intra_distances, labels=box_labels, patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=_MUTED, linewidth=1),
            capprops=dict(color=_MUTED, linewidth=1),
            flierprops=dict(marker="o", color=_MUTED, markersize=4, alpha=0.5),
        )
        for patch, fc, ec in zip(bp["boxes"], box_colors, box_edges):
            patch.set_facecolor(fc)
            patch.set_alpha(0.70)
            patch.set_edgecolor(ec)
    else:
        ax3.text(0.5, 0.5, "Dados insuficientes\npara boxplot",
                 ha="center", va="center", transform=ax3.transAxes,
                 color=_MUTED, fontsize=10)

    # ── 4. Heatmap de similaridade ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0:3])
    ax4.set_facecolor(_BG2)
    for spine in ax4.spines.values():
        spine.set_edgecolor(_BORDER)

    sorted_idx    = np.argsort(labels)
    sorted_emb    = embeddings[sorted_idx]
    sorted_names  = [names[i] for i in sorted_idx]
    sorted_labels = labels[sorted_idx]
    sim_matrix    = cosine_similarity(sorted_emb)

    im = ax4.imshow(sim_matrix, cmap="magma", aspect="auto", vmin=0, vmax=1)

    # Linhas de divisão entre clusters — coloridas por cluster
    cur = 0
    for lbl in np.unique(sorted_labels):
        sz = int(np.sum(sorted_labels == lbl))
        ax4.axhline(cur - 0.5, color=_pt(lbl), linewidth=1.4, alpha=0.55)
        ax4.axvline(cur - 0.5, color=_pt(lbl), linewidth=1.4, alpha=0.55)
        cur += sz
    ax4.axhline(cur - 0.5, color=_BORDER, linewidth=1)
    ax4.axvline(cur - 0.5, color=_BORDER, linewidth=1)

    interval = max(1, len(sorted_names) // 20)
    ticks    = np.arange(0, len(sorted_names), interval)
    fsize    = 8 if len(sorted_names) <= 20 else 6
    ax4.set_xticks(ticks)
    ax4.set_yticks(ticks)
    ax4.set_xticklabels([sorted_names[i] for i in ticks],
                        rotation=45, ha="right", fontsize=fsize, color=_MUTED)
    ax4.set_yticklabels([sorted_names[i] for i in ticks],
                        fontsize=fsize, color=_MUTED)
    ax4.set_title("Matriz de Similaridade Cosseno (ordenada por cluster)",
                  color=_TEXT, fontsize=12, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label("Similaridade Cosseno", color=_MUTED, fontsize=9)
    cbar.ax.tick_params(colors=_MUTED)

    # ── 5. Rodapé com estatísticas ────────────────────────────────────────────
    from sklearn.metrics import silhouette_score
    valid      = labels != -1
    silhouette = np.nan
    if np.sum(valid) > 1 and len(np.unique(labels[valid])) > 1:
        try:
            silhouette = silhouette_score(embeddings[valid], labels[valid])
        except Exception:
            pass

    fig.text(
        0.01, 0.005,
        f"Datasets: {len(embeddings)}  ·  Clusters: {n_clusters}  "
        f"·  Outliers: {int(np.sum(labels == -1))}  "
        f"·  Silhouette: {silhouette:.3f}  "
        f"·  Embedding dim: {embeddings.shape[1]}",
        fontsize=9, color=_MUTED,
        bbox=dict(facecolor=_BG2, alpha=0.85, edgecolor=_BORDER,
                  boxstyle="round,pad=0.4"),
    )

    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"  ✓ Visualização salva em: {out}")
    return out


# ============================================================================
# GRÁFICO 2: DENDROGRAMA
# ============================================================================

def plot_dendrogram(
    embeddings:      np.ndarray,
    names:           List[str],
    output_filename: str = "dendrogram.jpg",
    method:          str = "ward",
) -> Optional[str]:
    """
    Gera dendrograma hierárquico com tema escuro. Salva em img/.
    """
    _ensure_dirs()
    from scipy.cluster.hierarchy import dendrogram, linkage

    try:
        Z = linkage(embeddings, method=method, metric="euclidean")
    except Exception as e:
        print(f"  ⚠ Erro ao calcular linkage: {e}")
        return None

    out = _img_path(output_filename)

    fig, ax = plt.subplots(figsize=(15, 8), facecolor=_BG)
    ax.set_facecolor(_BG2)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)

    ddata = dendrogram(
        Z,
        labels=names,
        leaf_rotation=60,
        leaf_font_size=9,
        show_contracted=True,
        truncate_mode="lastp",
        p=min(30, len(names)),
        ax=ax,
        # Cores uniformes — sobrescritas abaixo por link_color_func
        above_threshold_color=_MUTED,
        color_threshold=0,
    )

    # Recolore links com a paleta do projeto
    _palette = _C_POINT[:6] + [_MUTED]
    color_cycle = {}
    for i, (xs, ys, color) in enumerate(
        zip(ddata["icoord"], ddata["dcoord"], ddata["leaves_color_list"]
            if "leaves_color_list" in ddata else [_MUTED] * len(ddata["icoord"]))
    ):
        pass  # dendrogram já aplicou cores acima

    # Ajusta cores dos elementos do dendrograma desenhados
    for coll in ax.collections:
        coll.set_color(_C_POINT[0])
    for line in ax.get_lines():
        line.set_color(_C_POINT[0])
        line.set_alpha(0.75)

    ax.tick_params(axis="x", colors=_MUTED, labelsize=8)
    ax.tick_params(axis="y", colors=_MUTED, labelsize=8)
    ax.set_title(f"Dendrograma Hierárquico — método: {method}",
                 color=_TEXT, fontsize=14, fontweight="bold")
    ax.set_xlabel("Datasets", color=_MUTED, fontsize=11)
    ax.set_ylabel("Distância", color=_MUTED, fontsize=11)
    ax.grid(True, alpha=0.06, color="white", linestyle="--", axis="y")

    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"  ✓ Dendrograma salvo em: {out}")
    return out


# ============================================================================
# RELATÓRIOS TEXTUAIS
# ============================================================================

def export_cluster_report(
    embeddings:   np.ndarray,
    names:        List[str],
    labels:       np.ndarray,
    csv_filename: str = "cluster_report.csv",
    txt_filename: str = "cluster_analysis.txt",
) -> Tuple[str, str]:
    """
    Exporta relatório CSV e análise textual detalhada. Salva em output/.
    """
    _ensure_dirs()
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    csv_path = _output_path(csv_filename)
    txt_path = _output_path(txt_filename)

    # CSV
    df = pd.DataFrame({"dataset": names, "cluster": labels,
                        "is_outlier": labels == -1})
    if len(embeddings) > 2:
        try:
            coords = _reduce_to_2d(embeddings)
            df["tsne_x"] = coords[:, 0]
            df["tsne_y"] = coords[:, 1]
        except Exception:
            pass
    df.to_csv(csv_path, index=False)

    # Métricas
    valid = labels != -1
    silhouette = davies_bouldin = np.nan
    if np.sum(valid) > 1 and len(np.unique(labels[valid])) > 1:
        try:
            silhouette     = silhouette_score(embeddings[valid], labels[valid])
            davies_bouldin = davies_bouldin_score(embeddings[valid], labels[valid])
        except Exception:
            pass

    unique_labels = np.unique(labels)
    n_clusters    = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # TXT
    with open(txt_path, "w", encoding="utf-8") as f:
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
        f.write("    Silhouette     → mais próximo de 1 = melhor separação\n")
        f.write("    Davies-Bouldin → menor valor = clusters mais compactos\n\n")

        f.write("DISTRIBUIÇÃO DETALHADA:\n")
        for label in sorted(unique_labels):
            count  = int(np.sum(labels == label))
            pct    = count / len(names) * 100
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
    names:      List[str],
    labels:     np.ndarray,
    prefix:     str = "cluster_results",
) -> Dict[str, Optional[str]]:
    """
    Gera todos os gráficos e relatórios de clustering.

    Saídas:
      img/    → visualização JPG + dendrograma JPG
      output/ → relatório CSV + análise TXT

    Returns:
        Dicionário com os caminhos de cada arquivo gerado.
    """
    _ensure_dirs()
    print("\n" + "=" * 70)
    print("GERANDO GRÁFICOS E RELATÓRIOS")
    print(f"  Imagens    → {IMG_DIR}/")
    print(f"  Relatórios → {OUTPUT_DIR}/")
    print("=" * 70)

    results: Dict[str, Optional[str]] = {
        "visualization": None,
        "dendrogram":    None,
        "csv_report":    None,
        "text_report":   None,
    }

    try:
        results["visualization"] = plot_cluster_visualization(
            embeddings, names, labels,
            output_filename=f"{prefix}_visualization.jpg",
        )
    except Exception as e:
        print(f"  ⚠ Visualização falhou: {e}")

    try:
        results["dendrogram"] = plot_dendrogram(
            embeddings, names,
            output_filename=f"{prefix}_dendrogram.jpg",
        )
    except Exception as e:
        print(f"  ⚠ Dendrograma falhou: {e}")

    try:
        csv_path, txt_path = export_cluster_report(
            embeddings, names, labels,
            csv_filename=f"{prefix}_report.csv",
            txt_filename=f"{prefix}_analysis.txt",
        )
        results["csv_report"]  = csv_path
        results["text_report"] = txt_path
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
    dim     = 128
    centers = [np.ones(dim) * v for v in [0.0, 1.5, 3.0]]
    embeddings, names, true_labels = [], [], []
    for i, c in enumerate(centers):
        for j in range(4):
            embeddings.append(c + np.random.randn(dim) * 0.2)
            names.append(f"Dataset_C{i}_{j}")
            true_labels.append(i)

    generate_cluster_plots(
        np.array(embeddings), names, np.array(true_labels), prefix="test"
    )