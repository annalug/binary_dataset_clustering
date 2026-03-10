"""
run_real_datasets.py
====================
Análise de clusterização via instâncias para datasets de malware.
Versão leve — não trava o computador.

Uso:
    python run_real_datasets.py
    python run_real_datasets.py --data ../data --clusters 3
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # salva arquivo sem abrir janela (bem mais leve)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import umap as umap_lib
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ===========================================================================
# PARÂMETROS DE LEVEZA
# ===========================================================================
PCA_DIM      = 20   # reduz cada dataset para 20d antes de qualquer cálculo
MMD_SAMPLES  = 60   # instâncias usadas no cálculo do MMD
VIS_SAMPLES  = 40   # instâncias por dataset no t-SNE/UMAP
TSNE_ITER    = 500
TSNE_PERP    = 15


# ===========================================================================
# 1. CARREGAMENTO
# ===========================================================================

def load_datasets(folder: str, label_col: str = "class"):
    csv_files = sorted(f for f in os.listdir(folder) if f.endswith(".csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em: {folder}")

    dfs, names = [], []
    print(f"\n📂 Carregando {len(csv_files)} datasets de '{folder}'")
    print(f"{'─'*58}")
    print(f"  {'Nome':<38} {'Amostras':>8} {'Features':>8}")
    print(f"{'─'*58}")

    for fname in csv_files:
        path = os.path.join(folder, fname)
        df   = pd.read_csv(path)
        name = fname.replace(".csv", "")

        # Detectar coluna de label
        for lc in [label_col, "Class", "CLASS", "label", "target"]:
            if lc in df.columns:
                if lc != label_col:
                    df = df.rename(columns={lc: label_col})
                break

        n_feat = len(df.columns) - (1 if label_col in df.columns else 0)
        print(f"  {name:<38} {len(df):>8} {n_feat:>8}")
        dfs.append(df)
        names.append(name)

    print(f"{'─'*58}\n")
    return dfs, names


# ===========================================================================
# 2. EXTRAÇÃO DE ARRAYS (sem alinhamento de nomes — cada dataset é independente)
# ===========================================================================

def extract_arrays(dfs: list, names: list, label_col: str = "class"):
    """
    Extrai os arrays de features de cada DataFrame.
    NÃO tenta alinhar colunas — cada dataset mantém seu próprio espaço.
    O alinhamento acontece via PCA (espaço latente compartilhado).
    """
    arrays = []
    for df, name in zip(dfs, names):
        feat_cols = [c for c in df.columns if c != label_col]
        X = df[feat_cols].values.astype(float)

        # Remover colunas constantes (não têm informação)
        mask = X.std(axis=0) > 0
        X    = X[:, mask]

        arrays.append(X)
    return arrays


# ===========================================================================
# 3. PCA INDEPENDENTE POR DATASET → espaço latente comum
# ===========================================================================

def project_to_latent(arrays: list, names: list, n_components: int = PCA_DIM):
    """
    Projeta cada dataset para um espaço latente de n_components dimensões via PCA.

    Por que isso funciona mesmo sem alinhar colunas?
    Os componentes principais capturam os padrões de covariância das instâncias.
    Dois datasets com distribuições similares terão estruturas de covariância
    similares, mesmo que os nomes das features sejam diferentes.
    """
    projected = []
    print(f"🗜️  PCA independente por dataset → {n_components}d")
    print(f"{'─'*58}")
    for X, name in zip(arrays, names):
        n = min(n_components, X.shape[1], X.shape[0] - 1)
        Xp = PCA(n_components=n, random_state=42).fit_transform(X)

        # Padding com zeros para garantir mesma dimensão final
        if Xp.shape[1] < n_components:
            pad = np.zeros((Xp.shape[0], n_components - Xp.shape[1]))
            Xp  = np.hstack([Xp, pad])

        projected.append(Xp)
        print(f"  {name:<38} {X.shape[1]:>5}d → {Xp.shape[1]:>2}d")

    print(f"{'─'*58}\n")
    return projected


# ===========================================================================
# 4. MATRIZ DE DISTÂNCIAS MMD
# ===========================================================================

def mmd(X, Y, gamma=1.0):
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return float(XX.mean() + YY.mean() - 2 * XY.mean())


def build_distance_matrix(projected: list, names: list,
                           n_samples: int = MMD_SAMPLES, seed: int = 42):
    rng = np.random.RandomState(seed)
    n   = len(projected)
    D   = np.zeros((n, n))

    print(f"📏 Calculando MMD entre pares ({n_samples} instâncias cada)...")
    print(f"{'─'*58}")
    for i in range(n):
        for j in range(i + 1, n):
            Xi = projected[i]
            Xj = projected[j]
            si = rng.choice(len(Xi), min(len(Xi), n_samples), replace=False)
            sj = rng.choice(len(Xj), min(len(Xj), n_samples), replace=False)
            d  = mmd(Xi[si], Xj[sj])
            D[i, j] = D[j, i] = d

        # Progresso linha a linha
        done = i + 1
        bar  = "█" * done + "░" * (n - done)
        print(f"  [{bar}] {done}/{n}  {names[i][:30]}", end="\r")

    print(f"\n{'─'*58}\n")
    return D


# ===========================================================================
# 5. CLUSTERIZAÇÃO
# ===========================================================================

def cluster(D: np.ndarray, n_clusters: int):
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters, metric='precomputed', linkage='average'
    )
    labels = clusterer.fit_predict(D)
    score  = silhouette_score(D, labels, metric='precomputed')
    print(f"✅ Silhouette Score: {score:.4f}\n")
    return labels, score


def find_best_k(D: np.ndarray, k_min: int = 2, k_max: int = None,
                out_dir: str = ".") -> tuple:
    """
    Testa k de k_min até k_max e escolhe o melhor pelo Silhouette Score.
    Também plota o gráfico de Silhouette por k.
    """
    n = D.shape[0]
    if k_max is None:
        k_max = min(n - 1, 8)

    print(f"\n🔍 Buscando melhor número de clusters (k={k_min}..{k_max})...")
    print(f"{'─'*40}")

    scores = {}
    for k in range(k_min, k_max + 1):
        clusterer = AgglomerativeClustering(
            n_clusters=k, metric='precomputed', linkage='average'
        )
        lbs   = clusterer.fit_predict(D)
        score = silhouette_score(D, lbs, metric='precomputed')
        scores[k] = (score, lbs)
        bar = "█" * int(score * 20)
        print(f"  k={k}  Silhouette={score:.4f}  {bar}")

    best_k      = max(scores, key=lambda k: scores[k][0])
    best_score  = scores[best_k][0]
    best_labels = scores[best_k][1]

    print(f"{'─'*40}")
    print(f"  ✅ Melhor k = {best_k}  (Silhouette = {best_score:.4f})\n")

    # Plot Silhouette por k
    ks       = list(scores.keys())
    sil_vals = [scores[k][0] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    ax.plot(ks, sil_vals, color="#4fc3f7", linewidth=2, marker='o',
            markersize=8, markerfacecolor="white", markeredgecolor="#4fc3f7")
    ax.scatter([best_k], [best_score], color="#ef5350", s=120, zorder=5,
               label=f"Melhor k={best_k} ({best_score:.3f})")
    ax.axvline(best_k, color="#ef5350", linestyle="--", alpha=0.4)
    ax.set_xlabel("Número de clusters (k)", color="#aaaaaa")
    ax.set_ylabel("Silhouette Score", color="#aaaaaa")
    ax.set_title("Seleção automática de k", color="white", fontsize=12)
    ax.tick_params(colors="#666666")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.1, color="white", linestyle="--")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white")
    plt.tight_layout()
    out = os.path.join(out_dir, "silhouette_by_k.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="#0f0f1a")
    print(f"✅ Gráfico de k salvo: {out}")
    plt.close()

    return best_k, best_score, best_labels


# ===========================================================================
# 6. VISUALIZAÇÃO
# ===========================================================================

def plot_tsne_umap(projected, names, labels, use_umap=False,
                   n_samples=VIS_SAMPLES, seed=42, out_dir="."):
    rng = np.random.RandomState(seed)

    all_rows, meta = [], []
    for i, X in enumerate(projected):
        idx = rng.choice(len(X), min(len(X), n_samples), replace=False)
        all_rows.append(X[idx])
        for _ in range(len(idx)):
            meta.append({
                "Dataset": names[i].replace("balanced_", "").replace("rfe_", "")
                                    .replace("semidroid_", "sem_")
                                    .replace("statistical_", "stat_"),
                "Cluster": f"C{labels[i]}"
            })

    X_vis   = np.vstack(all_rows)
    df_meta = pd.DataFrame(meta)

    methods = {}

    # t-SNE
    print("🔵 t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(TSNE_PERP, len(X_vis) - 1),
                random_state=42, max_iter=TSNE_ITER, init='pca')
    X_2d = tsne.fit_transform(X_vis)
    methods["t-SNE"] = X_2d

    # UMAP (opcional)
    if use_umap and UMAP_AVAILABLE:
        print("🟣 UMAP...")
        reducer = umap_lib.UMAP(n_components=2, n_neighbors=10,
                                min_dist=0.1, random_state=42)
        methods["UMAP"] = reducer.fit_transform(X_vis)
    elif use_umap and not UMAP_AVAILABLE:
        print("⚠️  UMAP não instalado. pip install umap-learn")

    n_plots   = len(methods)
    fig, axes = plt.subplots(1, n_plots, figsize=(11 * n_plots, 8),
                             facecolor="#0f0f1a")
    if n_plots == 1:
        axes = [axes]

    n_cl = len(np.unique(labels))

    # Paleta distinta e de alto contraste por cluster
    # Cores dos pontos (alpha baixo — fundo da nuvem)
    POINT_COLORS = [
        "#4fc3f7",  # azul claro
        "#ef5350",  # vermelho
        "#66bb6a",  # verde
        "#ffa726",  # laranja
        "#ab47bc",  # roxo
        "#26c6da",  # ciano
    ]
    # Cores dos labels (mais saturadas que os pontos)
    LABEL_COLORS = [
        "#e1f5fe",  # branco azulado
        "#ffcdd2",  # rosa claro
        "#c8e6c9",  # verde claro
        "#ffe0b2",  # laranja claro
        "#f3e5f5",  # lilás claro
        "#e0f7fa",  # ciano claro
    ]
    # Cores do contorno dos pontos (mesma família, mais saturada)
    EDGE_COLORS = [
        "#0288d1",
        "#c62828",
        "#2e7d32",
        "#e65100",
        "#6a1b9a",
        "#00838f",
    ]

    point_colors = {f"C{i}": POINT_COLORS[i % len(POINT_COLORS)] for i in range(n_cl)}
    label_colors = {f"C{i}": LABEL_COLORS[i % len(LABEL_COLORS)] for i in range(n_cl)}
    edge_colors  = {f"C{i}": EDGE_COLORS[i % len(EDGE_COLORS)]   for i in range(n_cl)}

    ds_cluster = {
        row["Dataset"]: row["Cluster"]
        for _, row in df_meta.drop_duplicates("Dataset").iterrows()
    }

    for ax, (method, X_2d) in zip(axes, methods.items()):
        ax.set_facecolor("#0f0f1a")
        df_meta["x"], df_meta["y"] = X_2d[:, 0], X_2d[:, 1]

        # --- Pontos com borda sutil para separar sobreposições ---
        for cid in sorted(point_colors):
            sub = df_meta[df_meta["Cluster"] == cid]
            if sub.empty:
                continue
            # Sombra/halo atrás dos pontos
            ax.scatter(sub["x"], sub["y"],
                       c=edge_colors[cid], s=55, alpha=0.15,
                       edgecolors='none', zorder=2)
            # Pontos principais
            ax.scatter(sub["x"], sub["y"],
                       c=point_colors[cid], s=28, alpha=0.75,
                       edgecolors=edge_colors[cid], linewidths=0.4,
                       label=cid, zorder=3)

        # --- Labels dos datasets ---
        for ds_name in sorted(df_meta["Dataset"].unique()):
            sub  = df_meta[df_meta["Dataset"] == ds_name]
            cid  = ds_cluster.get(ds_name, "C0")
            cx, cy = sub["x"].mean(), sub["y"].mean()

            # Ponto central destacado
            ax.scatter(cx, cy,
                       c=point_colors[cid], s=90, zorder=5,
                       edgecolors="white", linewidths=1.2)

            # Caixa de texto com borda colorida
            ax.annotate(
                f"{ds_name}",
                (cx, cy),
                xytext=(0, 12), textcoords="offset points",
                fontsize=7.5, fontweight="bold",
                color=label_colors[cid],
                ha="center",
                bbox=dict(
                    facecolor="#1a1a2e",
                    alpha=0.85,
                    edgecolor=edge_colors[cid],
                    linewidth=0.8,
                    boxstyle="round,pad=0.3"
                ),
                zorder=6
            )

        # --- Estilo do eixo ---
        ax.set_title(method, fontsize=14, fontweight="bold",
                     color="white", pad=12)
        ax.set_xlabel("Dim 1", color="#aaaaaa", fontsize=10)
        ax.set_ylabel("Dim 2", color="#aaaaaa", fontsize=10)
        ax.tick_params(colors="#666666")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        ax.grid(True, alpha=0.08, color="white", linestyle="--")

        # Legenda com fundo escuro
        legend = ax.legend(
            title="Cluster", fontsize=9, title_fontsize=9,
            facecolor="#1a1a2e", edgecolor="#333355",
            labelcolor="white",
            loc="upper right"
        )
        legend.get_title().set_color("#aaaaaa")

    fig.suptitle(
        f"Clusterização via Instâncias  (PCA {PCA_DIM}d + MMD)\n"
        f"Silhouette = {score_global:.4f}",
        fontsize=13, color="white", y=1.01
    )
    plt.tight_layout()
    out = os.path.join(out_dir, "cluster_instances_tsne_umap.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="#0f0f1a")
    print(f"✅ Gráfico salvo: {out}")
    plt.close()


def plot_heatmap(D, names, labels, out_dir="."):
    short = [n.replace("balanced_", "").replace("rfe_", "")
              .replace("semidroid_", "sem_").replace("statistical_", "stat_")
             for n in names]
    order  = np.argsort(labels)
    D_ord  = D[np.ix_(order, order)]
    n_ord  = [short[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(7, len(names)), max(6, len(names) - 1)))
    sns.heatmap(D_ord, xticklabels=n_ord, yticklabels=n_ord,
                cmap="YlOrRd", annot=True, fmt=".3f",
                linewidths=0.5, ax=ax)
    ax.set_title("Matriz de Distâncias MMD (ordenada por cluster)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, "distance_heatmap.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"✅ Heatmap salvo: {out}")
    plt.close()


def print_summary(names, labels, D):
    print("\n" + "="*55)
    print("RESULTADO FINAL")
    print("="*55)
    df = pd.DataFrame({"Dataset": names, "Cluster": labels}).sort_values("Cluster")
    print(df.to_string(index=False))

    print(f"\nPares MAIS PRÓXIMOS (MMD):")
    pairs = [(i, j, D[i,j]) for i in range(len(names))
             for j in range(i+1, len(names))]
    pairs.sort(key=lambda x: x[2])
    for i, j, d in pairs[:3]:
        print(f"  {names[i][:30]:30s} ↔ {names[j][:30]:30s}  MMD={d:.4f}")

    print(f"\nPares MAIS DISTANTES (MMD):")
    for i, j, d in pairs[-3:]:
        print(f"  {names[i][:30]:30s} ↔ {names[j][:30]:30s}  MMD={d:.4f}")

    df.to_csv("clustering_summary.csv", index=False)
    print("\n✅ Resumo salvo: clustering_summary.csv")


# ===========================================================================
# MAIN
# ===========================================================================

score_global = 0.0  # preenchido no main, usado no plot


def main():
    global score_global

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     default="../data")
    parser.add_argument("--clusters", type=int, default=None,
                        help="Número de clusters. Se omitido, escolhe automaticamente.")
    parser.add_argument("--umap",     action="store_true",
                        help="Ativar UMAP (mais lento)")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Carregar
    dfs, names = load_datasets(args.data)

    # 2. Extrair arrays
    arrays = extract_arrays(dfs, names)

    # 3. PCA independente → espaço latente comum de PCA_DIM dimensões
    projected = project_to_latent(arrays, names, n_components=PCA_DIM)

    # 4. Matriz de distâncias MMD
    D = build_distance_matrix(projected, names)

    # 5. Clusterização — automática ou manual
    if args.clusters is None:
        best_k, score_global, labels = find_best_k(D, out_dir=out_dir)
        print(f"🔬 Usando k={best_k} (escolhido automaticamente)")
    else:
        print(f"🔬 Clusterizando em {args.clusters} grupos (definido manualmente)...")
        labels, score_global = cluster(D, args.clusters)
        best_k = args.clusters

    # 6. Visualização
    plot_tsne_umap(projected, names, labels,
                   use_umap=args.umap, out_dir=out_dir)
    plot_heatmap(D, names, labels, out_dir=out_dir)

    # 7. Resumo
    print_summary(names, labels, D)


if __name__ == "__main__":
    main()