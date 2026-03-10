"""
Instance-Level Distribution Analysis for Binary Malware Datasets
=================================================================
Compara datasets binários de malware analisando instâncias individuais.
Suporta datasets com colunas DIFERENTES (nomes de permissões heterogêneos).

Pipeline:
  1. Normalização dos nomes de features  (android.permission.X → X)
  2. Alinhamento dos espaços             (union com zeros / intersection)
  3. Opcional: embedding semântico       (sentence-transformers)
  4. MMD + Sliced Wasserstein            (métricas de distância)
  5. t-SNE + UMAP                        (visualização das instâncias)

Dependências:
    pip install umap-learn scikit-learn scipy matplotlib seaborn
    pip install sentence-transformers   # opcional, para embedding semântico
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.stats import wasserstein_distance

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("⚠️  UMAP não encontrado. Instale com: pip install umap-learn")
    UMAP_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


# ===========================================================================
# 1. ALINHAMENTO DE FEATURES HETEROGÊNEAS
# ===========================================================================

def normalize_permission(name: str) -> str:
    """
    Normaliza nomes de features de diferentes datasets para um formato comum.

    Exemplos:
      'android.permission.READ_SMS'                    → 'READ_SMS'
      'com.android.browser.permission.WRITE_BOOKMARKS' → 'WRITE_BOOKMARKS'
      'Dangerous Permission: INSTALL_PACKAGES'         → 'INSTALL_PACKAGES'
      'Suspicious Intent Filter: USER_PRESENT'         → 'INTENT_USER_PRESENT'
      'Hidden Apk'                                     → 'HIDDEN_APK'
      'Reads phone data at startup'                    → 'READS_PHONE_DATA_AT_STARTUP'
      'class'                                          → 'class'  (preservado)
    """
    if name == "class":
        return name

    name = name.strip()

    # Remove prefixos de pacote android/com.*
    name = re.sub(r'^[\w\.]+\.permission\.', '', name)

    # Remove prefixos descritivos com ':'
    name = re.sub(r'^Dangerous Permission:\s*',        '',        name, flags=re.IGNORECASE)
    name = re.sub(r'^Normal Permission:\s*',           '',        name, flags=re.IGNORECASE)
    name = re.sub(r'^Suspicious Intent Filter:\s*',    'INTENT_', name, flags=re.IGNORECASE)
    name = re.sub(r'^Intent Filter:\s*',               'INTENT_', name, flags=re.IGNORECASE)

    # Normaliza espaços/hifens/pontos → underscore, uppercase
    name = re.sub(r'[\s\-\.]+', '_', name.strip())
    name = name.upper()

    return name


def align_dataframes(
    dfs: list,
    names: list,
    strategy: str = "union",
    label_col: str = "class",
    verbose: bool = True
) -> tuple:
    """
    Alinha N DataFrames com colunas heterogêneas num espaço comum.

    Parâmetros
    ----------
    dfs       : lista de DataFrames (cada um pode ter colunas diferentes)
    names     : nomes dos datasets
    strategy  : 'union'        → todas as features, zeros onde ausente
                'intersection' → só features em comum
    label_col : coluna de rótulo a separar (não entra nas features)
    verbose   : imprime estatísticas do alinhamento

    Retorna
    -------
    arrays    : lista de np.ndarray (n_samples, n_features_alinhadas)
    labels    : lista de Series/None com rótulos de cada dataset
    cols      : lista das colunas finais alinhadas
    """
    # 1. Normalizar nomes de todas as colunas
    normalized = []
    for df in dfs:
        df = df.copy()
        df.columns = [normalize_permission(c) for c in df.columns]

        # Resolver duplicatas geradas pela normalização:
        # ex: "android.permission.READ_SMS" e "Dangerous Permission: READ_SMS"
        # ambos viram "READ_SMS" → fazer OR binário (max) e manter coluna única
        if df.columns.duplicated().any():
            dupes = df.columns[df.columns.duplicated(keep=False)].unique().tolist()
            for col in dupes:
                merged = df.loc[:, df.columns == col].max(axis=1)
                df = df.loc[:, df.columns != col].copy()
                df[col] = merged

        normalized.append(df)

    # 2. Separar features do label
    feature_sets = []
    label_list   = []
    for df in normalized:
        if label_col in df.columns:
            label_list.append(df[label_col])
            df = df.drop(columns=[label_col])
        else:
            label_list.append(None)
        feature_sets.append(set(df.columns))

    # 3. Calcular espaço alinhado
    if strategy == "intersection":
        cols = sorted(set.intersection(*feature_sets))
    else:  # union
        cols = sorted(set.union(*feature_sets))

    if verbose:
        print(f"\n{'─'*55}")
        print(f"Alinhamento de features ({strategy.upper()})")
        for name, fs in zip(names, feature_sets):
            overlap = len(fs & set(cols)) / len(cols) * 100
            print(f"  {name:35s}: {len(fs):4d} features  |  cobertura: {overlap:.1f}%")
        print(f"  {'Espaço final':35s}: {len(cols):4d} features")
        print(f"{'─'*55}")

    # 4. Reindexar com zeros nas features ausentes
    arrays = []
    for df in normalized:
        if label_col in df.columns:
            df = df.drop(columns=[label_col])
        # Garantia extra: remover qualquer duplicata residual antes do reindex
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        X = df.reindex(columns=cols, fill_value=0).values.astype(float)
        arrays.append(X)

    return arrays, label_list, cols


def align_with_semantic_embedding(
    dfs: list,
    names: list,
    label_col: str = "class",
    sim_threshold: float = 0.82,
    model_name: str = "all-MiniLM-L6-v2",
    verbose: bool = True
) -> tuple:
    """
    Alinha features usando embeddings semânticos (sentence-transformers).
    Útil quando nomes são parecidos mas não idênticos após normalização.

    Ex: 'READ_SMS' e 'READS_SMS_AT_STARTUP' → mapeados para a mesma feature.

    Requer: pip install sentence-transformers
    """
    if not SBERT_AVAILABLE:
        raise ImportError(
            "sentence-transformers não instalado.\npip install sentence-transformers"
        )

    print(f"\n🤖 Carregando modelo de embedding: {model_name}")
    model = SentenceTransformer(model_name)

    # Normalizar colunas
    norm_dfs = []
    for df in dfs:
        df = df.copy()
        df.columns = [normalize_permission(c) for c in df.columns]
        norm_dfs.append(df)

    # Usar as features do primeiro dataset como vocabulário de referência
    ref_cols = [c for c in norm_dfs[0].columns if c != label_col]
    ref_emb  = model.encode(ref_cols, show_progress_bar=False)

    aligned = []
    for i, df in enumerate(norm_dfs):
        feat_cols = [c for c in df.columns if c != label_col]

        if feat_cols == ref_cols:
            aligned.append(df[feat_cols].values.astype(float))
            continue

        cur_emb = model.encode(feat_cols, show_progress_bar=False)
        sim_mat = cosine_similarity(ref_emb, cur_emb)  # (ref, cur)

        X_new  = np.zeros((len(df), len(ref_cols)))
        mapped = 0
        for r_idx in range(len(ref_cols)):
            best_j = sim_mat[r_idx].argmax()
            best_s = sim_mat[r_idx, best_j]
            if best_s >= sim_threshold:
                X_new[:, r_idx] = df[feat_cols[best_j]].values
                mapped += 1

        if verbose:
            print(f"  {names[i]:35s}: {mapped}/{len(ref_cols)} features mapeadas "
                  f"(threshold={sim_threshold})")
        aligned.append(X_new)

    return aligned, ref_cols


# ===========================================================================
# 2. MÉTRICAS DE DISTÂNCIA ENTRE DISTRIBUIÇÕES
# ===========================================================================

def mmd(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """Maximum Mean Discrepancy. Retorna 0 quando distribuições são idênticas."""
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return float(XX.mean() + YY.mean() - 2 * XY.mean())


def sliced_wasserstein(X: np.ndarray, Y: np.ndarray,
                        n_projections: int = 100, seed: int = 42) -> float:
    """Sliced Wasserstein Distance via projeções aleatórias 1D."""
    rng = np.random.RandomState(seed)
    d   = X.shape[1]
    total = 0.0
    for _ in range(n_projections):
        v = rng.randn(d)
        v /= np.linalg.norm(v) + 1e-12
        total += wasserstein_distance(X @ v, Y @ v)
    return total / n_projections


def build_distance_matrix(arrays: list, metric: str = "mmd") -> np.ndarray:
    """Constrói matriz de distâncias N×N entre todos os pares."""
    n  = len(arrays)
    fn = mmd if metric == "mmd" else sliced_wasserstein
    D  = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = fn(arrays[i], arrays[j])
            D[i, j] = D[j, i] = d
    return D


# ===========================================================================
# 3. REDUÇÃO DE DIMENSIONALIDADE
# ===========================================================================

def _pca_reduce(X: np.ndarray, n: int = 50) -> np.ndarray:
    n_comp = min(n, X.shape[1], X.shape[0] - 1)
    return PCA(n_components=n_comp, random_state=42).fit_transform(X)


def reduce_tsne(X: np.ndarray, perplexity: int = 30) -> np.ndarray:
    if X.shape[1] > 50:
        print(f"  → PCA: {X.shape[1]}d → 50d")
        X = _pca_reduce(X)
    print(f"  → t-SNE: {X.shape[1]}d → 2d")
    return TSNE(n_components=2, perplexity=perplexity,
                random_state=42, max_iter=1000, init='pca').fit_transform(X)


def reduce_umap(X: np.ndarray, n_neighbors: int = 15,
                min_dist: float = 0.1) -> np.ndarray:
    if not UMAP_AVAILABLE:
        raise ImportError("pip install umap-learn")
    if X.shape[1] > 50:
        print(f"  → PCA: {X.shape[1]}d → 50d")
        X = _pca_reduce(X)
    print(f"  → UMAP: {X.shape[1]}d → 2d")
    return umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                     min_dist=min_dist, random_state=42).fit_transform(X)


# ===========================================================================
# 4. COMPARAÇÃO DIRETA DE DOIS DataFrames
# ===========================================================================

def compare_two_dataframes(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    name_a: str = "Dataset A",
    name_b: str = "Dataset B",
    label_col: str = "class",
    align_strategy: str = "union",
    samples: int = 300,
    seed: int = 42
) -> dict:
    """
    Pipeline completo para comparar dois datasets com colunas heterogêneas.

    Parâmetros
    ----------
    df_a, df_b       : DataFrames com features binárias e coluna 'class'
    name_a, name_b   : nomes dos datasets
    label_col        : coluna de rótulo (removida antes da análise)
    align_strategy   : 'union' | 'intersection'
    samples          : instâncias por dataset (sub-amostragem)

    Retorna
    -------
    dict com mmd, wasserstein, colunas alinhadas, DataFrames 2D
    """
    # 1. Alinhar features
    arrays, _, cols = align_dataframes(
        [df_a, df_b], [name_a, name_b],
        strategy=align_strategy, label_col=label_col
    )
    X, Y = arrays[0], arrays[1]

    # 2. Sub-amostragem balanceada
    rng = np.random.RandomState(seed)
    X   = X[rng.choice(len(X), min(len(X), samples), replace=False)]
    Y   = Y[rng.choice(len(Y), min(len(Y), samples), replace=False)]

    print(f"\nInstâncias usadas: {len(X)} ({name_a}) + {len(Y)} ({name_b})")
    print(f"Features alinhadas: {X.shape[1]}")

    # 3. Métricas
    print("\n📏 Calculando métricas de distância...")
    score_mmd = mmd(X, Y)
    score_w   = sliced_wasserstein(X, Y)
    print(f"  MMD:                {score_mmd:.6f}")
    print(f"  Sliced Wasserstein: {score_w:.6f}")

    # 4. Redução + plot
    X_all      = np.vstack([X, Y])
    labels_str = [name_a] * len(X) + [name_b] * len(Y)
    dfs_2d     = {}

    print("\n🔵 t-SNE...")
    pts = reduce_tsne(X_all)
    dfs_2d["t-SNE"] = pd.DataFrame(
        {"x": pts[:, 0], "y": pts[:, 1], "Dataset": labels_str}
    )

    if UMAP_AVAILABLE:
        print("\n🟣 UMAP...")
        pts = reduce_umap(X_all)
        dfs_2d["UMAP"] = pd.DataFrame(
            {"x": pts[:, 0], "y": pts[:, 1], "Dataset": labels_str}
        )

    _plot_two(dfs_2d, name_a, name_b, score_mmd, score_w)

    return {"mmd": score_mmd, "wasserstein": score_w, "cols": cols, "dfs_2d": dfs_2d}


# ===========================================================================
# 5. CLUSTERIZAÇÃO DE MÚLTIPLOS DataFrames
# ===========================================================================

def cluster_dataframes(
    dfs: list,
    names: list,
    n_clusters: int = 3,
    label_col: str = "class",
    align_strategy: str = "union",
    dist_metric: str = "mmd",
    samples_per_dataset: int = 60,
    seed: int = 42
) -> tuple:
    """
    Clusteriza N DataFrames com features heterogêneas via distribuição de instâncias.

    Parâmetros
    ----------
    dfs                 : lista de DataFrames (colunas podem ser diferentes)
    names               : nomes dos datasets
    n_clusters          : número de clusters
    label_col           : coluna de rótulo (ignorada na análise)
    align_strategy      : 'union' | 'intersection'
    dist_metric         : 'mmd' | 'wasserstein'
    samples_per_dataset : instâncias por dataset para visualização

    Retorna
    -------
    labels      : array de cluster por dataset
    dist_matrix : matriz de distâncias N×N
    """
    rng = np.random.RandomState(seed)

    # 1. Alinhar
    arrays, _, cols = align_dataframes(
        dfs, names, strategy=align_strategy, label_col=label_col
    )

    # 2. Matriz de distâncias
    print(f"\n📏 Construindo matriz de distâncias ({dist_metric.upper()})...")
    D = build_distance_matrix(arrays, metric=dist_metric)

    # 3. Clusterização hierárquica
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters, metric='precomputed', linkage='average'
    )
    labels = clusterer.fit_predict(D)
    score  = silhouette_score(D, labels, metric='precomputed')
    print(f"✅ Silhouette Score: {score:.4f}")

    # 4. Visualização das instâncias
    all_rows, meta = [], []
    for i, X in enumerate(arrays):
        idx = rng.choice(len(X), min(len(X), samples_per_dataset), replace=False)
        all_rows.append(X[idx])
        for _ in range(len(idx)):
            meta.append({"Dataset": names[i], "Cluster": f"C{labels[i]}"})

    X_vis  = np.vstack(all_rows)
    dfs_2d = {}

    print("\n🔵 t-SNE das instâncias...")
    pts = reduce_tsne(X_vis)
    df  = pd.DataFrame(meta)
    df["x"], df["y"] = pts[:, 0], pts[:, 1]
    dfs_2d["t-SNE"] = df

    if UMAP_AVAILABLE:
        print("\n🟣 UMAP das instâncias...")
        pts = reduce_umap(X_vis)
        df  = pd.DataFrame(meta)
        df["x"], df["y"] = pts[:, 0], pts[:, 1]
        dfs_2d["UMAP"] = df

    _plot_clusters(dfs_2d, labels, names, score, dist_metric, cols)

    # 5. Tabela resumo
    print("\n📋 TABELA DE CLUSTERS:")
    print(pd.DataFrame({"Dataset": names, "Cluster": labels})
          .sort_values("Cluster").to_string(index=False))

    return labels, D


# ===========================================================================
# 6. PLOTS
# ===========================================================================

def _plot_two(dfs_2d, name_a, name_b, score_mmd, score_w):
    n = len(dfs_2d)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]

    palette = {name_a: "#2196F3", name_b: "#F44336"}

    for ax, (method, df) in zip(axes, dfs_2d.items()):
        for name, color in palette.items():
            sub = df[df["Dataset"] == name]
            ax.scatter(sub["x"], sub["y"], c=color, label=name,
                       alpha=0.45, s=18, edgecolors='none')
        ax.set_title(method, fontsize=13, fontweight="bold")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.legend(title="Dataset")
        ax.grid(True, alpha=0.15)
        sns.despine(ax=ax)

    fig.suptitle(
        f"{name_a}  vs  {name_b}\n"
        f"MMD = {score_mmd:.4f}   |   Sliced Wasserstein = {score_w:.4f}",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    out = "comparison_two_datasets.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✅ Gráfico salvo: {out}")
    plt.show()


def _plot_clusters(dfs_2d, labels, names, silhouette, dist_metric, cols):
    n = len(dfs_2d)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 7))
    if n == 1:
        axes = [axes]

    n_clusters     = len(np.unique(labels))
    palette        = sns.color_palette("husl", n_clusters)
    cluster_colors = {f"C{i}": palette[i] for i in range(n_clusters)}
    ds_cluster     = {name: f"C{labels[i]}" for i, name in enumerate(names)}

    for ax, (method, df) in zip(axes, dfs_2d.items()):
        for cid, color in cluster_colors.items():
            sub = df[df["Cluster"] == cid]
            ax.scatter(sub["x"], sub["y"], c=[color], label=cid,
                       alpha=0.35, s=15, edgecolors='none')

        for name in names:
            sub   = df[df["Dataset"] == name]
            cx    = sub["x"].mean()
            cy    = sub["y"].mean()
            color = cluster_colors[ds_cluster[name]]
            ax.annotate(
                f"[{ds_cluster[name]}] {name}", (cx, cy),
                fontsize=8, fontweight="bold", color=color,
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.5)
            )

        ax.set_title(method, fontsize=13, fontweight="bold")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.legend(title="Cluster", fontsize=9)
        ax.grid(True, alpha=0.12)
        sns.despine(ax=ax)

    fig.suptitle(
        f"Clusterização via Instâncias  ({dist_metric.upper()})\n"
        f"Silhouette = {silhouette:.4f}  |  Features alinhadas = {len(cols)}",
        fontsize=12
    )
    plt.tight_layout()
    out = "cluster_instances_tsne_umap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✅ Gráfico salvo: {out}")
    plt.show()


def plot_distance_heatmap(dist_matrix: np.ndarray, names: list,
                           labels: np.ndarray, metric: str = "MMD"):
    """Heatmap da matriz de distâncias, ordenado por cluster."""
    order  = np.argsort(labels)
    D_ord  = dist_matrix[np.ix_(order, order)]
    n_ord  = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(6, len(names)), max(5, len(names) - 1)))
    sns.heatmap(D_ord, xticklabels=n_ord, yticklabels=n_ord,
                cmap="YlOrRd", annot=len(names) <= 12, fmt=".3f",
                linewidths=0.4, ax=ax)
    ax.set_title(f"Matriz de Distâncias ({metric})", fontsize=13)
    plt.tight_layout()
    out = "distance_heatmap.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"✅ Heatmap salvo: {out}")
    plt.show()


# ===========================================================================
# 7. EXEMPLO DE USO
# ===========================================================================

if __name__ == "__main__":

    rng = np.random.RandomState(0)

    # Dataset A: formato "android.permission.*"
    cols_a = [
        "android.permission.INTERNET",
        "android.permission.READ_SMS",
        "android.permission.WRITE_CALL_LOG",
        "android.permission.CAMERA",
        "android.permission.RECORD_AUDIO",
        "com.android.launcher.permission.INSTALL_SHORTCUT",
        "android.permission.ACCESS_FINE_LOCATION",
        "android.permission.READ_CONTACTS",
        "android.permission.SEND_SMS",
        "android.permission.RECEIVE_BOOT_COMPLETED",
    ]
    X_a  = (rng.rand(300, len(cols_a)) < 0.3).astype(int)
    df_a = pd.DataFrame(X_a, columns=cols_a)
    df_a["class"] = rng.randint(0, 2, 300)

    # Dataset B: formato "Dangerous Permission: *" (parcialmente sobreposto)
    cols_b = [
        "Dangerous Permission: INSTALL_PACKAGES",
        "Dangerous Permission: READ_SMS",        # mesmo que READ_SMS acima
        "Dangerous Permission: SEND_SMS",        # mesmo que SEND_SMS acima
        "Dangerous Permission: CAMERA",          # mesmo que CAMERA acima
        "Dangerous Permission: RECORD_AUDIO",
        "Suspicious Intent Filter: USER_PRESENT",
        "Hidden Apk",
        "Sends SMS to Suspicious Number(s)",
        "Reads phone data at startup",
        "Starts service at startup",
    ]
    X_b  = (rng.rand(300, len(cols_b)) < 0.65).astype(int)
    df_b = pd.DataFrame(X_b, columns=cols_b)
    df_b["class"] = rng.randint(0, 2, 300)

    # ------------------------------------------------------------------
    # EXEMPLO 1: Comparação direta de 2 datasets heterogêneos
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("EXEMPLO 1: Comparação direta (colunas heterogêneas)")
    print("="*60)

    result = compare_two_dataframes(
        df_a, df_b,
        name_a="MalwareA_android",
        name_b="MalwareB_dangerous",
        label_col="class",
        align_strategy="union",   # troque por 'intersection' para só features comuns
        samples=200
    )

    # ------------------------------------------------------------------
    # EXEMPLO 2: Clusterização de múltiplos datasets heterogêneos
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("EXEMPLO 2: Clusterização de múltiplos datasets")
    print("="*60)

    dfs_multi   = []
    names_multi = []
    for i in range(5):
        p   = 0.25 if i < 3 else 0.70
        col = cols_a if i % 2 == 0 else cols_b
        X   = (rng.rand(256, len(col)) < p).astype(int)
        df  = pd.DataFrame(X, columns=col)
        df["class"] = rng.randint(0, 2, 256)
        dfs_multi.append(df)
        names_multi.append(f"ds_{i:02d}_{'A' if i%2==0 else 'B'}")

    labels, D = cluster_dataframes(
        dfs_multi, names_multi,
        n_clusters=2,
        label_col="class",
        align_strategy="union",
        dist_metric="mmd",
        samples_per_dataset=60
    )

    plot_distance_heatmap(D, names_multi, labels, metric="MMD")