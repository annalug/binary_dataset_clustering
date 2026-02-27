"""
version_a_direct.py — Versão A: Embeddings fixos + distância direta

Fluxo:
  1. Cada dataset → embedding fixo (estatísticas + semântica dos nomes das colunas)
  2. Matriz de distâncias calculada diretamente (sem treino)
  3. Clustering hierárquico + DBSCAN
  4. Comparação com agrupamento do time de negócios
  5. Visualizações salvas em img/  |  relatórios em output/

Não requer TensorFlow. Dependências: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from cluster_selection import find_best_k, plot_k_selection
from scipy import stats
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

IMG_DIR    = "img"
OUTPUT_DIR = "output"
DATA_DIR   = "./data"
CLASS_COL  = "class"

# Dimensões do embedding
SEMANTIC_DIM  = 50   # dimensões do componente semântico (TF-IDF + SVD)
STAT_DIM      = 50   # dimensões do componente estatístico

# Clustering — k é escolhido automaticamente pelos dados
K_MIN         = 2    # menor k a testar
K_MAX         = None # None = min(n_datasets-1, 5)
DBSCAN_EPS    = 1.5  # ajuste conforme seus dados
DBSCAN_MIN    = 2

# Agrupamento do time de negócios
# Preencha com os grupos que o time de negócios te passou.
# Chave = nome do dataset (sem .csv), valor = grupo (int ou string)
BUSINESS_GROUPS: Dict[str, int] = {
    # Exemplo — substitua pelos grupos reais:
    # "balanced_adroit":                          0,
    # "balanced_androcrawl":                      0,
    # "balanced_android_permissions":             0,
    # "balanced_drebin215":                       1,
    # "balanced_defensedroid_apicalls_closeness": 1,
    # "balanced_defensedroid_apicalls_degree":    1,
    # "balanced_defensedroid_apicalls_katz":      1,
    # "balanced_defensedroid_prs":                2,
    # "balanced_kronodroid_emulator":             2,
    # "balanced_kronodroid_real_device":          2,
}


def _ensure_dirs():
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _img(f):    return os.path.join(IMG_DIR, f)
def _out(f):    return os.path.join(OUTPUT_DIR, f)


# ============================================================================
# 1. CARREGAMENTO
# ============================================================================

def load_datasets(folder: str, class_col: str = CLASS_COL):
    """Carrega todos os CSVs da pasta, separa X e nomes das colunas."""
    datasets, names, column_names_list = [], [], []

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(folder, fname)
        df   = pd.read_csv(path)

        if class_col not in df.columns:
            print(f"  ⚠ '{fname}' sem coluna '{class_col}' — pulando")
            continue

        X     = df.drop(columns=[class_col]).values
        X     = np.clip(X, 0, 1).astype(np.float32)
        cols  = [c for c in df.columns if c != class_col]
        name  = fname.replace(".csv", "")

        datasets.append(X)
        names.append(name)
        column_names_list.append(cols)

        print(f"  ✓ {name}: {X.shape[0]} amostras × {X.shape[1]} features")

    return datasets, names, column_names_list


# ============================================================================
# 2. EMBEDDING FIXO
# ============================================================================

class DatasetEmbedder:
    """
    Gera um vetor de representação fixo para cada dataset a partir de:
      - Componente estatístico: densidade, variância, histogramas, correlações
      - Componente semântico:   TF-IDF nos nomes das colunas + SVD

    Nenhum treino necessário — os embeddings são determinísticos.
    """

    def __init__(self, semantic_dim: int = SEMANTIC_DIM, stat_dim: int = STAT_DIM):
        self.semantic_dim = semantic_dim
        self.stat_dim     = stat_dim
        self._tfidf = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 2),
            stop_words='english',
            analyzer='word',
            token_pattern=r"[a-zA-Z][a-zA-Z0-9]+"
        )
        self._svd         = TruncatedSVD(n_components=semantic_dim, random_state=42)
        self._tfidf_fitted = False
        self._svd_fitted   = False

    # ------------------------------------------------------------------
    # ESTATÍSTICAS
    # ------------------------------------------------------------------

    def _stat_features(self, X: np.ndarray) -> np.ndarray:
        """Extrai ~50 features estatísticas do dataset."""
        col_means = np.mean(X, axis=0)   # densidade por coluna
        col_stds  = np.std(X,  axis=0)

        feats = []

        # Globais (6)
        feats += [
            np.mean(X),                          # densidade global
            np.std(X),                           # variância global
            np.mean(col_means),                  # média das densidades
            np.std(col_means),                   # dispersão das densidades
            np.mean(col_stds),                   # média dos desvios
            float(np.mean(col_stds == 0)),       # % colunas constantes
        ]

        # Percentis da distribuição de densidades por coluna (9)
        for p in [5, 10, 25, 50, 75, 90, 95, 99, 100]:
            feats.append(float(np.percentile(col_means, p)))

        # Histograma de densidades (10 bins → 10 features)
        hist, _ = np.histogram(col_means, bins=10, range=(0, 1))
        feats  += (hist / max(hist.sum(), 1)).tolist()

        # Correlação média entre colunas (1)
        if X.shape[1] > 1:
            sample = X[:, :min(50, X.shape[1])]   # limita para não explodir memória
            corr   = np.corrcoef(sample.T)
            np.fill_diagonal(corr, 0)
            feats.append(float(np.mean(np.abs(corr))))
        else:
            feats.append(0.0)

        # Skewness e kurtosis médios das colunas (2)
        try:
            sk = float(np.mean([stats.skew(X[:, i])     for i in range(min(X.shape[1], 100))]))
            ku = float(np.mean([stats.kurtosis(X[:, i]) for i in range(min(X.shape[1], 100))]))
        except Exception:
            sk, ku = 0.0, 0.0
        feats += [sk, ku]

        # Proporções de features com densidade em faixas (5)
        for lo, hi in [(0, .1), (.1, .3), (.3, .7), (.7, .9), (.9, 1.01)]:
            feats.append(float(np.mean((col_means >= lo) & (col_means < hi))))

        # Shape normalizado (2)
        feats += [
            min(X.shape[0] / 10000, 1.0),
            min(X.shape[1] / 1000,  1.0),
        ]

        arr = np.array(feats, dtype=np.float32)

        # Pad ou trunca para stat_dim
        if len(arr) < self.stat_dim:
            arr = np.concatenate([arr, np.zeros(self.stat_dim - len(arr))])
        return arr[:self.stat_dim]

    # ------------------------------------------------------------------
    # SEMÂNTICA
    # ------------------------------------------------------------------

    def _preprocess_col_names(self, col_names: List[str]) -> List[str]:
        """Limpa prefixos típicos de features Android."""
        cleaned = []
        prefixes = [
            "android.permission.", "dangerous permission:",
            "permission:", "suspicious intent filter:",
            "com.android.", "balanced_", "_balanced",
        ]
        for name in col_names:
            s = str(name).lower()
            for p in prefixes:
                s = s.replace(p, " ")
            s = s.replace("_", " ").replace(".", " ")
            s = " ".join(s.split())
            cleaned.append(s if s else "unknown")
        return cleaned

    def _semantic_features(self, col_names: List[str]) -> np.ndarray:
        """TF-IDF nos nomes das colunas → SVD → vetor (semantic_dim,)."""
        docs      = self._preprocess_col_names(col_names)
        tfidf_mat = self._tfidf.transform(docs)   # tfidf já foi fitted em embed()
        n_vocab   = tfidf_mat.shape[1]

        if n_vocab < 2:
            # Vocabulário muito pequeno — usa média direta sem SVD
            dense   = tfidf_mat.toarray().astype(np.float32)
            reduced = dense
        elif not self._svd_fitted:
            n_comp = min(self.semantic_dim, n_vocab - 1, tfidf_mat.shape[0] - 1)
            self._svd.set_params(n_components=max(1, n_comp))
            reduced = self._svd.fit_transform(tfidf_mat)
            self._svd_fitted = True
        else:
            reduced = self._svd.transform(tfidf_mat)

        # Agrega embeddings de colunas → embedding de dataset (mean, max, std)
        agg = np.concatenate([
            np.mean(reduced, axis=0),
            np.max(reduced,  axis=0),
            np.std(reduced,  axis=0),
        ])

        # Garante exatamente semantic_dim dimensões
        if len(agg) > self.semantic_dim:
            agg = agg[:self.semantic_dim]
        elif len(agg) < self.semantic_dim:
            agg = np.concatenate([agg, np.zeros(self.semantic_dim - len(agg))])

        return agg.astype(np.float32)

    # ------------------------------------------------------------------
    # EMBED
    # ------------------------------------------------------------------

    def embed(
        self,
        datasets:    List[np.ndarray],
        names:       List[str],
        col_names_list: List[List[str]],
    ) -> np.ndarray:
        """
        Gera matriz de embeddings (n_datasets, stat_dim + semantic_dim).

        Na primeira chamada, faz fit do TF-IDF em todos os datasets juntos
        para ter um vocabulário consistente.
        """
        print(f"\n{'='*70}")
        print(f"GERANDO EMBEDDINGS FIXOS ({self.stat_dim + self.semantic_dim} dim)")
        print(f"  Estatístico : {self.stat_dim} dim")
        print(f"  Semântico   : {self.semantic_dim} dim")
        print(f"{'='*70}")

        # Fit global do TF-IDF (todos os nomes de colunas juntos)
        if not self._tfidf_fitted:
            all_docs = []
            for cols in col_names_list:
                all_docs.extend(self._preprocess_col_names(cols))
            self._tfidf.fit(all_docs)
            self._tfidf_fitted = True

        embeddings = []
        for i, (X, name, cols) in enumerate(zip(datasets, names, col_names_list), 1):
            stat = self._stat_features(X)
            sem  = self._semantic_features(cols)
            emb  = np.concatenate([stat, sem])
            embeddings.append(emb)
            print(f"  [{i:2d}/{len(datasets)}] {name}")

        matrix = np.array(embeddings, dtype=np.float32)
        print(f"\n  ✓ Shape: {matrix.shape}")
        return matrix


# ============================================================================
# 3. CLUSTERING
# ============================================================================

def run_clustering(embeddings: np.ndarray, names: List[str]) -> Dict:
    """
    Seleção automática de k + DBSCAN como segunda opinião.
    Nenhum k pré-definido — os dados decidem.
    """
    scaler = StandardScaler()
    X      = scaler.fit_transform(embeddings)

    # Seleção automática de k
    k_results   = find_best_k(X, names, k_min=K_MIN, k_max=K_MAX)
    best_k      = k_results['best_k']
    labels_hier = k_results['labels'][best_k]

    # DBSCAN como segunda opinião (sem k fixo)
    db        = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN)
    labels_db = db.fit_predict(X)

    def _silhouette(emb, lbls):
        valid = lbls != -1
        if np.sum(valid) > 1 and len(np.unique(lbls[valid])) > 1:
            try:
                return silhouette_score(emb[valid], lbls[valid])
            except Exception:
                pass
        return float('nan')

    sil_hier = _silhouette(X, labels_hier)
    sil_db   = _silhouette(X, labels_db)

    n_db = len(set(labels_db) - {-1})
    print(f"\n  DBSCAN (eps={DBSCAN_EPS}) → {n_db} clusters:")
    for lbl in sorted(set(labels_db)):
        members = [n for n, l in zip(names, labels_db) if l == lbl]
        tag = "Outliers" if lbl == -1 else f"Cluster {lbl}"
        print(f"    {tag}: {', '.join(members)}")
    print(f"  Silhouette DBSCAN: {sil_db:.4f}")

    return {
        'hierarchical': labels_hier,
        'dbscan':        labels_db,
        'X_scaled':      X,
        'best_k':        best_k,
        'k_results':     k_results,
        'silhouette':    {'hierarchical': sil_hier, 'dbscan': sil_db},
    }


# ============================================================================
# 4. COMPARAÇÃO COM NEGÓCIOS
# ============================================================================

def compare_with_business(
    names:          List[str],
    model_labels:   np.ndarray,
    business_groups: Dict[str, int],
    method_name:    str = "hierarchical",
) -> Dict:
    """
    Compara clusters do modelo com agrupamento do time de negócios.
    Calcula ARI e NMI. Funciona mesmo se BUSINESS_GROUPS estiver vazio.
    """
    print(f"\n{'='*70}")
    print(f"COMPARAÇÃO COM TIME DE NEGÓCIOS ({method_name})")
    print(f"{'='*70}")

    if not business_groups:
        print("\n  ⚠ BUSINESS_GROUPS está vazio.")
        print("  Preencha o dicionário BUSINESS_GROUPS no topo do script")
        print("  com os grupos que o time de negócios te passou.")
        return {}

    # Alinha labels
    bus_labels = np.array([business_groups.get(n, -1) for n in names])
    mod_labels = model_labels.copy()

    # Filtra datasets sem grupo de negócios definido
    mask = bus_labels != -1
    if mask.sum() < 2:
        print("  ⚠ Menos de 2 datasets com grupo de negócios definido.")
        return {}

    bus_valid = bus_labels[mask]
    mod_valid = mod_labels[mask]
    nms_valid = [n for n, m in zip(names, mask) if m]

    ari = adjusted_rand_score(bus_valid, mod_valid)
    nmi = normalized_mutual_info_score(bus_valid, mod_valid)

    print(f"\n  ARI (Adjusted Rand Index) : {ari:.4f}")
    print(f"  NMI (Norm. Mutual Info)   : {nmi:.4f}")
    print()
    print("  Interpretação:")
    print(f"    ARI = 1.0 → concordância perfeita com negócios")
    print(f"    ARI = 0.0 → agrupamento aleatório")
    print(f"    ARI < 0   → pior que aleatório")

    print(f"\n  Comparação por dataset:")
    print(f"    {'Dataset':45s} {'Modelo':10s} {'Negócios':10s} {'Match'}")
    print(f"    {'-'*70}")
    for name, ml, bl in zip(nms_valid, mod_valid, bus_valid):
        match = "✓" if ml == bl else "✗"
        # Nota: match direto é rígido demais (labels podem ter IDs diferentes)
        # ARI/NMI são as métricas corretas; isso é só para inspeção visual
        print(f"    {name:45s} {ml:<10} {bl:<10} {match}")

    return {'ari': ari, 'nmi': nmi, 'business_labels': bus_labels}


# ============================================================================
# 5. VISUALIZAÇÕES
# ============================================================================

def plot_all(
    embeddings:      np.ndarray,
    names:           List[str],
    cluster_results: Dict,
    business_groups: Dict[str, int],
):
    _ensure_dirs()
    X_scaled = cluster_results['X_scaled']

    # Redução 2D via PCA (determinística, sem parâmetros)
    pca    = PCA(n_components=2, random_state=42)
    X_2d   = pca.fit_transform(X_scaled)
    var_ex = pca.explained_variance_ratio_

    # ---- Fig 1: clusters lado a lado ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Versão A — Embeddings Fixos (estatística + semântica)", fontsize=14, fontweight='bold')

    best_k  = cluster_results['best_k']
    configs = [
        ('hierarchical', f'Hierárquico — k={best_k} (auto)', cluster_results['hierarchical']),
        ('dbscan',       f'DBSCAN (eps={DBSCAN_EPS})',       cluster_results['dbscan']),
    ]

    cmap = plt.cm.tab10

    for ax, (key, title, labels) in zip(axes[:2], configs):
        unique = np.unique(labels)
        colors = cmap(np.linspace(0, 0.9, len(unique)))

        for lbl, col in zip(unique, colors):
            mask = labels == lbl
            tag  = "Outliers" if lbl == -1 else f"Cluster {lbl}"
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[col], s=160, label=tag,
                       edgecolors='white', linewidth=1.2, zorder=3)
            for idx in np.where(mask)[0]:
                ax.annotate(names[idx], (X_2d[idx, 0], X_2d[idx, 1]),
                            fontsize=7, ha='center', va='bottom',
                            xytext=(0, 6), textcoords='offset points')

        sil = cluster_results['silhouette'][key]
        ax.set_title(f"{title}\nSilhouette: {sil:.3f}", fontsize=11)
        ax.set_xlabel(f"PC1 ({var_ex[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_ex[1]*100:.1f}%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Terceiro painel: grupos de negócios (se disponível)
    ax3 = axes[2]
    if business_groups:
        bus_labels = np.array([business_groups.get(n, -1) for n in names])
        unique_bus = np.unique(bus_labels[bus_labels != -1])
        colors_bus = cmap(np.linspace(0, 0.9, len(unique_bus) + 1))

        for lbl, col in zip(np.unique(bus_labels), colors_bus):
            mask = bus_labels == lbl
            tag  = "Sem grupo" if lbl == -1 else f"Negócios {lbl}"
            ax3.scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=[col], s=160, label=tag, marker='D',
                        edgecolors='white', linewidth=1.2, zorder=3)
            for idx in np.where(mask)[0]:
                ax3.annotate(names[idx], (X_2d[idx, 0], X_2d[idx, 1]),
                             fontsize=7, ha='center', va='bottom',
                             xytext=(0, 6), textcoords='offset points')
        ax3.set_title("Agrupamento do Time de Negócios", fontsize=11)
    else:
        ax3.text(0.5, 0.5, "Preencha\nBUSINESS_GROUPS\nno topo do script",
                 ha='center', va='center', transform=ax3.transAxes, fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax3.set_title("Agrupamento do Time de Negócios", fontsize=11)

    ax3.set_xlabel(f"PC1 ({var_ex[0]*100:.1f}%)")
    ax3.set_ylabel(f"PC2 ({var_ex[1]*100:.1f}%)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = _img("version_a_clusters.jpg")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {path}")

    # ---- Fig extra: seleção de k ----
    plot_k_selection(
        cluster_results['k_results'],
        title=f"Versão A — Seleção do Número de Clusters (best k={best_k})",
        output_path=_img("version_a_k_selection.jpg")
    )

    # ---- Fig 2: heatmap de similaridade ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Versão A — Matriz de Similaridade", fontsize=14, fontweight='bold')

    sim = cosine_similarity(embeddings)
    # ordena por cluster hierárquico
    order = np.argsort(cluster_results['hierarchical'])
    sim_ord  = sim[order][:, order]
    names_ord = [names[i] for i in order]

    for ax, (mat, title) in zip(axes, [
        (sim,     "Original (ordem dos CSVs)"),
        (sim_ord, f"Ordenada — Hierárquico k={best_k} (auto)"),
    ]):
        im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        n_list = names_ord if "Ordenada" in title else names
        ax.set_xticklabels(n_list, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(n_list, fontsize=7)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path2 = _img("version_a_similarity.jpg")
    plt.savefig(path2, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {path2}")

    # ---- Fig 3: dendrograma ----
    Z = linkage(X_scaled, method='ward')
    plt.figure(figsize=(14, 6))
    dendrogram(Z, labels=names, leaf_rotation=45, leaf_font_size=9)
    plt.title("Versão A — Dendrograma Hierárquico", fontsize=13, fontweight='bold')
    plt.xlabel("Dataset")
    plt.ylabel("Distância")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path3 = _img("version_a_dendrogram.jpg")
    plt.savefig(path3, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {path3}")

    return [path, path2, path3]


# ============================================================================
# 6. RELATÓRIOS
# ============================================================================

def save_reports(
    names:           List[str],
    embeddings:      np.ndarray,
    cluster_results: Dict,
    business_metrics: Dict,
):
    _ensure_dirs()

    # CSV com todos os resultados
    df = pd.DataFrame({
        'dataset':       names,
        'cluster_hier':  cluster_results['hierarchical'],
        'cluster_dbscan': cluster_results['dbscan'],
    })
    if business_metrics.get('business_labels') is not None:
        df['business_group'] = business_metrics['business_labels']

    csv_path = _out("version_a_clusters.csv")
    df.to_csv(csv_path, index=False)
    print(f"  ✓ {csv_path}")

    # Matriz de similaridade
    sim = pd.DataFrame(
        cosine_similarity(embeddings),
        index=names, columns=names
    )
    sim_path = _out("version_a_similarity_matrix.csv")
    sim.to_csv(sim_path)
    print(f"  ✓ {sim_path}")

    # Relatório textual
    lines = [
        "=" * 70,
        "VERSÃO A — RELATÓRIO DE CLUSTERING (Embeddings Fixos)",
        "=" * 70,
        "",
        f"Datasets analisados  : {len(names)}",
        f"Dimensão embedding   : {embeddings.shape[1]} ({STAT_DIM} estat. + {SEMANTIC_DIM} semântico)",
        f"Silhouette Hierárq.  : {cluster_results['silhouette']['hierarchical']:.4f}",
        f"Silhouette DBSCAN    : {cluster_results['silhouette']['dbscan']:.4f}",
        "",
    ]

    if business_metrics:
        lines += [
            "COMPARAÇÃO COM TIME DE NEGÓCIOS:",
            f"  ARI : {business_metrics.get('ari', 'N/A'):.4f}  (1=perfeito, 0=aleatório)",
            f"  NMI : {business_metrics.get('nmi', 'N/A'):.4f}  (1=perfeito, 0=sem info)",
            "",
        ]

    for method, key in [("HIERÁRQUICO", 'hierarchical'), ("DBSCAN", 'dbscan')]:
        lines.append(f"CLUSTERS — {method}:")
        for lbl in sorted(np.unique(cluster_results[key])):
            members = [n for n, l in zip(names, cluster_results[key]) if l == lbl]
            tag = "Outliers" if lbl == -1 else f"Cluster {lbl}"
            lines.append(f"  {tag}: {', '.join(members)}")
        lines.append("")

    txt_path = _out("version_a_report.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  ✓ {txt_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    _ensure_dirs()
    print("\n" + "=" * 70)
    print("VERSÃO A — EMBEDDINGS FIXOS + DISTÂNCIA DIRETA")
    print("=" * 70)

    # 1. Carrega
    print("\n📂 Carregando datasets...")
    datasets, names, col_names_list = load_datasets(DATA_DIR)
    if not datasets:
        print("Nenhum dataset encontrado em", DATA_DIR)
        return

    # 2. Embeddings fixos
    embedder   = DatasetEmbedder(semantic_dim=SEMANTIC_DIM, stat_dim=STAT_DIM)
    embeddings = embedder.embed(datasets, names, col_names_list)

    # 3. Clustering
    cluster_results = run_clustering(embeddings, names)

    # 4. Comparação com negócios
    biz = compare_with_business(
        names,
        cluster_results['hierarchical'],
        BUSINESS_GROUPS,
        method_name="hierarchical"
    )

    # 5. Visualizações
    print("\n📊 Gerando visualizações...")
    plot_all(embeddings, names, cluster_results, BUSINESS_GROUPS)

    # 6. Relatórios
    print("\n💾 Salvando relatórios...")
    save_reports(names, embeddings, cluster_results, biz)

    print("\n" + "=" * 70)
    print("✅ VERSÃO A CONCLUÍDA")
    print(f"   Imagens    → {IMG_DIR}/")
    print(f"   Relatórios → {OUTPUT_DIR}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()