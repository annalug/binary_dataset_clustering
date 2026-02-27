"""
version_b_autoencoder.py — Versão B: Embeddings fixos → Autoencoder → espaço latente refinado

Fluxo:
  1. Mesmo embedding fixo da Versão A (estatísticas + semântica)
  2. Autoencoder leve comprime (100,) → (32,) e reconstrói
     → o espaço latente (32,) captura estrutura essencial
  3. A rede siamesa entra como COMPARADOR:
     recebe pares de embeddings latentes e calcula distância
  4. Clustering + comparação com negócios
  5. Visualizações com lado a lado Versão A vs Versão B

Não usa labels. Treinamento 100% não supervisionado.
Requer: tensorflow, scikit-learn, scipy, matplotlib, seaborn.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from cluster_selection import find_best_k, plot_k_selection
from typing import List, Dict, Optional

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, callbacks

# Reutiliza o embedder da Versão A
from version_a_direct import DatasetEmbedder, load_datasets, compare_with_business

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

IMG_DIR    = "img"
OUTPUT_DIR = "output"
DATA_DIR   = "./data"
CLASS_COL  = "class"

# Embedding fixo (mesmo da versão A)
SEMANTIC_DIM = 50
STAT_DIM     = 50
FIXED_DIM    = STAT_DIM + SEMANTIC_DIM   # 100

# Autoencoder
LATENT_DIM   = 32     # dimensão do espaço latente
HIDDEN_DIMS  = [64, 48]  # camadas intermediárias encoder

# Treinamento do autoencoder
AE_EPOCHS    = 200
AE_BATCH     = 4      # pequeno porque temos poucos datasets
AE_LR        = 1e-3
AE_PATIENCE  = 30     # early stopping

# Clustering — k escolhido automaticamente
K_MIN        = 2
K_MAX        = None   # None = min(n_datasets-1, 5)
DBSCAN_EPS   = 1.0
DBSCAN_MIN   = 2

# Agrupamento do time de negócios — preencha com seus dados
BUSINESS_GROUPS: Dict[str, int] = {
    # Exemplo:
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


def _img(f):  return os.path.join(IMG_DIR, f)
def _out(f):  return os.path.join(OUTPUT_DIR, f)


# ============================================================================
# AUTOENCODER
# ============================================================================

class DatasetAutoencoder:
    """
    Autoencoder totalmente conectado (MLP) para comprimir embeddings fixos.

    Arquitetura:
      Encoder: (100,) → [64] → [48] → (32,)  ← espaço latente
      Decoder: (32,)  → [48] → [64] → (100,) ← reconstrução

    A loss é puramente de reconstrução (MSE).
    Nenhum label necessário.

    Após o treino, o encoder extrai o embedding latente de cada dataset.
    A rede siamesa usa esses embeddings para calcular distâncias.
    """

    def __init__(
        self,
        input_dim:   int = FIXED_DIM,
        latent_dim:  int = LATENT_DIM,
        hidden_dims: List[int] = None,
        lr:          float = AE_LR,
    ):
        self.input_dim   = input_dim
        self.latent_dim  = latent_dim
        self.hidden_dims = hidden_dims or HIDDEN_DIMS
        self.lr          = lr

        self.encoder:     Optional[Model] = None
        self.autoencoder: Optional[Model] = None
        self.history      = None

        self._build()

    def _build(self):
        sep = "=" * 70
        print(f"\n{sep}")
        print("CONSTRUINDO AUTOENCODER")
        print(sep)

        # --- Encoder ---
        inp = layers.Input(shape=(self.input_dim,), name='encoder_input')
        x   = inp

        for i, dim in enumerate(self.hidden_dims):
            x = layers.Dense(dim, name=f'enc_{i}')(x)
            x = layers.BatchNormalization(name=f'enc_bn_{i}')(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.1)(x)

        latent = layers.Dense(self.latent_dim, name='latent')(x)
        # L2-normalização: força os embeddings latentes na hiperesfera unitária
        # → distâncias cosine e euclideanas equivalentes → clustering mais estável
        latent_norm = layers.UnitNormalization(axis=-1, name='latent_l2')(latent)

        self.encoder = Model(inputs=inp, outputs=latent_norm, name='encoder')

        # --- Decoder ---
        dec_inp = layers.Input(shape=(self.latent_dim,), name='decoder_input')
        y = dec_inp

        for i, dim in enumerate(reversed(self.hidden_dims)):
            y = layers.Dense(dim, name=f'dec_{i}')(y)
            y = layers.BatchNormalization(name=f'dec_bn_{i}')(y)
            y = layers.Activation('relu')(y)

        # Saída sem ativação (valores contínuos, já normalizados pelo StandardScaler)
        reconstruction = layers.Dense(self.input_dim, name='reconstruction')(y)
        decoder = Model(inputs=dec_inp, outputs=reconstruction, name='decoder')

        # --- Autoencoder completo ---
        ae_input = layers.Input(shape=(self.input_dim,), name='ae_input')
        encoded  = self.encoder(ae_input)
        decoded  = decoder(encoded)

        self.autoencoder = Model(inputs=ae_input, outputs=decoded, name='autoencoder')
        self.autoencoder.compile(
            optimizer=optimizers.Adam(learning_rate=self.lr),
            loss='mse',
            metrics=['mae']
        )

        total_params = self.autoencoder.count_params()
        print(f"\n  input_dim  : {self.input_dim}")
        print(f"  hidden     : {self.hidden_dims}")
        print(f"  latent_dim : {self.latent_dim}  (L2-normalizado)")
        print(f"  params     : {total_params:,}")
        print(sep + "\n")

    def train(self, X: np.ndarray, epochs: int = AE_EPOCHS, batch_size: int = AE_BATCH):
        """
        Treina o autoencoder.

        Com poucos datasets (ex: 10), usamos todos como treino.
        Validation é feito com leave-one-out implícito pelo early stopping.
        """
        print(f"\n{'='*70}")
        print("TREINAMENTO DO AUTOENCODER")
        print(f"{'='*70}")
        print(f"  Amostras : {X.shape[0]}")
        print(f"  Epochs   : {epochs}  |  batch: {batch_size}")

        # Com < 5 amostras, não faz sentido validation_split
        val_split = 0.2 if X.shape[0] >= 5 else 0.0

        cb = [
            callbacks.EarlyStopping(
                monitor='val_loss' if val_split > 0 else 'loss',
                patience=AE_PATIENCE,
                restore_best_weights=True,
                verbose=0
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if val_split > 0 else 'loss',
                factor=0.5, patience=15, min_lr=1e-6, verbose=0
            ),
        ]

        self.history = self.autoencoder.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=cb,
            verbose=0
        )

        final_loss = self.history.history['loss'][-1]
        n_epochs   = len(self.history.history['loss'])
        print(f"  Convergiu em {n_epochs} épocas  |  loss final: {final_loss:.6f}")

        return self.history

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Extrai embeddings latentes (L2-normalizados)."""
        return self.encoder.predict(X, verbose=0)

    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """MSE de reconstrução por amostra — útil para detectar outliers."""
        X_rec = self.autoencoder.predict(X, verbose=0)
        return np.mean((X - X_rec) ** 2, axis=1)

    def save(self, prefix: str = "models/autoencoder"):
        os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
        self.encoder.save(f"{prefix}_encoder.keras")
        self.autoencoder.save(f"{prefix}_full.keras")
        print(f"  ✓ Salvo em {prefix}_*.keras")

    def load(self, prefix: str = "models/autoencoder"):
        self.encoder     = keras.models.load_model(f"{prefix}_encoder.keras")
        self.autoencoder = keras.models.load_model(f"{prefix}_full.keras")
        print(f"  ✓ Carregado de {prefix}_*.keras")


# ============================================================================
# REDE SIAMESA (comparador de embeddings latentes)
# ============================================================================

class SiameseDistanceLayer(layers.Layer):
    """
    Calcula distância euclidiana entre dois embeddings latentes.
    Usa subclasse de Layer (não Lambda) para serialização correta.
    """
    def call(self, inputs):
        a, b   = inputs
        sq     = tf.math.squared_difference(a, b)
        summed = tf.reduce_sum(sq, axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(summed, 1e-12))

    def get_config(self):
        return super().get_config()


def build_siamese_comparator(latent_dim: int = LATENT_DIM) -> Model:
    """
    Rede siamesa que compara dois embeddings latentes.
    Recebe pares (emb_a, emb_b) e retorna distância euclidiana.

    Nota: este comparador NÃO é treinado — ele apenas calcula a distância
    entre os embeddings já aprendidos pelo autoencoder.
    Toda a aprendizagem de representação acontece no autoencoder.
    """
    inp_a = layers.Input(shape=(latent_dim,), name='emb_a')
    inp_b = layers.Input(shape=(latent_dim,), name='emb_b')
    dist  = SiameseDistanceLayer(name='distance')([inp_a, inp_b])
    return Model(inputs=[inp_a, inp_b], outputs=dist, name='siamese_comparator')


# ============================================================================
# CLUSTERING
# ============================================================================

def run_clustering(latent_embs: np.ndarray, names: List[str]) -> Dict:
    """
    Clustering no espaço latente com seleção automática de k.
    Nenhum k pré-definido.
    """
    print(f"\n{'='*70}")
    print("CLUSTERING NO ESPAÇO LATENTE")
    print(f"{'='*70}")

    scaler = StandardScaler()
    X      = scaler.fit_transform(latent_embs)

    def _sil(emb, lbls):
        valid = lbls != -1
        if np.sum(valid) > 1 and len(np.unique(lbls[valid])) > 1:
            try:
                return silhouette_score(emb[valid], lbls[valid])
            except Exception:
                pass
        return float('nan')

    # Seleção automática de k
    k_results   = find_best_k(X, names, k_min=K_MIN, k_max=K_MAX)
    best_k      = k_results['best_k']
    labels_hier = k_results['labels'][best_k]
    sil_hier    = _sil(X, labels_hier)

    # DBSCAN segunda opinião
    db        = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN)
    labels_db = db.fit_predict(X)
    sil_db    = _sil(X, labels_db)
    n_db      = len(set(labels_db) - {-1})

    print(f"\n  DBSCAN (eps={DBSCAN_EPS}) → {n_db} clusters, Silhouette: {sil_db:.4f}")
    for lbl in sorted(set(labels_db)):
        members = [n for n, l in zip(names, labels_db) if l == lbl]
        tag = "Outliers" if lbl == -1 else f"Cluster {lbl}"
        print(f"    {tag}: {', '.join(members)}")

    return {
        'hierarchical': labels_hier,
        'dbscan':        labels_db,
        'X_scaled':      X,
        'best_k':        best_k,
        'k_results':     k_results,
        'silhouette':    {'hierarchical': sil_hier, 'dbscan': sil_db},
    }


# ============================================================================
# VISUALIZAÇÕES
# ============================================================================

def plot_all(
    fixed_embs:      np.ndarray,
    latent_embs:     np.ndarray,
    names:           List[str],
    cluster_results: Dict,
    ae_history,
    business_groups: Dict,
):
    _ensure_dirs()
    X_scaled = cluster_results['X_scaled']
    cmap     = plt.cm.tab10

    # PCA 2D nos embeddings latentes
    pca  = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    best_k = cluster_results['best_k']
    var  = pca.explained_variance_ratio_

    # ---- Fig 1: curva de treino ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Versão B — Treinamento do Autoencoder", fontsize=13, fontweight='bold')

    h    = ae_history.history
    ep   = range(1, len(h['loss']) + 1)
    axes[0].plot(ep, h['loss'], label='Treino')
    if 'val_loss' in h:
        axes[0].plot(ep, h['val_loss'], label='Validação')
    axes[0].set_title("Loss (MSE)")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, h['mae'], label='Treino')
    if 'val_mae' in h:
        axes[1].plot(ep, h['val_mae'], label='Validação')
    axes[1].set_title("MAE de Reconstrução")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    p0 = _img("version_b_training.jpg")
    plt.savefig(p0, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {p0}")

    # ---- Fig extra: seleção de k ----
    plot_k_selection(
        cluster_results['k_results'],
        title=f"Versão B — Seleção do Número de Clusters (best k={cluster_results['best_k']})",
        output_path=_img("version_b_k_selection.jpg")
    )

    # ---- Fig 2: clusters ----
    n_panels = 4 if business_groups else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 7))
    fig.suptitle("Versão B — Espaço Latente (Autoencoder + Siamese)", fontsize=13, fontweight='bold')

    # PCA nos embeddings FIXOS (para comparação direta)
    pca_fixed  = PCA(n_components=2, random_state=42)
    X_fixed_2d = pca_fixed.fit_transform(StandardScaler().fit_transform(fixed_embs))
    var_fixed  = pca_fixed.explained_variance_ratio_

    configs = [
        (axes[0], X_fixed_2d,  var_fixed, cluster_results['hierarchical'],
         f"Fixo — Hierárquico k={best_k} (auto)\nSilhouette: {cluster_results['silhouette']['hierarchical']:.3f}"),
        (axes[1], X_2d,        var,       cluster_results['hierarchical'],
         f"Latente — Hierárquico k={best_k} (auto)\nSilhouette: {cluster_results['silhouette']['hierarchical']:.3f}"),
        (axes[2], X_2d,        var,       cluster_results['dbscan'],
         f"Latente — DBSCAN (eps={DBSCAN_EPS})\nSilhouette: {cluster_results['silhouette']['dbscan']:.3f}"),
    ]

    for ax, coords, v_ex, labels, title in configs:
        unique = np.unique(labels)
        cols   = cmap(np.linspace(0, 0.9, len(unique)))
        for lbl, col in zip(unique, cols):
            mask = labels == lbl
            tag  = "Outliers" if lbl == -1 else f"Cluster {lbl}"
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[col], s=160, label=tag,
                       edgecolors='white', linewidth=1.2, zorder=3)
            for idx in np.where(mask)[0]:
                ax.annotate(names[idx], (coords[idx, 0], coords[idx, 1]),
                            fontsize=7, ha='center', va='bottom',
                            xytext=(0, 6), textcoords='offset points')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"PC1 ({v_ex[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({v_ex[1]*100:.1f}%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Painel de negócios
    if business_groups and n_panels == 4:
        ax4 = axes[3]
        bus_labels = np.array([business_groups.get(n, -1) for n in names])
        unique_bus = np.unique(bus_labels)
        cols_bus   = cmap(np.linspace(0, 0.9, len(unique_bus)))
        for lbl, col in zip(unique_bus, cols_bus):
            mask = bus_labels == lbl
            tag  = "Sem grupo" if lbl == -1 else f"Negócios {lbl}"
            ax4.scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=[col], s=160, label=tag, marker='D',
                        edgecolors='white', linewidth=1.2, zorder=3)
            for idx in np.where(mask)[0]:
                ax4.annotate(names[idx], (X_2d[idx, 0], X_2d[idx, 1]),
                             fontsize=7, ha='center', va='bottom',
                             xytext=(0, 6), textcoords='offset points')
        ax4.set_title("Agrupamento do Time de Negócios", fontsize=10)
        ax4.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
        ax4.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    p1 = _img("version_b_clusters.jpg")
    plt.savefig(p1, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {p1}")

    # ---- Fig 3: heatmaps fixo vs latente ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Versão B — Matriz de Similaridade: Fixo vs Latente", fontsize=13, fontweight='bold')

    for ax, emb, title in [
        (axes[0], fixed_embs,  "Embedding Fixo (100 dim)"),
        (axes[1], latent_embs, f"Embedding Latente ({LATENT_DIM} dim)"),
    ]:
        sim = cosine_similarity(emb)
        im  = ax.imshow(sim, cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    p2 = _img("version_b_similarity.jpg")
    plt.savefig(p2, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {p2}")

    # ---- Fig 4: dendrograma no espaço latente ----
    Z = linkage(X_scaled, method='ward')
    plt.figure(figsize=(14, 6))
    dendrogram(Z, labels=names, leaf_rotation=45, leaf_font_size=9)
    plt.title("Versão B — Dendrograma (espaço latente)", fontsize=13, fontweight='bold')
    plt.xlabel("Dataset")
    plt.ylabel("Distância (espaço latente)")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    p3 = _img("version_b_dendrogram.jpg")
    plt.savefig(p3, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {p3}")

    # ---- Fig 5: distâncias siamesas (matriz de distância latente) ----
    comparator  = build_siamese_comparator(LATENT_DIM)
    n           = len(latent_embs)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            a = latent_embs[i:i+1]
            b = latent_embs[j:j+1]
            dist_matrix[i, j] = float(
                comparator.predict([a, b], verbose=0)[0][0]
            )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dist_matrix, cmap='plasma_r')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title(
        "Versão B — Distâncias Siamesas (espaço latente)\n"
        "Azul escuro = muito similares | Amarelo = muito distantes",
        fontsize=11, fontweight='bold'
    )
    plt.colorbar(im, ax=ax, label='Distância euclidiana latente')
    plt.tight_layout()
    p4 = _img("version_b_siamese_distances.jpg")
    plt.savefig(p4, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {p4}")

    return [p0, p1, p2, p3, p4]


# ============================================================================
# RELATÓRIOS
# ============================================================================

def save_reports(
    names:            List[str],
    fixed_embs:       np.ndarray,
    latent_embs:      np.ndarray,
    cluster_results:  Dict,
    business_metrics: Dict,
    ae:               DatasetAutoencoder,
):
    _ensure_dirs()

    # CSV de clusters
    df = pd.DataFrame({
        'dataset':        names,
        'cluster_hier':   cluster_results['hierarchical'],
        'cluster_dbscan': cluster_results['dbscan'],
        'recon_error':    ae.get_reconstruction_error(
            StandardScaler().fit_transform(
                np.vstack([fixed_embs])   # mesma escala do treino
            )
        ),
    })
    if business_metrics.get('business_labels') is not None:
        df['business_group'] = business_metrics['business_labels']

    csv_path = _out("version_b_clusters.csv")
    df.to_csv(csv_path, index=False)
    print(f"  ✓ {csv_path}")

    # Matriz de similaridade latente
    sim = pd.DataFrame(
        cosine_similarity(latent_embs),
        index=names, columns=names
    )
    sim.to_csv(_out("version_b_similarity_latent.csv"))
    print(f"  ✓ {_out('version_b_similarity_latent.csv')}")

    # Relatório textual
    lines = [
        "=" * 70,
        "VERSÃO B — RELATÓRIO (Autoencoder + Siamese)",
        "=" * 70,
        "",
        f"Datasets           : {len(names)}",
        f"Embedding fixo     : {fixed_embs.shape[1]} dim ({STAT_DIM} estat. + {SEMANTIC_DIM} sem.)",
        f"Espaço latente     : {LATENT_DIM} dim (L2-normalizado)",
        f"Silhouette Hierárq.: {cluster_results['silhouette']['hierarchical']:.4f}",
        f"Silhouette DBSCAN  : {cluster_results['silhouette']['dbscan']:.4f}",
        "",
    ]

    if business_metrics:
        lines += [
            "COMPARAÇÃO COM TIME DE NEGÓCIOS:",
            f"  ARI : {business_metrics.get('ari', 'N/A'):.4f}",
            f"  NMI : {business_metrics.get('nmi', 'N/A'):.4f}",
            "",
        ]

    lines.append("ERRO DE RECONSTRUÇÃO POR DATASET:")
    rec_errs = ae.get_reconstruction_error(
        StandardScaler().fit_transform(fixed_embs)
    )
    for name, err in sorted(zip(names, rec_errs), key=lambda x: -x[1]):
        lines.append(f"  {name:45s}: {err:.6f}")
    lines.append("")

    for method, key in [("HIERÁRQUICO", 'hierarchical'), ("DBSCAN", 'dbscan')]:
        lines.append(f"CLUSTERS — {method}:")
        for lbl in sorted(np.unique(cluster_results[key])):
            members = [n for n, l in zip(names, cluster_results[key]) if l == lbl]
            tag = "Outliers" if lbl == -1 else f"Cluster {lbl}"
            lines.append(f"  {tag}: {', '.join(members)}")
        lines.append("")

    txt_path = _out("version_b_report.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  ✓ {txt_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    _ensure_dirs()
    print("\n" + "=" * 70)
    print("VERSÃO B — AUTOENCODER + REDE SIAMESA")
    print("=" * 70)

    # 1. Carrega datasets
    print("\n📂 Carregando datasets...")
    datasets, names, col_names_list = load_datasets(DATA_DIR)
    if not datasets:
        print("Nenhum dataset encontrado em", DATA_DIR)
        return

    # 2. Embeddings fixos (mesmo da Versão A)
    print("\n🔢 Gerando embeddings fixos...")
    embedder   = DatasetEmbedder(semantic_dim=SEMANTIC_DIM, stat_dim=STAT_DIM)
    fixed_embs = embedder.embed(datasets, names, col_names_list)

    # Normaliza para o autoencoder
    scaler     = StandardScaler()
    fixed_norm = scaler.fit_transform(fixed_embs).astype(np.float32)

    # 3. Autoencoder — comprime (100,) → (32,) sem nenhum label
    print("\n🧠 Treinando autoencoder...")
    ae = DatasetAutoencoder(
        input_dim=FIXED_DIM,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        lr=AE_LR,
    )
    ae.train(fixed_norm, epochs=AE_EPOCHS, batch_size=AE_BATCH)
    ae.save()

    # 4. Embeddings latentes
    latent_embs = ae.get_embeddings(fixed_norm)
    print(f"\n  Latent embeddings: {latent_embs.shape}")
    print(f"  Norms (esperado ≈ 1.0): {np.linalg.norm(latent_embs, axis=1).round(4)}")

    # 5. Clustering no espaço latente
    cluster_results = run_clustering(latent_embs, names)

    # 6. Comparação com negócios
    biz = compare_with_business(
        names,
        cluster_results['hierarchical'],
        BUSINESS_GROUPS,
        method_name="hierarchical (latente)"
    )

    # 7. Visualizações
    print("\n📊 Gerando visualizações...")
    plot_all(fixed_embs, latent_embs, names, cluster_results, ae.history, BUSINESS_GROUPS)

    # 8. Relatórios
    print("\n💾 Salvando relatórios...")
    save_reports(names, fixed_embs, latent_embs, cluster_results, biz, ae)

    print("\n" + "=" * 70)
    print("✅ VERSÃO B CONCLUÍDA")
    print(f"   Imagens    → {IMG_DIR}/")
    print(f"   Relatórios → {OUTPUT_DIR}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()