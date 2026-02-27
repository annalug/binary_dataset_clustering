"""
config.py — Configuração central do projeto

Todos os hiperparâmetros, caminhos e constantes ficam aqui.
Importe com:  from config import CFG
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import os


# ============================================================================
# CAMINHOS
# ============================================================================

@dataclass
class Paths:
    data:    str = "./data"
    output:  str = "./output"
    img:     str = "./img"
    models:  str = "./models"

    def ensure_all(self):
        """Cria todos os diretórios se não existirem."""
        for path in (self.output, self.img, self.models):
            os.makedirs(path, exist_ok=True)

    def output_file(self, filename: str) -> str:
        return os.path.join(self.output, filename)

    def img_file(self, filename: str) -> str:
        return os.path.join(self.img, filename)

    def model_file(self, filename: str) -> str:
        return os.path.join(self.models, filename)


# ============================================================================
# PADRONIZAÇÃO (DatasetStandardizer)
# ============================================================================

@dataclass
class StandardizerConfig:
    target_samples:    int   = 256    # Linhas após pad/subsample
    target_features:   int   = 100    # Colunas após PCA/truncamento
    use_pca:           bool  = True   # False → truncamento simples
    min_variance_ratio: float = 0.90  # Aviso se PCA capturar menos que isso
    random_state:      int   = 42


# ============================================================================
# REDE SIAMESA (SiameseNet)
# ============================================================================

@dataclass
class SiameseConfig:
    # Arquitetura
    architecture:   str   = 'default'   # 'light' | 'default' | 'deep'
    embedding_dim:  int   = 128         # Dimensão do vetor de saída do encoder
    input_shape:    Tuple = (256, 100, 1)  # Deve bater com StandardizerConfig

    # Treinamento
    learning_rate:  float = 1e-4
    epochs:         int   = 50
    batch_size:     int   = 32
    validation_split: float = 0.20

    # Callbacks
    early_stopping_patience: int   = 10
    reduce_lr_patience:      int   = 5
    reduce_lr_factor:        float = 0.5
    min_lr:                  float = 1e-7

    # Loss / Contrastiva
    margin:         float = 1.0   # Margem para contrastive loss

    # Serialização
    model_prefix:   str   = "siamese_model"


# ============================================================================
# GERADOR DE PARES (PairGenerator)
# ============================================================================

@dataclass
class PairConfig:
    # Labels de tipo de dataset (semântica de DATASET, não de malware/benigno)
    # Exemplo: 0=permissions, 1=api_calls, 2=network_features
    # Deve ser definido pelo usuário conforme os datasets carregados
    dataset_type_map: dict = field(default_factory=dict)
    # ^ chave: nome do dataset  →  valor: int (tipo de features)

    n_pairs:        int   = 1000
    balance_ratio:  float = 0.50   # 0.5 = 50% similares, 50% diferentes
    pairs_per_class: int  = 200    # Usado em create_balanced_pairs
    random_state:   int   = 42


# ============================================================================
# CLUSTERING
# ============================================================================

@dataclass
class ClusteringConfig:
    # Hierárquico
    method:          str   = 'hierarchical'   # 'hierarchical' | 'dbscan' | 'affinity'
    n_clusters:      int   = 3                # Usado quando method='hierarchical'
    linkage:         str   = 'ward'           # 'ward' | 'complete' | 'average'

    # DBSCAN
    dbscan_eps:      float = 0.3
    dbscan_min_samples: int = 2

    # Threshold (create_clusters_from_ranking)
    similarity_threshold: float = 0.70
    min_cluster_size:     int   = 2

    # Ranking
    ranking_metric:  str   = 'cosine'   # 'cosine' | 'euclidean'
    ranking_top_k:   int   = 10


# ============================================================================
# VISUALIZAÇÃO (plot_clusters)
# ============================================================================

@dataclass
class PlotConfig:
    dpi:            int   = 300
    fig_width:      int   = 20
    fig_height:     int   = 12
    colormap:       str   = 'tab20'
    tsne_max_iter:  int   = 1000
    tsne_perplexity_max: int = 30
    plot_prefix:    str   = "final_clusters"


# ============================================================================
# CONFIG GLOBAL (ponto de entrada único)
# ============================================================================

@dataclass
class Config:
    paths:       Paths            = field(default_factory=Paths)
    standardizer: StandardizerConfig = field(default_factory=StandardizerConfig)
    siamese:     SiameseConfig    = field(default_factory=SiameseConfig)
    pairs:       PairConfig       = field(default_factory=PairConfig)
    clustering:  ClusteringConfig = field(default_factory=ClusteringConfig)
    plot:        PlotConfig       = field(default_factory=PlotConfig)

    # CSV: nome da coluna de rótulo nos arquivos de dados
    class_column: str = "class"

    def __post_init__(self):
        # Garante que input_shape do siamese bate com o standardizer
        self.siamese.input_shape = (
            self.standardizer.target_samples,
            self.standardizer.target_features,
            1
        )


# Instância global — importe CFG nos outros módulos
CFG = Config()


# ============================================================================
# UTILITÁRIO: imprime resumo da config
# ============================================================================

def print_config(cfg: Config = CFG):
    sep = "=" * 70
    print(f"\n{sep}")
    print("CONFIGURAÇÃO DO PROJETO")
    print(sep)

    sections = {
        "Caminhos": cfg.paths,
        "Padronização": cfg.standardizer,
        "Rede Siamesa": cfg.siamese,
        "Pares": cfg.pairs,
        "Clustering": cfg.clustering,
        "Visualização": cfg.plot,
    }

    for title, section in sections.items():
        print(f"\n  [{title}]")
        for key, val in vars(section).items():
            if isinstance(val, dict) and not val:
                val = "{}"
            print(f"    {key:30s}: {val}")

    print(f"\n  [Geral]")
    print(f"    {'class_column':30s}: {cfg.class_column}")
    print(f"\n{sep}\n")


if __name__ == "__main__":
    print_config()