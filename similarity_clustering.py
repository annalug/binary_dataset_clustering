"""
Clustering baseado em similaridade de embeddings
"""

import os
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

IMG_DIR = "img"

def _ensure_img_dir():
    os.makedirs(IMG_DIR, exist_ok=True)


class SimilarityBasedClustering:
    """
    Clustering de datasets baseado em similaridade
    """

    def __init__(self, method: str = 'hierarchical', **kwargs):
        """
        Args:
            method: 'hierarchical', 'dbscan', 'affinity'
        """
        self.method = method
        self.kwargs = kwargs
        self.labels_ = None
        self.cluster_centers_ = None

    def fit_predict(self, embeddings: np.ndarray, names: List[str] = None) -> np.ndarray:
        """
        Aplica clustering nos embeddings

        Args:
            embeddings: Array (n_datasets, embedding_dim)
            names: Nomes dos datasets (opcional)

        Returns:
            Labels de cluster
        """
        if names is None:
            names = [f"Dataset_{i}" for i in range(len(embeddings))]

        print(f"\n🔍 Aplicando clustering ({self.method}):")
        print(f"   {len(embeddings)} datasets, dimensão: {embeddings.shape[1]}")

        if self.method == 'hierarchical':
            # Clustering hierárquico
            n_clusters = self.kwargs.get('n_clusters', 3)
            linkage_method = self.kwargs.get('linkage', 'ward')

            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage_method,
                metric='euclidean'
            )

        elif self.method == 'dbscan':
            # DBSCAN (baseado em densidade)
            eps = self.kwargs.get('eps', 0.3)
            min_samples = self.kwargs.get('min_samples', 2)

            clustering = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='euclidean'
            )

        elif self.method == 'affinity':
            # Affinity Propagation
            from sklearn.cluster import AffinityPropagation
            damping = self.kwargs.get('damping', 0.9)

            clustering = AffinityPropagation(
                damping=damping,
                random_state=42
            )

        # Aplica clustering
        self.labels_ = clustering.fit_predict(embeddings)

        # Calcula centros dos clusters
        self._calculate_cluster_centers(embeddings)

        # Exibe resultados
        self._display_clustering_results(names, embeddings)

        return self.labels_

    def _calculate_cluster_centers(self, embeddings: np.ndarray):
        """Calcula centros dos clusters"""
        unique_labels = np.unique(self.labels_)
        self.cluster_centers_ = []

        for label in unique_labels:
            if label == -1:  # Noise no DBSCAN
                continue
            mask = self.labels_ == label
            cluster_center = np.mean(embeddings[mask], axis=0)
            self.cluster_centers_.append(cluster_center)

        self.cluster_centers_ = np.array(self.cluster_centers_)

    def _display_clustering_results(self, names: List[str], embeddings: np.ndarray):
        """Exibe resultados do clustering"""
        unique_labels = np.unique(self.labels_)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(self.labels_ == -1) if -1 in self.labels_ else 0

        print(f"\n📊 Resultados do clustering:")
        print(f"   Número de clusters: {n_clusters}")
        print(f"   Pontos de ruído: {n_noise}")

        # Distribuição por cluster
        print(f"\n📈 Distribuição:")
        for label in sorted(unique_labels):
            count = np.sum(self.labels_ == label)
            if label == -1:
                print(f"   Ruído: {count} datasets")
            else:
                datasets_in_cluster = [names[i] for i in range(len(names)) if self.labels_[i] == label]
                print(f"   Cluster {label}: {count} datasets")
                if len(datasets_in_cluster) <= 10:  # Mostra nomes se poucos datasets
                    print(f"     {', '.join(datasets_in_cluster)}")

        # Coesão dos clusters
        if n_clusters > 0 and len(self.cluster_centers_) > 0:
            print(f"\n🎯 Coesão dos clusters (distância média intra-cluster):")
            for label in range(n_clusters):
                mask = self.labels_ == label
                if np.sum(mask) > 0:
                    cluster_points = embeddings[mask]
                    center = self.cluster_centers_[label]
                    distances = np.linalg.norm(cluster_points - center, axis=1)
                    avg_distance = np.mean(distances)
                    print(f"   Cluster {label}: {avg_distance:.4f}")

    def plot_clusters(self, embeddings: np.ndarray, names: List[str], save: bool = True):
        """Visualiza clusters (redução para 2D). Salva em img/ se save=True."""
        _ensure_img_dir()
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        # Reduz para 2D usando t-SNE
        if embeddings.shape[1] > 2:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            embeddings_2d = embeddings

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Gráfico 1: Clusters
        scatter = axes[0].scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=self.labels_, cmap='tab20', s=100, alpha=0.7
        )

        # Adiciona nomes próximos aos pontos
        for i, name in enumerate(names):
            axes[0].annotate(
                name,
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=8, alpha=0.7
            )

        axes[0].set_title(f'Clusters de Datasets ({self.method})')
        axes[0].set_xlabel('Dimensão 1')
        axes[0].set_ylabel('Dimensão 2')
        axes[0].grid(True, alpha=0.3)

        # Legenda de clusters
        unique_labels = np.unique(self.labels_)
        legend_labels = [f'Cluster {label}' if label != -1 else 'Ruído' for label in unique_labels]
        axes[0].legend(handles=scatter.legend_elements()[0], labels=legend_labels)

        # Gráfico 2: Matriz de similaridade
        if len(names) <= 20:  # Só mostra se poucos datasets
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embeddings)

            im = axes[1].imshow(similarity_matrix, cmap='viridis', aspect='auto')
            axes[1].set_title('Matriz de Similaridade')
            axes[1].set_xticks(range(len(names)))
            axes[1].set_yticks(range(len(names)))
            axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            axes[1].set_yticklabels(names, fontsize=8)
            plt.colorbar(im, ax=axes[1])
        else:
            axes[1].axis('off')
            axes[1].text(0.5, 0.5, 'Matriz de similaridade\nnão mostrada (muitos datasets)',
                         ha='center', va='center', transform=axes[1].transAxes)

        plt.tight_layout()
        if save:
            out = os.path.join(IMG_DIR, f"similarity_clusters_{self.method}.jpg")
            plt.savefig(out, dpi=150, bbox_inches='tight')
            print(f"  ✓ Gráfico salvo em: {out}")
        plt.show()


def create_clusters_from_ranking(
        embeddings: np.ndarray,
        names: List[str],
        threshold: float = 0.7,
        min_cluster_size: int = 2
) -> Dict[int, List[str]]:
    """
    Cria clusters baseado em ranking (abordagem simples por threshold)

    Args:
        embeddings: Embeddings dos datasets
        names: Nomes dos datasets
        threshold: Limiar de similaridade para formar cluster
        min_cluster_size: Tamanho mínimo do cluster

    Returns:
        Dicionário {cluster_id: [nomes dos datasets]}
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Calcula matriz de similaridade
    similarity_matrix = cosine_similarity(embeddings)

    clusters = {}
    assigned = set()
    cluster_id = 0

    # Ordena datasets pela "centralidade" (similaridade média)
    centrality_scores = np.mean(similarity_matrix, axis=1)
    sorted_indices = np.argsort(centrality_scores)[::-1]

    for i in sorted_indices:
        if i in assigned:
            continue

        # Encontra datasets similares
        similar_indices = np.where(similarity_matrix[i] >= threshold)[0]
        similar_indices = [idx for idx in similar_indices if idx not in assigned]

        # Forma cluster se atingir tamanho mínimo
        if len(similar_indices) >= min_cluster_size:
            cluster_names = [names[idx] for idx in similar_indices]
            clusters[cluster_id] = cluster_names

            # Marca como atribuído
            assigned.update(similar_indices)
            cluster_id += 1

    # Datasets não agrupados (outliers)
    unassigned = [names[idx] for idx in range(len(names)) if idx not in assigned]
    if unassigned:
        clusters[-1] = unassigned  # -1 para outliers

    return clusters


def analyze_cluster_stability(
        embeddings: np.ndarray,
        names: List[str],
        n_trials: int = 10
):
    """
    Analisa estabilidade dos clusters com diferentes métodos
    """
    methods = ['hierarchical', 'dbscan']
    results = {}

    for method in methods:
        print(f"\n🧪 Testando estabilidade do método: {method}")

        all_labels = []
        for trial in range(n_trials):
            # Varia parâmetros ligeiramente
            if method == 'hierarchical':
                kwargs = {'n_clusters': np.random.randint(2, 5)}
            else:
                kwargs = {'eps': np.random.uniform(0.2, 0.4)}

            clusterer = SimilarityBasedClustering(method=method, **kwargs)
            labels = clusterer.fit_predict(embeddings, names)
            all_labels.append(labels)

        # Calcula consistência
        if n_trials > 1:
            from sklearn.metrics import adjusted_rand_score
            consistency_scores = []
            for i in range(n_trials):
                for j in range(i + 1, n_trials):
                    score = adjusted_rand_score(all_labels[i], all_labels[j])
                    consistency_scores.append(score)

            avg_consistency = np.mean(consistency_scores)
            print(f"   Consistência média: {avg_consistency:.4f}")

        results[method] = all_labels

    return results


def main():
    """Exemplo de uso"""
    print("\n" + "=" * 70)
    print("CLUSTERING BASEADO EM SIMILARIDADE")
    print("=" * 70)

    # Exemplo com dados simulados
    np.random.seed(42)
    n_datasets = 15
    embedding_dim = 128

    # Simula 3 clusters reais
    cluster_centers = [
        np.random.randn(embedding_dim) * 0.5 + 1.0,
        np.random.randn(embedding_dim) * 0.5 + 2.5,
        np.random.randn(embedding_dim) * 0.5 + 0.0
    ]

    embeddings = []
    names = []
    true_labels = []

    for i, center in enumerate(cluster_centers):
        n_in_cluster = n_datasets // 3
        for j in range(n_in_cluster):
            # Adiciona ruído ao centro do cluster
            point = center + np.random.randn(embedding_dim) * 0.1
            embeddings.append(point)
            names.append(f"Dataset_C{i}_{j}")
            true_labels.append(i)

    embeddings = np.array(embeddings)

    print(f"\n📊 Dados simulados:")
    print(f"   {len(embeddings)} datasets")
    print(f"   {len(np.unique(true_labels))} clusters reais")

    # Método 1: Clustering hierárquico
    print("\n" + "-" * 70)
    print("MÉTODO 1: CLUSTERING HIERÁRQUICO")
    print("-" * 70)

    clusterer_hier = SimilarityBasedClustering(
        method='hierarchical',
        n_clusters=3,
        linkage='ward'
    )

    labels_hier = clusterer_hier.fit_predict(embeddings, names)
    clusterer_hier.plot_clusters(embeddings, names)

    # Método 2: DBSCAN
    print("\n" + "-" * 70)
    print("MÉTODO 2: DBSCAN (BASEADO EM DENSIDADE)")
    print("-" * 70)

    clusterer_dbscan = SimilarityBasedClustering(
        method='dbscan',
        eps=0.3,
        min_samples=2
    )

    labels_dbscan = clusterer_dbscan.fit_predict(embeddings, names)
    clusterer_dbscan.plot_clusters(embeddings, names)

    # Método 3: Baseado em ranking (threshold)
    print("\n" + "-" * 70)
    print("MÉTODO 3: CLUSTERING POR THRESHOLD (SIMPLES)")
    print("-" * 70)

    clusters_threshold = create_clusters_from_ranking(
        embeddings, names, threshold=0.8, min_cluster_size=2
    )

    print("\n📊 Clusters encontrados:")
    for cluster_id, cluster_names in clusters_threshold.items():
        if cluster_id == -1:
            print(f"   Outliers ({len(cluster_names)}): {', '.join(cluster_names)}")
        else:
            print(f"   Cluster {cluster_id} ({len(cluster_names)}): {', '.join(cluster_names)}")

    # Análise de estabilidade
    print("\n" + "-" * 70)
    print("ANÁLISE DE ESTABILIDADE")
    print("-" * 70)

    analyze_cluster_stability(embeddings, names, n_trials=3)


if __name__ == "__main__":
    main()