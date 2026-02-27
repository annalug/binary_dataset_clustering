import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


class InstanceDistributionClustering:
    """
    Clusteriza datasets baseando-se na distribuição das suas instâncias individuais.
    """

    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.dataset_signatures = None
        self.labels = None

    def fit_predict(self, datasets_std: np.ndarray):
        """
        Args:
            datasets_std: Shape (n_datasets, n_samples, n_features, 1)
        """
        n_datasets = datasets_std.shape[0]
        n_features = datasets_std.shape[2]

        print(f"📦 Extraindo assinaturas de {n_datasets} datasets...")

        # 1. Gerar a "Assinatura" de cada dataset (Média das instâncias no espaço de features)
        # Em vez de um embedding da rede, usamos a estatística das linhas.
        signatures = []
        for i in range(n_datasets):
            # Flatten das instâncias (remover canal 1)
            instances = datasets_std[i].reshape(datasets_std.shape[1], -1)

            # Calculamos o centroide (média) das instâncias desse dataset
            # Isso representa o "ponto central" do comportamento do malware nesse dataset
            centroid = np.mean(instances, axis=0)
            signatures.append(centroid)

        self.dataset_signatures = np.array(signatures)

        # 2. Aplicar Clustering Hierárquico sobre os centroides
        clusterer = AgglomerativeClustering(n_clusters=self.n_clusters, metric='euclidean', linkage='ward')
        self.labels = clusterer.fit_predict(self.dataset_signatures)

        score = silhouette_score(self.dataset_signatures, self.labels)
        print(f"✅ Clustering concluído. Silhouette Score: {score:.4f}")

        return self.labels

    def plot_cluster_summary(self, names):
        """Exibe quais datasets ficaram em cada cluster"""
        for i in range(self.n_clusters):
            print(f"\n🔹 Cluster {i}:")
            indices = np.where(self.labels == i)[0]
            for idx in indices:
                print(f"  - {names[idx]}")

# --- Exemplo de uso no seu pipeline principal ---
# clusterer = InstanceDistributionClustering(n_clusters=3)
# labels = clusterer.fit_predict(datasets_std)
# clusterer.plot_cluster_summary(names)