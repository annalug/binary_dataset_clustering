"""
Implementação do Framework de Clusterização Automática Baseado em Otimização Binária
Inspirado no artigo: "Automatic Data Clustering Framework Using Nature-Inspired Binary Optimization Algorithms"
Autores: Behnaz Merikhi e M. R. Soleymani, IEEE Access 2021
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Callable
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# ESTRUTURAS DE DADOS E CONFIGURAÇÃO
# ============================================================================

@dataclass
class ClusteringConfig:
    """Configuração para o framework de clusterização"""
    K_min: int = 1
    K_max: int = None  # Será calculado como floor(sqrt(m))
    n_particles: int = 30
    max_iterations: int = 100
    merge_threshold: float = 0.9  # Limiar para fusão de clusters (0.9 do artigo)
    distance_metric: str = 'euclidean'
    binary_mode: bool = False  # Modo para dados binários
    optimizer_type: str = 'BPSO'  # BPSO, BBA, BGA, BDA


@dataclass
class ClusterResult:
    """Resultado da clusterização"""
    labels: np.ndarray
    n_clusters: int
    centroids: np.ndarray
    representatives: np.ndarray
    distortion_deviation: float
    intra_cluster_dist: float
    inter_cluster_dist: float
    db_index: float
    cost_history: List[float]


# ============================================================================
# OTIMIZADORES BINÁRIOS (ABSTRACTION)
# ============================================================================

class BinaryOptimizer(ABC):
    """Interface para otimizadores binários"""

    def __init__(self, n_particles: int, n_dimensions: int):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.particles = None
        self.velocities = None
        self.pbest_positions = None
        self.pbest_values = None
        self.gbest_position = None
        self.gbest_value = float('inf')
        self.initialize_population()

    def initialize_population(self):
        """Inicializa população com valores binários aleatórios"""
        self.particles = np.random.randint(0, 2,
                                           size=(self.n_particles, self.n_dimensions))
        self.velocities = np.random.uniform(-1, 1,
                                            size=(self.n_particles, self.n_dimensions))
        self.pbest_positions = self.particles.copy()
        self.pbest_values = np.full(self.n_particles, float('inf'))

    @abstractmethod
    def update_velocity(self, iteration: int, max_iterations: int):
        pass

    @abstractmethod
    def update_position(self):
        pass

    def optimize(self, objective_func: Callable, max_iterations: int) -> Tuple[np.ndarray, float]:
        """Executa o processo de otimização"""
        cost_history = []

        for iteration in range(max_iterations):
            # Avalia todas as partículas
            for i in range(self.n_particles):
                cost = objective_func(self.particles[i])

                # Atualiza melhor pessoal
                if cost < self.pbest_values[i]:
                    self.pbest_values[i] = cost
                    self.pbest_positions[i] = self.particles[i].copy()

                # Atualiza melhor global
                if cost < self.gbest_value:
                    self.gbest_value = cost
                    self.gbest_position = self.particles[i].copy()

            # Atualiza velocidades e posições
            self.update_velocity(iteration, max_iterations)
            self.update_position()

            cost_history.append(self.gbest_value)

        return self.gbest_position, self.gbest_value, cost_history


class BinaryPSO(BinaryOptimizer):
    """Binary Particle Swarm Optimization (BPSO)"""

    def __init__(self, n_particles: int, n_dimensions: int,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        super().__init__(n_particles, n_dimensions)
        self.w = w  # Inércia
        self.c1 = c1  # Coeficiente cognitivo
        self.c2 = c2  # Coeficiente social

    def update_velocity(self, iteration: int, max_iterations: int):
        """Atualiza velocidades usando equação do PSO"""
        r1 = np.random.rand(self.n_particles, self.n_dimensions)
        r2 = np.random.rand(self.n_particles, self.n_dimensions)

        cognitive = self.c1 * r1 * (self.pbest_positions - self.particles)
        social = self.c2 * r2 * (self.gbest_position - self.particles)

        self.velocities = (self.w * self.velocities +
                           cognitive + social)

        # Limita velocidades
        self.velocities = np.clip(self.velocities, -4, 4)

    def update_position(self):
        """Atualiza posições usando função sigmoide"""
        # Função sigmoide V-shaped
        sigmoid = 1 / (1 + np.exp(-self.velocities))
        rand_values = np.random.rand(self.n_particles, self.n_dimensions)

        # Decisão binária baseada na sigmoide
        new_positions = np.zeros_like(self.particles)
        new_positions[rand_values < sigmoid] = 1

        self.particles = new_positions


# ============================================================================
# FRAMEWORK DE CLUSTERIZAÇÃO PRINCIPAL
# ============================================================================

class AutomaticBinaryClusteringFramework:
    """
    Framework de Clusterização Automática Baseado em Otimização Binária

    Implementa os principais conceitos do artigo:
    1. Codificação binária para representação de clusters
    2. Estágio inicial de clusterização
    3. Estágio de fusão e modificação
    4. Função objetivo com restrição de distorção similar
    """

    def __init__(self, config: ClusteringConfig = None):
        self.config = config or ClusteringConfig()
        self.optimizer = None
        self.X = None
        self.n_samples = None
        self.n_features = None
        self.bits_per_cluster = None
        self.total_bits = None

    def fit(self, X: np.ndarray) -> ClusterResult:
        """Executa o framework completo de clusterização"""
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Configura K_max se não fornecido
        if self.config.K_max is None:
            self.config.K_max = int(np.floor(np.sqrt(self.n_samples)))

        # Configura métrica de distância para dados binários
        if self.config.binary_mode:
            self.config.distance_metric = 'hamming'

        # Calcula bits necessários por cluster
        self.bits_per_cluster = int(np.ceil(np.log2(self.config.K_max)))
        self.total_bits = self.n_samples * self.bits_per_cluster

        print(f"Configuração: {self.n_samples} amostras, K_max={self.config.K_max}")
        print(f"Bits por cluster: {self.bits_per_cluster}, Total bits: {self.total_bits}")

        # Inicializa otimizador
        self.optimizer = BinaryPSO(
            n_particles=self.config.n_particles,
            n_dimensions=self.total_bits
        )

        # Executa otimização
        print("Iniciando otimização...")
        best_solution, best_cost, cost_history = self.optimizer.optimize(
            objective_func=self._objective_function,
            max_iterations=self.config.max_iterations
        )

        # Decodifica a melhor solução
        final_labels = self._decode_solution(best_solution)

        # Aplica os estágios de re-clusterização e fusão
        final_labels, final_stats = self._apply_reclustering_and_merging(final_labels)

        # Coleta resultados
        result = ClusterResult(
            labels=final_labels,
            n_clusters=len(np.unique(final_labels)),
            centroids=self._calculate_centroids(final_labels),
            representatives=self._select_representatives(final_labels),
            distortion_deviation=final_stats['distortion_deviation'],
            intra_cluster_dist=final_stats['intra_cluster_dist'],
            inter_cluster_dist=final_stats['inter_cluster_dist'],
            db_index=final_stats['db_index'],
            cost_history=cost_history
        )

        return result

    def _objective_function(self, binary_vector: np.ndarray) -> float:
        """Função objetivo do artigo (equação 8)"""

        # 1. Decodifica os labels dos clusters
        labels = self._decode_solution(binary_vector)

        # 2. Estágio inicial de clusterização
        labels, representatives = self._initial_clustering_stage(labels)

        # 3. Estágio de fusão e modificação
        labels, stats = self._merging_modifying_stage(labels, representatives)

        # Extrai parâmetros para função objetivo
        K = len(np.unique(labels))
        delta = stats['distortion_deviation']
        d_max_avg = stats['avg_max_distortion']
        E_min_avg = stats['avg_min_inter_dist']
        K_max = self.config.K_max

        # Calcula função objetivo (equação 8 do artigo)
        if E_min_avg == 0:
            E_min_avg = 1e-10  # Previne divisão por zero

        objective_value = (K_max * delta * d_max_avg) / E_min_avg

        return objective_value

    def _decode_solution(self, binary_vector: np.ndarray) -> np.ndarray:
        """Decodifica vetor binário para labels de cluster"""

        # Reshape para (n_samples, bits_per_cluster)
        binary_matrix = binary_vector.reshape(self.n_samples, self.bits_per_cluster)

        # Converte binário para decimal
        labels = np.zeros(self.n_samples, dtype=int)
        powers_of_two = 2 ** np.arange(self.bits_per_cluster)[::-1]

        for i in range(self.n_samples):
            decimal_value = np.dot(binary_matrix[i], powers_of_two)
            labels[i] = decimal_value % self.config.K_max  # Garante valor dentro do range

        return labels

    def _initial_clustering_stage(self, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estágio inicial de clusterização
        Seleciona representantes e re-aloca pontos
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Calcula centróides
        centroids = self._calculate_centroids(labels)

        # Seleciona representantes (ponto mais próximo do centróide)
        representatives = self._select_representatives(labels, centroids)

        # Re-clusteriza baseado nos representantes
        new_labels = np.zeros(self.n_samples, dtype=int)

        for i in range(self.n_samples):
            distances = []
            for rep_idx in representatives:
                if self.config.binary_mode:
                    dist = self._hamming_distance(self.X[i], self.X[rep_idx])
                else:
                    dist = np.linalg.norm(self.X[i] - self.X[rep_idx])
                distances.append(dist)

            new_labels[i] = np.argmin(distances)

        return new_labels, representatives

    def _merging_modifying_stage(self, labels: np.ndarray,
                                 representatives: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Estágio de fusão e modificação de clusters
        Fusão de clusters com distorção muito pequena
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Calcula distorção máxima por cluster
        cluster_distortions = []
        for label in unique_labels:
            mask = labels == label
            cluster_points = self.X[mask]

            if len(cluster_points) == 0:
                distortion = 0
            else:
                # Distância máxima dentro do cluster
                if self.config.binary_mode:
                    # Para dados binários
                    centroid_idx = representatives[label]
                    distances = [self._hamming_distance(p, self.X[centroid_idx])
                                 for p in cluster_points]
                else:
                    # Para dados contínuos
                    centroid = np.mean(cluster_points, axis=0)
                    distances = [np.linalg.norm(p - centroid)
                                 for p in cluster_points]

                distortion = np.max(distances) if distances else 0

            cluster_distortions.append(distortion)

        cluster_distortions = np.array(cluster_distortions)
        max_distortion = np.max(cluster_distortions)

        # Identifica clusters defeituosos (inequação 4 do artigo)
        defective_clusters = []
        for idx, distortion in enumerate(cluster_distortions):
            if distortion <= self.config.merge_threshold * max_distortion:
                defective_clusters.append(unique_labels[idx])

        # Fusão de clusters defeituosos
        if len(defective_clusters) > 1:
            # Fusiona clusters dois a dois (simplificação)
            merged_labels = labels.copy()

            # Mantém o primeiro cluster, mescla os outros nele
            main_cluster = defective_clusters[0]
            for cluster_label in defective_clusters[1:]:
                merged_labels[merged_labels == cluster_label] = main_cluster

            labels = merged_labels

        # Recalcula estatísticas após fusão
        stats = self._calculate_cluster_statistics(labels)

        return labels, stats

    def _calculate_cluster_statistics(self, labels: np.ndarray) -> Dict[str, float]:
        """Calcula estatísticas dos clusters"""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Distorção por cluster
        cluster_distortions = []
        intra_distances = []
        inter_distances = []

        for label in unique_labels:
            mask = labels == label
            cluster_points = self.X[mask]

            if len(cluster_points) > 0:
                # Distorção máxima
                if self.config.binary_mode:
                    centroid = self._binary_majority_rule(cluster_points)
                    distances = [self._hamming_distance(p, centroid)
                                 for p in cluster_points]
                else:
                    centroid = np.mean(cluster_points, axis=0)
                    distances = [np.linalg.norm(p - centroid)
                                 for p in cluster_points]

                max_distortion = np.max(distances) if distances else 0
                cluster_distortions.append(max_distortion)

                # Distâncias intra-cluster
                if len(cluster_points) > 1:
                    intra_dist = pairwise_distances(cluster_points,
                                                    metric=self._get_distance_metric()).sum()
                    intra_distances.append(intra_dist)

        # Calcula métricas
        if cluster_distortions:
            delta = np.max(cluster_distortions) - np.min(cluster_distortions)  # Desvio de distorção
            d_max_avg = np.mean(cluster_distortions)  # Média da distorção máxima
        else:
            delta = 0
            d_max_avg = 0

        # Distâncias mínimas inter-cluster
        min_inter_dists = []
        centroids = self._calculate_centroids(labels)

        for i, label_i in enumerate(unique_labels):
            min_dist = float('inf')
            centroid_i = centroids[i]

            for j, label_j in enumerate(unique_labels):
                if i != j:
                    centroid_j = centroids[j]

                    if self.config.binary_mode:
                        dist = self._hamming_distance(centroid_i, centroid_j)
                    else:
                        dist = np.linalg.norm(centroid_i - centroid_j)

                    if dist < min_dist:
                        min_dist = dist

            if min_dist < float('inf'):
                min_inter_dists.append(min_dist)

        E_min_avg = np.mean(min_inter_dists) if min_inter_dists else 1

        # DB Index (simplificado)
        db_index = self._calculate_db_index(labels)

        return {
            'distortion_deviation': delta,
            'avg_max_distortion': d_max_avg,
            'avg_min_inter_dist': E_min_avg,
            'intra_cluster_dist': np.sum(intra_distances) if intra_distances else 0,
            'inter_cluster_dist': np.sum(min_inter_dists) if min_inter_dists else 0,
            'db_index': db_index
        }

    def _calculate_db_index(self, labels: np.ndarray) -> float:
        """Calcula DB Index simplificado"""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        if n_clusters <= 1:
            return 0

        # Calcula dispersão intra-cluster
        S = []
        centroids = []

        for label in unique_labels:
            mask = labels == label
            cluster_points = self.X[mask]

            if len(cluster_points) > 0:
                if self.config.binary_mode:
                    centroid = self._binary_majority_rule(cluster_points)
                    dispersion = np.mean([self._hamming_distance(p, centroid)
                                          for p in cluster_points])
                else:
                    centroid = np.mean(cluster_points, axis=0)
                    dispersion = np.mean([np.linalg.norm(p - centroid)
                                          for p in cluster_points])

                S.append(dispersion)
                centroids.append(centroid)

        # Calcula DB Index
        R_values = []
        for i in range(n_clusters):
            max_R = 0
            for j in range(n_clusters):
                if i != j:
                    if self.config.binary_mode:
                        M_ij = self._hamming_distance(centroids[i], centroids[j])
                    else:
                        M_ij = np.linalg.norm(centroids[i] - centroids[j])

                    if M_ij > 0:
                        R_ij = (S[i] + S[j]) / M_ij
                        max_R = max(max_R, R_ij)

            R_values.append(max_R)

        return np.mean(R_values) if R_values else 0

    def _calculate_centroids(self, labels: np.ndarray) -> np.ndarray:
        """Calcula centróides dos clusters"""
        unique_labels = np.unique(labels)
        centroids = []

        for label in unique_labels:
            mask = labels == label
            cluster_points = self.X[mask]

            if len(cluster_points) > 0:
                if self.config.binary_mode:
                    centroid = self._binary_majority_rule(cluster_points)
                else:
                    centroid = np.mean(cluster_points, axis=0)

                centroids.append(centroid)

        return np.array(centroids)

    def _select_representatives(self, labels: np.ndarray,
                                centroids: np.ndarray = None) -> np.ndarray:
        """Seleciona representantes (ponto mais próximo do centróide)"""
        if centroids is None:
            centroids = self._calculate_centroids(labels)

        unique_labels = np.unique(labels)
        representatives = []

        for idx, label in enumerate(unique_labels):
            mask = labels == label
            cluster_points = self.X[mask]
            indices = np.where(mask)[0]

            if len(cluster_points) > 0:
                centroid = centroids[idx]

                # Encontra ponto mais próximo do centróide
                if self.config.binary_mode:
                    distances = [self._hamming_distance(p, centroid)
                                 for p in cluster_points]
                else:
                    distances = [np.linalg.norm(p - centroid)
                                 for p in cluster_points]

                closest_idx = np.argmin(distances)
                representatives.append(indices[closest_idx])

        return np.array(representatives)

    def _binary_majority_rule(self, binary_points: np.ndarray) -> np.ndarray:
        """Regra da maioria para dados binários (Exemplo 2 do artigo)"""
        if len(binary_points) == 0:
            return np.zeros(self.n_features)

        # Para cada feature, escolhe o valor mais frequente (0 ou 1)
        centroid = np.zeros(self.n_features)
        for j in range(self.n_features):
            counts = np.bincount(binary_points[:, j].astype(int))
            if len(counts) > 1:
                centroid[j] = np.argmax(counts)
            else:
                centroid[j] = 0

        return centroid

    def _hamming_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Distância de Hamming para dados binários"""
        return np.sum(a != b) / len(a)  # Normalizada

    def _get_distance_metric(self) -> str:
        """Retorna métrica de distância apropriada"""
        if self.config.binary_mode:
            return 'hamming'
        return self.config.distance_metric

    def _apply_reclustering_and_merging(self, labels: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Aplica re-clusterização e fusão até convergir"""
        max_iterations = 10
        prev_labels = labels.copy()

        for iteration in range(max_iterations):
            # Re-clusterização baseada em representantes
            centroids = self._calculate_centroids(labels)
            representatives = self._select_representatives(labels, centroids)

            # Atribui cada ponto ao representante mais próximo
            new_labels = np.zeros(self.n_samples, dtype=int)
            for i in range(self.n_samples):
                distances = []
                for rep_idx in representatives:
                    if self.config.binary_mode:
                        dist = self._hamming_distance(self.X[i], self.X[rep_idx])
                    else:
                        dist = np.linalg.norm(self.X[i] - self.X[rep_idx])
                    distances.append(dist)

                new_labels[i] = np.argmin(distances)

            # Verifica convergência
            if np.array_equal(new_labels, prev_labels):
                break

            prev_labels = new_labels.copy()
            labels = new_labels

        # Estágio final de fusão
        labels, stats = self._merging_modifying_stage(labels, [])

        return labels, stats


# ============================================================================
# FUNÇÕES AUXILIARES E VISUALIZAÇÃO
# ============================================================================

def generate_sample_datasets() -> Dict[str, np.ndarray]:
    """Gera datasets de exemplo para teste"""
    np.random.seed(42)

    datasets = {}

    # Dataset 1: Dados contínuos (2D para visualização)
    n_samples = 200
    # Cluster 1
    cluster1 = np.random.randn(n_samples // 4, 2) * 0.5 + [2, 2]
    # Cluster 2
    cluster2 = np.random.randn(n_samples // 4, 2) * 0.5 + [8, 8]
    # Cluster 3
    cluster3 = np.random.randn(n_samples // 4, 2) * 0.5 + [2, 8]
    # Cluster 4
    cluster4 = np.random.randn(n_samples // 4, 2) * 0.5 + [8, 2]

    datasets['continuous_2d'] = np.vstack([cluster1, cluster2, cluster3, cluster4])

    # Dataset 2: Dados binários simulados
    n_samples = 150
    n_features = 10

    # Gera 3 clusters com padrões binários diferentes
    binary_data = []
    for cluster_idx in range(3):
        n_cluster_samples = n_samples // 3
        cluster_pattern = np.zeros((n_cluster_samples, n_features))

        # Cada cluster tem um padrão diferente de features ativas
        start_idx = cluster_idx * 3
        end_idx = start_idx + 4

        for i in range(n_cluster_samples):
            # Features específicas do cluster com alta probabilidade
            pattern = np.zeros(n_features)
            pattern[start_idx:end_idx] = np.random.choice([0, 1], size=4, p=[0.2, 0.8])

            # Outras features com baixa probabilidade
            other_indices = [j for j in range(n_features) if j < start_idx or j >= end_idx]
            pattern[other_indices] = np.random.choice([0, 1], size=len(other_indices), p=[0.8, 0.2])

            cluster_pattern[i] = pattern

        binary_data.append(cluster_pattern)

    datasets['binary_10d'] = np.vstack(binary_data)

    # Dataset 3: Dados de alta dimensionalidade
    datasets['high_dim'] = np.random.randn(100, 20)

    return datasets


def plot_clustering_results(results: Dict[str, ClusterResult],
                            datasets: Dict[str, np.ndarray]):
    """Visualiza os resultados da clusterização"""

    n_datasets = len(results)
    fig, axes = plt.subplots(2, n_datasets, figsize=(5 * n_datasets, 10))

    if n_datasets == 1:
        axes = axes.reshape(2, 1)

    for idx, (dataset_name, result) in enumerate(results.items()):
        X = datasets[dataset_name]

        # Gráfico 1: Visualização dos clusters (para dados 2D)
        if X.shape[1] == 2:
            ax = axes[0, idx]
            scatter = ax.scatter(X[:, 0], X[:, 1], c=result.labels,
                                 cmap='tab20', s=50, alpha=0.7)

            # Marca representantes
            if hasattr(result, 'representatives') and result.representatives is not None:
                reps = result.representatives[:min(10, len(result.representatives))]
                ax.scatter(X[reps, 0], X[reps, 1],
                           c='red', s=200, marker='*', edgecolors='black', label='Representantes')
                ax.legend()

            ax.set_title(f'{dataset_name}\nClusters: {result.n_clusters}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')

        # Gráfico 2: Histórico de custo
        ax2 = axes[1, idx]
        ax2.plot(result.cost_history, 'b-', linewidth=2)
        ax2.set_xlabel('Iteração')
        ax2.set_ylabel('Valor da Função Objetivo')
        ax2.set_title('Convergência do Otimizador')
        ax2.grid(True, alpha=0.3)

        # Adiciona métricas no gráfico
        metrics_text = (f'Métricas:\n'
                        f'DB Index: {result.db_index:.3f}\n'
                        f'Dev Dist: {result.distortion_deviation:.3f}\n'
                        f'Intra: {result.intra_cluster_dist:.1f}\n'
                        f'Inter: {result.inter_cluster_dist:.1f}')

        ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes,
                 verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


def analyze_correlated_binary_datasets():
    """Análise específica para datasets binários correlacionados"""

    # Gera datasets binários com diferentes níveis de correlação
    np.random.seed(42)

    correlated_datasets = {}

    for correlation in [0.5, 0.6, 0.7]:
        n_samples = 120
        n_features = 15

        # Gera um dataset base com padrão
        base_pattern = np.random.choice([0, 1], size=n_features, p=[0.5, 0.5])

        dataset = []
        for i in range(n_samples):
            pattern = base_pattern.copy()

            # Introduz correlação
            n_changes = int((1 - correlation) * n_features)
            change_indices = np.random.choice(n_features, n_changes, replace=False)

            for idx in change_indices:
                pattern[idx] = 1 - pattern[idx]  # Inverte o bit

            dataset.append(pattern)

        correlated_datasets[f'binary_corr_{int(correlation * 100)}'] = np.array(dataset)

    # Aplica clusterização em cada dataset
    results = {}

    for name, X in correlated_datasets.items():
        print(f"\nProcessando {name}...")

        config = ClusteringConfig(
            binary_mode=True,
            optimizer_type='BPSO',
            n_particles=25,
            max_iterations=80
        )

        framework = AutomaticBinaryClusteringFramework(config)
        result = framework.fit(X)
        results[name] = result

        print(f"  Clusters encontrados: {result.n_clusters}")
        print(f"  DB Index: {result.db_index:.4f}")
        print(f"  Desvio de Distorção: {result.distortion_deviation:.4f}")

    return correlated_datasets, results


# ============================================================================
# EXEMPLO DE USO COMPLETO
# ============================================================================

def main():
    """Exemplo de uso completo do framework"""

    print("=" * 70)
    print("FRAMEWORK DE CLUSTERIZAÇÃO AUTOMÁTICA BASEADO EM OTIMIZAÇÃO BINÁRIA")
    print("Inspirado no artigo de Merikhi & Soleymani (IEEE Access, 2021)")
    print("=" * 70)

    # 1. Gera datasets de exemplo
    print("\n1. Gerando datasets de exemplo...")
    datasets = generate_sample_datasets()

    # 2. Processa cada dataset
    results = {}

    for dataset_name, X in datasets.items():
        print(f"\n2. Processando dataset: {dataset_name}")
        print(f"   Forma: {X.shape}")

        # Configuração baseada no tipo de dados
        if 'binary' in dataset_name:
            config = ClusteringConfig(
                binary_mode=True,
                optimizer_type='BPSO',
                n_particles=30,
                max_iterations=100
            )
        else:
            config = ClusteringConfig(
                binary_mode=False,
                optimizer_type='BPSO',
                n_particles=40,
                max_iterations=120
            )

        # Cria e executa o framework
        framework = AutomaticBinaryClusteringFramework(config)
        result = framework.fit(X)
        results[dataset_name] = result

        # Exibe resultados
        print(f"   Número de clusters encontrados: {result.n_clusters}")
        print(f"   DB Index: {result.db_index:.4f}")
        print(f"   Desvio de Distorção: {result.distortion_deviation:.4f}")
        print(f"   Distância Intra-cluster: {result.intra_cluster_dist:.2f}")
        print(f"   Distância Inter-cluster: {result.inter_cluster_dist:.2f}")

    # 3. Visualiza resultados
    print("\n3. Visualizando resultados...")
    plot_clustering_results(results, datasets)

    # 4. Análise específica para dados binários correlacionados
    print("\n4. Análise de datasets binários correlacionados...")
    correlated_datasets, corr_results = analyze_correlated_binary_datasets()

    # 5. Exemplo de uso programático
    print("\n5. Exemplo de uso programático:")

    # Para usar em seu próprio código:
    """
    from clustering_framework import AutomaticBinaryClusteringFramework, ClusteringConfig

    # Prepare seus dados
    X = seu_dataframe.values  # ou numpy array

    # Configure o framework
    config = ClusteringConfig(
        binary_mode=True,  # ou False para dados contínuos
        n_particles=30,
        max_iterations=100
    )

    # Execute a clusterização
    framework = AutomaticBinaryClusteringFramework(config)
    result = framework.fit(X)

    # Use os resultados
    print(f"Número de clusters: {result.n_clusters}")
    print(f"Labels: {result.labels}")
    print(f"Centroids: {result.centroids}")
    """

    print("\n" + "=" * 70)
    print("Execução concluída!")
    print("=" * 70)


if __name__ == "__main__":
    main()