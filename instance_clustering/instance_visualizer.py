import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd


def plot_instances_and_clusters(
        datasets_std: np.ndarray,
        dataset_names: list,
        cluster_labels: np.ndarray,
        max_datasets: int = 10,
        samples_per_dataset: int = 50
):
    """
    Plota as instâncias individuais e como elas se agrupam por dataset e cluster.

    Args:
        datasets_std: Array (n_datasets, samples, features, 1)
        dataset_names: Lista com nomes dos datasets
        cluster_labels: Labels resultantes do clustering de datasets
        max_datasets: Limite de datasets para não poluir o gráfico
        samples_per_dataset: Quantas linhas plotar por dataset
    """
    all_instances = []
    instance_labels_dataset = []
    instance_labels_cluster = []

    # 1. Preparação dos dados
    # Selecionamos apenas alguns datasets para visualização clara
    selected_indices = np.random.choice(len(datasets_std), min(len(datasets_std), max_datasets), replace=False)

    for idx in selected_indices:
        ds = datasets_std[idx].reshape(datasets_std.shape[1], -1)  # Remove canal e flatten

        # Sub-amostragem de instâncias para o plot
        sub_idx = np.random.choice(len(ds), min(len(ds), samples_per_dataset), replace=False)
        samples = ds[sub_idx]

        all_instances.append(samples)
        instance_labels_dataset.extend([dataset_names[idx]] * len(samples))
        instance_labels_cluster.extend([f"Cluster {cluster_labels[idx]}"] * len(samples))

    X_total = np.vstack(all_instances)

    # 2. Redução de Dimensionalidade (Espaço comum para todas as instâncias)
    print(f"Reduzindo {len(X_total)} instâncias para 2D...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X_total)

    df_plot = pd.DataFrame({
        'x': X_2d[:, 0],
        'y': X_2d[:, 1],
        'Dataset': instance_labels_dataset,
        'Cluster': instance_labels_cluster
    })

    # 3. Plotagem
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    # Criamos o scatter plot colorindo por Cluster e usando marcadores diferentes por Dataset
    scatter = sns.scatterplot(
        data=df_plot,
        x='x', y='y',
        hue='Cluster',
        style='Dataset',
        palette='viridis',
        s=60, alpha=0.6
    )

    # Adicionar polígonos (Convex Hull) ou sombras para destacar o agrupamento dos datasets
    for ds_name in df_plot['Dataset'].unique():
        ds_points = df_plot[df_plot['Dataset'] == ds_name][['x', 'y']].values
        center = ds_points.mean(axis=0)
        plt.annotate(ds_name, center, fontsize=9, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    plt.title("Distribuição de Instâncias: Agrupamento de Datasets no Espaço de Features", fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("instance_clustering_view.jpg", dpi=300)
    plt.show()

# Exemplo de integração no run_ranking.py:
# plot_instances_and_clusters(datasets_std, names, cluster_labels)