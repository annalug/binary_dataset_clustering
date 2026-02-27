import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from adjust_text import adjust_text  # Precisa instalar: pip install adjust_text

# Importando seus módulos
from dataset_loader import load_datasets_from_folder
from standardizer import DatasetStandardizer
from siamese import SiameseNet

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================
DATA_PATH = "./data"
N_CLUSTERS = 3
SAMPLES_PER_DATASET = 60


# ==========================================================
# FUNÇÃO DE VISUALIZAÇÃO (Fica dentro deste arquivo)
# ==========================================================
def plot_instance_cloud(df_meta, cluster_labels, dataset_names):
    """
    Gera o gráfico de nuvem de pontos com nomes organizados e coloridos.
    """
    plt.figure(figsize=(16, 10))
    sns.set_style("white")

    # Cores fixas para os clusters
    palette = sns.color_palette("husl", len(np.unique(cluster_labels)))

    # Plotar as instâncias (pontos)
    scatter = sns.scatterplot(
        data=df_meta, x='x', y='y',
        hue='Cluster', style='Cluster',
        alpha=0.4, s=30, palette=palette
    )

    # Mapeamento para saber a cor de cada dataset
    ds_to_cluster = {name: cluster_labels[i] for i, name in enumerate(dataset_names)}

    texts = []
    for name in dataset_names:
        subset = df_meta[df_meta['Dataset'] == name]
        c_id = ds_to_cluster[name]

        # Criar label com ID do cluster: "[0] nome_do_dataset"
        label_text = f"[{c_id}] {name}"

        # Adiciona o texto na posição média da nuvem
        t = plt.text(
            subset['x'].mean(),
            subset['y'].mean(),
            label_text,
            fontsize=10,
            fontweight='bold',
            color=palette[c_id]  # Cor do texto igual à do cluster
        )
        texts.append(t)

    # Ajusta os textos para não ficarem um em cima do outro
    print("Ajustando posições dos nomes (isso pode levar segundos)...")
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

    plt.title("Nuvem de Instâncias: Agrupamento de Datasets via Siamese Net", fontsize=16)
    plt.grid(True, alpha=0.1)

    output_file = "cluster_instancias_final.jpg"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico salvo como: {output_file}")
    plt.show()


# ==========================================================
# FLUXO PRINCIPAL
# ==========================================================
def main():
    # 1. Carregar dados
    datasets, names, _ = load_datasets_from_folder(DATA_PATH)
    standardizer = DatasetStandardizer(target_samples=256, target_features=100)
    datasets_std = standardizer.fit_transform_batch(datasets)

    # 2. Obter Embeddings do Encoder
    model = SiameseNet(input_shape=(256, 100, 1), embedding_dim=128)
    # model.load("siamese_model") # Carregue se tiver o modelo treinado

    dataset_embeddings = []
    for ds in datasets_std:
        emb = model.encoder.predict(np.expand_dims(ds, axis=0), verbose=0)
        dataset_embeddings.append(emb.flatten())

    dataset_embeddings = np.array(dataset_embeddings)

    # 3. Clusterização
    clusterer = AgglomerativeClustering(n_clusters=N_CLUSTERS)
    labels = clusterer.fit_predict(dataset_embeddings)

    # 4. Preparar Dados para o Plot
    all_rows = []
    row_metadata = []
    for i, ds in enumerate(datasets_std):
        rows = ds.reshape(256, 100)
        idx = np.random.choice(256, SAMPLES_PER_DATASET, replace=False)
        all_rows.append(rows[idx])

        for _ in range(SAMPLES_PER_DATASET):
            row_metadata.append({
                'Dataset': names[i],
                'Cluster': f"Cluster {labels[i]}"
            })

    # Redução t-SNE das instâncias
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(np.vstack(all_rows))

    df_meta = pd.DataFrame(row_metadata)
    df_meta['x'] = X_2d[:, 0]
    df_meta['y'] = X_2d[:, 1]

    # 5. Chamar a função de plot
    plot_instance_cloud(df_meta, labels, names)

    # 6. Exibir tabela no console
    resumo = pd.DataFrame({'Dataset': names, 'Cluster': labels}).sort_values('Cluster')
    print("\nTABELA DE CLUSTERIZAÇÃO:")
    print(resumo.to_string(index=False))


if __name__ == "__main__":
    main()