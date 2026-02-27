# Android Malware Dataset Clustering & Similarity Ranking

### This project provides a comprehensive framework for analyzing, ranking, and clustering Android malware datasets. By utilizing multiple approaches—from Deep Learning (Siamese Networks and Autoencoders) to Statistical/Semantic analysis—it identifies structural and behavioral similarities between different data sources (e.g., permissions, API calls, network features).
### 🚀 Key Features
The system offers three distinct methodologies to evaluate dataset similarity:
* Siamese Pipeline (Main): Uses a Convolutional Neural Network (CNN) Siamese architecture to learn embeddings directly from raw binary data structures.
* Version A (Direct Statistical/Semantic): A deterministic approach combining statistical descriptors (density, variance, histograms) with NLP (TF-IDF) on feature names.
* Version B (Autoencoder): An unsupervised approach that compresses high-dimensional descriptors into a refined latent space using an MLP Autoencoder.

### 📁 Project Structure

````
.
├── data/                   # Input folder for binary .csv datasets
├── img/                    # Generated visualizations (t-SNE, Dendrograms, heatmaps)
├── models/                 # Saved model weights (.keras)
├── output/                 # CSV reports and similarity matrices
│
├── config.py               # Central configuration (Hyperparameters, Paths, Constants)
├── run_ranking.py          # Main execution script (Siamese Pipeline)
├── version_a_direct.py     # Version A: Direct Statistical + Semantic approach
├── version_b_autoencoder.py# Version B: Latent Space + Autoencoder approach
│
├── siamese.py              # Siamese Network architecture & Contrastive Loss
├── standardizer.py         # Data reshaping, PCA, and padding logic
├── pair_generator.py       # Logic for creating training pairs (Similar vs Different)
├── ranking.py              # Similarity scoring and ranking logic
├── similarity_clustering.py # Clustering algorithms (Hierarchical, DBSCAN)
├── cluster_selection.py    # Automatic K-selection (Silhouette, Elbow method)
└── plot_clusters.py        # Advanced plotting utilities
````
### 🛠️ Requirements
Python 3.9+
TensorFlow 2.10+
Scikit-learn
Pandas & Numpy
Matplotlib & Seaborn
Scipy


### ⚙️ Configuration
Before running, adjust settings in config.py:
Dataset Types: Map your datasets to types in CFG.pairs.dataset_type_map (e.g., {'dataset_1': 0, 'dataset_2': 0} means they share the same structure/nature).
Shapes: Set target_samples and target_features for the standardizer.
Clustering: Choose the default ranking metric (cosine or euclidean).

### 🏃 How to Run
1. Main Pipeline (Siamese Network)
This version trains a model to recognize if two datasets belong to the same "type" based on their binary patterns.
````
python run_ranking.py
````

2. Version A (Deterministic)
Best for quick analysis without training. It uses column names (TF-IDF) and data statistics.
````
python version_a_direct.py
````

3. Version B (Latent Space)
Trains an Autoencoder to find a compressed representation of the datasets' characteristics.
````
python version_b_autoencoder.py
````

### 📊 Outputs & Interpretation
After execution, check the following directories:
* /img:
cluster_visualization.jpg: 2D t-SNE plot of your datasets.
dendrogram.jpg: Hierarchical relationship tree.
similarity_matrix.jpg: Heatmap showing how similar each dataset is to others.
* /output:
ranking_results.csv: A ranked list of the most similar datasets for each source.
clustering_report.txt: Summary including Silhouette Score and cluster compositions.
similarity_matrix.csv: Raw similarity scores.
### Metrics Used
* Silhouette Score: Measures how similar an object is to its own cluster compared to others (higher is better).
* ARI (Adjusted Rand Index): If business labels are provided, this measures the agreement between the model and human experts.
* Reconstruction Error (Ver. B): Helps identify "outlier" datasets that don't fit the general patterns of the collection.










