from dataset_loader import load_datasets_from_folder
from standardizer import DatasetStandardizer
from siamese import SiameseNet
from ranking import rank_similar_datasets
import numpy as np

DATA_PATH = "./data"

# 1. Load CSVs
datasets, names, labels = load_datasets_from_folder(DATA_PATH)

# 2. Standardize
standardizer = DatasetStandardizer()
datasets_std = standardizer.fit_transform_batch(datasets)

# 3. Load model
model = SiameseNet(
    input_shape=(256, 100, 1),
    embedding_dim=128
)
# model.load("siamese_model")

# 4. Embeddings
embeddings = np.array([
    model.get_embedding(d) for d in datasets_std
])

# 5. Ranking
query = names[0]  # ex: primeiro dataset
ranking = rank_similar_datasets(embeddings, names, query)

print(f"\n🔎 Ranking de similaridade para: {query}\n")
for name, score in ranking:
    print(f"{name:25s} → {score:.4f}")
