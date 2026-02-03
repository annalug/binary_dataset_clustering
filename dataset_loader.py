import os
import pandas as pd
import numpy as np
from typing import List, Tuple

def load_datasets_from_folder(
    folder_path: str,
    class_column: str = "class"
) -> Tuple[List[np.ndarray], List[str], List[np.ndarray]]:
    """
    Lê datasets CSV binários e separa features e rótulos.

    Returns:
        datasets: lista de arrays (n_samples, n_features)
        names: nomes dos arquivos
        labels: lista de vetores de classe (n_samples,)
    """
    datasets = []
    names = []
    labels = []

    for file in sorted(os.listdir(folder_path)):
        if not file.endswith(".csv"):
            continue

        path = os.path.join(folder_path, file)
        df = pd.read_csv(path)

        if class_column not in df.columns:
            raise ValueError(f"Arquivo {file} não possui coluna '{class_column}'")

        y = df[class_column].values
        X = df.drop(columns=[class_column]).values

        # garante binaridade
        X = np.clip(X, 0, 1).astype(np.float32)

        datasets.append(X)
        labels.append(y)
        names.append(file.replace(".csv", ""))

        print(f"✓ {file}: X={X.shape}, classes={np.unique(y)}")

    return datasets, names, labels
