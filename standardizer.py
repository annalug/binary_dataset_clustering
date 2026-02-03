"""
DatasetStandardizer - Versão Corrigida para Dataset-Level Similarity
Padronização independente de datasets binários de malware Android
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import List
import warnings

warnings.filterwarnings("ignore")


class DatasetStandardizer:
    """
    Padroniza datasets binários para formato fixo (target_samples, target_features, 1)

    ✔ Cada dataset é tratado de forma INDEPENDENTE
    ✔ PCA nunca é compartilhado entre datasets
    ✔ Correto para similaridade entre datasets (Siamese)
    """

    def __init__(
        self,
        target_samples: int = 256,
        target_features: int = 100,
        use_pca: bool = True,
        min_variance_ratio: float = 0.90,
        random_state: int = 42
    ):
        self.target_samples = target_samples
        self.target_features = target_features
        self.use_pca = use_pca
        self.min_variance_ratio = min_variance_ratio
        self.random_state = random_state

    # ------------------------------------------------------------------
    # PAD DE AMOSTRAS
    # ------------------------------------------------------------------
    def _pad_samples(self, data: np.ndarray) -> np.ndarray:
        n_samples = data.shape[0]

        if n_samples < self.target_samples:
            padding = np.zeros((self.target_samples - n_samples, data.shape[1]))
            return np.vstack([data, padding])

        if n_samples > self.target_samples:
            np.random.seed(self.random_state)
            idx = np.random.choice(n_samples, self.target_samples, replace=False)
            return data[np.sort(idx)]

        return data

    # ------------------------------------------------------------------
    # PAD DE FEATURES
    # ------------------------------------------------------------------
    def _pad_features(self, data: np.ndarray) -> np.ndarray:
        n_features = data.shape[1]

        if n_features < self.target_features:
            padding = np.zeros((data.shape[0], self.target_features - n_features))
            return np.hstack([data, padding])

        return data

    # ------------------------------------------------------------------
    # REDUÇÃO DE FEATURES (PCA LOCAL)
    # ------------------------------------------------------------------
    def _reduce_features_pca(self, data: np.ndarray) -> np.ndarray:
        n_features = data.shape[1]

        if n_features <= self.target_features:
            return self._pad_features(data)

        pca = PCA(
            n_components=self.target_features,
            random_state=self.random_state
        )

        transformed = pca.fit_transform(data)

        var_explained = np.sum(pca.explained_variance_ratio_)
        print(
            f"✓ PCA local: {n_features} → {self.target_features} "
            f"({var_explained * 100:.1f}% variância)"
        )

        if var_explained < self.min_variance_ratio:
            warnings.warn(
                f"PCA explica apenas {var_explained * 100:.1f}% da variância. "
                f"Considere aumentar target_features ou usar use_pca=False"
            )

        # Re-binariza
        return (transformed > 0.5).astype(np.float32)

    # ------------------------------------------------------------------
    # TRUNCAMENTO
    # ------------------------------------------------------------------
    def _reduce_features_truncate(self, data: np.ndarray) -> np.ndarray:
        n_features = data.shape[1]

        if n_features > self.target_features:
            return data[:, :self.target_features]

        if n_features < self.target_features:
            return self._pad_features(data)

        return data

    # ------------------------------------------------------------------
    # FIT + TRANSFORM (ÚNICO MÉTODO)
    # ------------------------------------------------------------------
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        if data.ndim != 2:
            raise ValueError(f"Esperado array 2D, recebido {data.shape}")

        print(
            f"\n📊 Fit-transform: {data.shape} → "
            f"({self.target_samples}, {self.target_features}, 1)"
        )

        # Garante binaridade
        data = np.clip(data, 0, 1).astype(np.float32)

        # 1. Padroniza amostras
        data = self._pad_samples(data)

        # 2. Padroniza features
        if self.use_pca:
            data = self._reduce_features_pca(data)
        else:
            data = self._reduce_features_truncate(data)

        # 3. Adiciona canal
        data = np.expand_dims(data, axis=-1)

        print(f"✓ Output: {data.shape}")
        return data

    # ------------------------------------------------------------------
    # BATCH (TODOS USAM FIT_TRANSFORM)
    # ------------------------------------------------------------------
    def fit_transform_batch(
        self,
        datasets: List[np.ndarray],
        show_progress: bool = True
    ) -> np.ndarray:

        if not datasets:
            raise ValueError("Lista de datasets vazia")

        print("\n" + "=" * 70)
        print(f"BATCH FIT-TRANSFORM (INDEPENDENTE): {len(datasets)} datasets")
        print("=" * 70)

        transformed = []

        for i, dataset in enumerate(datasets, start=1):
            if show_progress:
                print(f"\n[{i}/{len(datasets)}]", end=" ")
            transformed.append(self.fit_transform(dataset))

        result = np.array(transformed)

        print("\n" + "=" * 70)
        print(f"✓ Batch concluído: {result.shape}")
        print("=" * 70 + "\n")

        return result
