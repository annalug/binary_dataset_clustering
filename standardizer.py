"""
standardizer.py — Padronização de datasets binários para formato fixo

Correções aplicadas:
  - PCA NÃO re-binariza o output: valores contínuos preservam informação
    relevante para o encoder CNN (era (transformed > 0.5) antes)
  - Subsample usa seed local para não contaminar estado global do numpy
  - Parâmetros lidos de CFG por padrão; ainda aceita override via __init__
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import List, Optional
import warnings

from config import CFG, StandardizerConfig

warnings.filterwarnings("ignore")


class DatasetStandardizer:
    """
    Padroniza datasets binários para shape fixo (target_samples, target_features, 1).

    Cada dataset é transformado de forma INDEPENDENTE — nenhum PCA ou scaler
    é compartilhado entre datasets, o que é essencial para a comparação
    entre datasets na rede siamesa.

    Fluxo:
        dados brutos (n, m)
            → clip [0, 1]          (garante binaridade)
            → pad/subsample linhas (→ target_samples)
            → PCA ou truncamento   (→ target_features, valores CONTÍNUOS)
            → expand_dims          (→ target_samples, target_features, 1)
    """

    def __init__(self, cfg: Optional[StandardizerConfig] = None):
        """
        Args:
            cfg: StandardizerConfig. Se None, usa CFG.standardizer global.
        """
        c = cfg or CFG.standardizer
        self.target_samples    = c.target_samples
        self.target_features   = c.target_features
        self.use_pca           = c.use_pca
        self.min_variance_ratio = c.min_variance_ratio
        self.random_state      = c.random_state

    # ------------------------------------------------------------------
    # AMOSTRAS
    # ------------------------------------------------------------------

    def _pad_samples(self, data: np.ndarray) -> np.ndarray:
        """Pad com zeros ou subsample para chegar em target_samples."""
        n = data.shape[0]

        if n < self.target_samples:
            pad = np.zeros((self.target_samples - n, data.shape[1]), dtype=np.float32)
            return np.vstack([data, pad])

        if n > self.target_samples:
            rng = np.random.default_rng(self.random_state)   # seed local
            idx = rng.choice(n, self.target_samples, replace=False)
            return data[np.sort(idx)]

        return data

    # ------------------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------------------

    def _pad_features(self, data: np.ndarray) -> np.ndarray:
        """Pad com zeros à direita para chegar em target_features."""
        n_feat = data.shape[1]
        if n_feat < self.target_features:
            pad = np.zeros((data.shape[0], self.target_features - n_feat), dtype=np.float32)
            return np.hstack([data, pad])
        return data

    def _reduce_pca(self, data: np.ndarray) -> np.ndarray:
        """
        Reduz features via PCA local (fit+transform no próprio dataset).

        IMPORTANTE: não re-binariza o resultado. Valores contínuos
        carregam mais informação para o encoder CNN do que um threshold
        arbitrário de 0.5.
        """
        n_feat = data.shape[1]

        if n_feat <= self.target_features:
            return self._pad_features(data)

        # n_components não pode exceder min(n_samples, n_features)
        n_components = min(self.target_features, data.shape[0], n_feat)

        pca = PCA(n_components=n_components, random_state=self.random_state)
        transformed = pca.fit_transform(data).astype(np.float32)

        var_explained = float(np.sum(pca.explained_variance_ratio_))
        print(
            f"    PCA: {n_feat} → {n_components} features "
            f"({var_explained * 100:.1f}% variância explicada)"
        )

        if var_explained < self.min_variance_ratio:
            warnings.warn(
                f"PCA capturou apenas {var_explained * 100:.1f}% da variância "
                f"(mínimo esperado: {self.min_variance_ratio * 100:.0f}%). "
                f"Considere aumentar target_features.",
                UserWarning
            )

        # Pad se n_components < target_features (caso raro)
        if n_components < self.target_features:
            pad = np.zeros(
                (transformed.shape[0], self.target_features - n_components),
                dtype=np.float32
            )
            transformed = np.hstack([transformed, pad])

        return transformed

    def _truncate(self, data: np.ndarray) -> np.ndarray:
        """Trunca ou pad features sem PCA."""
        n_feat = data.shape[1]
        if n_feat > self.target_features:
            return data[:, :self.target_features]
        return self._pad_features(data)

    # ------------------------------------------------------------------
    # API PÚBLICA
    # ------------------------------------------------------------------

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transforma um único dataset para shape (target_samples, target_features, 1).

        Args:
            data: Array 2D (n_samples, n_features), valores binários 0/1

        Returns:
            Array 3D (target_samples, target_features, 1), dtype float32
        """
        if data.ndim != 2:
            raise ValueError(
                f"fit_transform espera array 2D, recebeu shape {data.shape}"
            )

        print(
            f"  {data.shape[0]:>6} amostras × {data.shape[1]:>5} features "
            f"→ ({self.target_samples}, {self.target_features}, 1)"
        )

        # 1. Garante valores em [0, 1]
        data = np.clip(data, 0.0, 1.0).astype(np.float32)

        # 2. Padroniza número de amostras
        data = self._pad_samples(data)

        # 3. Padroniza número de features
        if self.use_pca:
            data = self._reduce_pca(data)
        else:
            data = self._truncate(data)

        # 4. Adiciona dimensão de canal (para CNN)
        data = np.expand_dims(data, axis=-1)   # (S, F, 1)

        return data

    def fit_transform_batch(
        self,
        datasets: List[np.ndarray],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Transforma uma lista de datasets de forma INDEPENDENTE.

        Args:
            datasets: Lista de arrays 2D
            show_progress: Mostra progresso no terminal

        Returns:
            Array 4D (n_datasets, target_samples, target_features, 1)
        """
        if not datasets:
            raise ValueError("Lista de datasets vazia.")

        sep = "=" * 70
        print(f"\n{sep}")
        print(f"PADRONIZAÇÃO EM BATCH — {len(datasets)} datasets")
        print(f"  Modo: {'PCA local' if self.use_pca else 'truncamento'}")
        print(f"  Shape alvo: ({self.target_samples}, {self.target_features}, 1)")
        print(sep)

        transformed = []
        for i, ds in enumerate(datasets, start=1):
            if show_progress:
                print(f"\n[{i:2d}/{len(datasets)}]", end=" ")
            transformed.append(self.fit_transform(ds))

        result = np.array(transformed)   # (N, S, F, 1)

        print(f"\n{sep}")
        print(f"✓ Batch concluído → shape: {result.shape}")
        print(sep + "\n")

        return result