"""
siamese.py — Rede Neural Siamesa para comparação de datasets de malware Android

Correções aplicadas:
  - Removidas layers.Lambda (causavam falha ao salvar/carregar o modelo).
    Substituídas por:
      • L2-normalization → layers.UnitNormalization (Keras nativo)
      • Distância euclidiana → camada customizada EuclideanDistance
        (subclasse de layers.Layer, serializável)
  - Loss substituída por Contrastive Loss (mais adequada para siamese
    com distância euclidiana e margem configurável via CFG)
  - Hiperparâmetros lidos de CFG; ainda aceita override no __init__
  - ModelCheckpoint salva em CFG.paths.models/
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, callbacks
from typing import Tuple, Optional, List

from config import CFG, SiameseConfig


# ============================================================================
# CAMADA CUSTOMIZADA — distância euclidiana serializável
# ============================================================================

class EuclideanDistance(layers.Layer):
    """
    Calcula a distância euclidiana entre dois vetores de embedding.

    Entrada : [emb_left (B, D), emb_right (B, D)]
    Saída   : distância (B, 1)

    Usa subclasse de Layer em vez de Lambda para garantir
    serialização correta ao salvar/carregar o modelo.
    """

    def call(self, inputs):
        left, right = inputs
        sq_diff = tf.math.squared_difference(left, right)      # (B, D)
        sum_sq  = tf.reduce_sum(sq_diff, axis=1, keepdims=True)  # (B, 1)
        return tf.sqrt(tf.maximum(sum_sq, 1e-12))              # evita sqrt(0)

    def get_config(self):
        return super().get_config()


# ============================================================================
# LOSS CONTRASTIVA
# ============================================================================

def contrastive_loss(margin: float = 1.0):
    """
    Contrastive Loss (Hadsell et al., 2006).

    Para pares similares   (y=1): loss = dist²
    Para pares diferentes  (y=0): loss = max(0, margin − dist)²

    Args:
        margin: Margem mínima entre pares diferentes.

    Returns:
        Função de loss compatível com model.compile(loss=...).
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        dist   = tf.squeeze(y_pred, axis=-1)      # (B,)

        similar_loss   = y_true * tf.square(dist)
        different_loss = (1.0 - y_true) * tf.square(tf.maximum(margin - dist, 0.0))

        return tf.reduce_mean(0.5 * (similar_loss + different_loss))

    loss_fn.__name__ = f"contrastive_loss_m{margin}"
    return loss_fn


# ============================================================================
# REDE SIAMESA
# ============================================================================

class SiameseNet:
    """
    Rede Siamesa com encoder CNN compartilhado.

    Arquitetura:
      Encoder CNN  →  Dense(embedding_dim)  →  UnitNormalization (L2)
      Duas cópias com pesos compartilhados
      EuclideanDistance entre os dois embeddings
      Saída: distância escalar (0 = idênticos)

    Loss: Contrastive Loss com margem configurável
    """

    def __init__(self, cfg: Optional[SiameseConfig] = None):
        """
        Args:
            cfg: SiameseConfig. Se None, usa CFG.siamese global.
        """
        c = cfg or CFG.siamese
        self.input_shape    = c.input_shape
        self.embedding_dim  = c.embedding_dim
        self.learning_rate  = c.learning_rate
        self.architecture   = c.architecture
        self.margin         = c.margin
        self.model_prefix   = CFG.paths.model_file(c.model_prefix)

        # Hiperparâmetros de treino (guardados para uso em train())
        self._train_cfg = c

        self.encoder:        Optional[Model] = None
        self.siamese_model:  Optional[Model] = None
        self.history = None

        self._build_models()

    # ------------------------------------------------------------------
    # BLOCOS DE ENCODER
    # ------------------------------------------------------------------

    def _conv_block(self, x, filters: int, dropout: float):
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout)(x)
        return x

    def _build_encoder_light(self, inputs):
        """2 blocos convolucionais — mais rápido."""
        x = self._conv_block(inputs, 32,  dropout=0.20)
        x = self._conv_block(x,      64,  dropout=0.20)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.30)(x)
        return x

    def _build_encoder_default(self, inputs):
        """3 blocos convolucionais — balanceado."""
        x = self._conv_block(inputs, 32,  dropout=0.25)
        x = self._conv_block(x,      64,  dropout=0.25)
        x = self._conv_block(x,      128, dropout=0.30)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.40)(x)
        return x

    def _build_encoder_deep(self, inputs):
        """4 blocos convolucionais + 2 dense — maior capacidade."""
        x = inputs
        for filters in [32, 64, 128, 256]:
            x = self._conv_block(x, filters, dropout=0.25)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.40)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.40)(x)
        return x

    # ------------------------------------------------------------------
    # CONSTRUÇÃO DO MODELO
    # ------------------------------------------------------------------

    def _build_models(self):
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"CONSTRUINDO REDE SIAMESA  [{self.architecture.upper()}]")
        print(sep)

        # --- Encoder ---
        inputs = layers.Input(shape=self.input_shape, name='encoder_input')

        builders = {
            'light':   self._build_encoder_light,
            'deep':    self._build_encoder_deep,
        }
        x = builders.get(self.architecture, self._build_encoder_default)(inputs)

        # Projeção no espaço de embedding
        emb = layers.Dense(self.embedding_dim, name='projection')(x)

        # L2-normalização via camada nativa do Keras (serializável)
        emb = layers.UnitNormalization(axis=-1, name='l2_norm')(emb)

        self.encoder = Model(inputs=inputs, outputs=emb, name='encoder')

        print(f"\n  Encoder:")
        print(f"    input  : {self.input_shape}")
        print(f"    output : ({self.embedding_dim},)  [L2-normalizado]")
        print(f"    params : {self.encoder.count_params():,}")

        # --- Modelo siamês ---
        input_left  = layers.Input(shape=self.input_shape, name='input_left')
        input_right = layers.Input(shape=self.input_shape, name='input_right')

        emb_left  = self.encoder(input_left)
        emb_right = self.encoder(input_right)

        # Distância euclidiana (camada customizada, sem Lambda)
        distance = EuclideanDistance(name='euclidean_dist')([emb_left, emb_right])

        self.siamese_model = Model(
            inputs=[input_left, input_right],
            outputs=distance,
            name='siamese_network'
        )

        self.siamese_model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss=contrastive_loss(margin=self.margin),
            metrics=['mae']   # MAE na distância como proxy de monitoramento
        )

        print(f"\n  Siamese model:")
        print(f"    loss   : contrastive_loss(margin={self.margin})")
        print(f"    params : {self.siamese_model.count_params():,}")
        print(f"{sep}\n")

    # ------------------------------------------------------------------
    # TREINAMENTO
    # ------------------------------------------------------------------

    def train(
        self,
        pairs_left:  np.ndarray,
        pairs_right: np.ndarray,
        labels:      np.ndarray,
        epochs:              Optional[int]   = None,
        batch_size:          Optional[int]   = None,
        validation_split:    Optional[float] = None,
        early_stopping_patience: Optional[int] = None,
        reduce_lr_patience:      Optional[int] = None,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Treina a rede siamesa com contrastive loss.

        Args:
            pairs_left / pairs_right: Arrays (n_pairs, *input_shape)
            labels: Array (n_pairs,) — 1 = par similar, 0 = par diferente
            *: parâmetros opcionais; se None, usa CFG.siamese

        Returns:
            History object do Keras
        """
        c = self._train_cfg
        epochs       = epochs       or c.epochs
        batch_size   = batch_size   or c.batch_size
        val_split    = validation_split or c.validation_split
        es_patience  = early_stopping_patience or c.early_stopping_patience
        lr_patience  = reduce_lr_patience      or c.reduce_lr_patience

        sep = "=" * 70
        print(f"\n{sep}\nTREINAMENTO\n{sep}")
        print(f"  Pares      : {len(labels)}")
        print(f"  Similares  : {int(np.sum(labels == 1))} ({np.mean(labels == 1)*100:.1f}%)")
        print(f"  Diferentes : {int(np.sum(labels == 0))} ({np.mean(labels == 0)*100:.1f}%)")
        print(f"  Epochs     : {epochs}  |  batch: {batch_size}  |  val: {val_split}")
        print(sep)

        os.makedirs(os.path.dirname(self.model_prefix) or ".", exist_ok=True)

        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=es_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=c.reduce_lr_factor,
                patience=lr_patience,
                min_lr=c.min_lr,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=f"{self.model_prefix}_best.keras",
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
        ]

        self.history = self.siamese_model.fit(
            [pairs_left, pairs_right],
            labels,
            validation_split=val_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )

        print(f"\n{sep}\n✓ TREINAMENTO CONCLUÍDO\n{sep}\n")
        return self.history

    # ------------------------------------------------------------------
    # INFERÊNCIA
    # ------------------------------------------------------------------

    def get_embedding(self, dataset: np.ndarray) -> np.ndarray:
        """
        Extrai o vetor de embedding de um dataset.

        Args:
            dataset: Array (target_samples, target_features, 1)

        Returns:
            Array 1D (embedding_dim,), L2-normalizado
        """
        if dataset.ndim == 3:
            dataset = np.expand_dims(dataset, axis=0)   # (1, S, F, 1)
        return self.encoder.predict(dataset, verbose=0)[0]

    def predict_distance(
        self,
        dataset_left:  np.ndarray,
        dataset_right: np.ndarray
    ) -> float:
        """
        Retorna a distância euclidiana entre dois datasets.

        0.0 = idênticos no espaço de embedding.
        """
        if dataset_left.ndim  == 3: dataset_left  = dataset_left[np.newaxis]
        if dataset_right.ndim == 3: dataset_right = dataset_right[np.newaxis]

        dist = self.siamese_model.predict([dataset_left, dataset_right], verbose=0)
        return float(dist[0][0])

    def predict_similarity(
        self,
        dataset_left:  np.ndarray,
        dataset_right: np.ndarray
    ) -> float:
        """
        Converte distância em similaridade usando exp(-d).
        Retorna valor em (0, 1]; quanto maior, mais similar.
        """
        d = self.predict_distance(dataset_left, dataset_right)
        return float(np.exp(-d))

    def compare_with_multiple(
        self,
        query:      np.ndarray,
        references: List[np.ndarray],
        names:      Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Compara um dataset query com uma lista de referências.

        Returns:
            Lista de (nome, similaridade) ordenada do mais para o menos similar.
        """
        if names is None:
            names = [f"Dataset_{i}" for i in range(len(references))]

        results = [
            (name, self.predict_similarity(query, ref))
            for name, ref in zip(names, references)
        ]
        return sorted(results, key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # SERIALIZAÇÃO
    # ------------------------------------------------------------------

    def save(self):
        """Salva encoder e modelo completo em CFG.paths.models/."""
        os.makedirs(os.path.dirname(self.model_prefix) or ".", exist_ok=True)
        self.encoder.save(f"{self.model_prefix}_encoder.keras")
        self.siamese_model.save(f"{self.model_prefix}_full.keras")
        print(f"✓ Modelos salvos em: {self.model_prefix}_*.keras")

    def load(self):
        """Carrega encoder e modelo completo de CFG.paths.models/."""
        self.encoder = keras.models.load_model(
            f"{self.model_prefix}_encoder.keras",
            custom_objects={"EuclideanDistance": EuclideanDistance}
        )
        self.siamese_model = keras.models.load_model(
            f"{self.model_prefix}_full.keras",
            custom_objects={
                "EuclideanDistance": EuclideanDistance,
                "loss_fn": contrastive_loss(self.margin)
            }
        )
        print(f"✓ Modelos carregados de: {self.model_prefix}_*.keras")

    def summary(self):
        print("\n" + "=" * 70 + "\nENCODER\n" + "=" * 70)
        self.encoder.summary()
        print("\n" + "=" * 70 + "\nSIAMESE MODEL\n" + "=" * 70)
        self.siamese_model.summary()


# ============================================================================
# TESTES
# ============================================================================

if __name__ == "__main__":
    from config import CFG, SiameseConfig

    print("\n" + "=" * 70)
    print("TESTE DA REDE SIAMESA")
    print("=" * 70)

    np.random.seed(42)
    n_pairs = 100
    shape = CFG.siamese.input_shape

    pairs_left  = np.random.randint(0, 2, size=(n_pairs, *shape)).astype(np.float32)
    pairs_right = np.random.randint(0, 2, size=(n_pairs, *shape)).astype(np.float32)
    labels      = np.random.randint(0, 2, size=(n_pairs,)).astype(np.float32)

    # ---- Teste 1: arquitetura default ----
    model = SiameseNet()
    history = model.train(pairs_left, pairs_right, labels, epochs=2, verbose=0)

    sample = np.random.randint(0, 2, size=shape).astype(np.float32)
    emb  = model.get_embedding(sample)
    dist = model.predict_distance(sample, sample)
    sim  = model.predict_similarity(sample, sample)

    print(f"  Embedding shape  : {emb.shape}")
    print(f"  L2 norm          : {np.linalg.norm(emb):.6f}  (esperado ≈ 1.0)")
    print(f"  Distância self   : {dist:.6f}  (esperado ≈ 0.0)")
    print(f"  Similaridade self: {sim:.6f}  (esperado ≈ 1.0)")

    assert emb.shape == (CFG.siamese.embedding_dim,), "Embedding com shape errado"
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5,    "Embedding não está L2-normalizado"
    assert dist < 0.01,                                "Distância consigo mesmo deve ser ≈ 0"

    # ---- Teste 2: compare_with_multiple ----
    refs  = [np.random.randint(0, 2, size=shape).astype(np.float32) for _ in range(4)]
    names = [f"Ref_{i}" for i in range(4)]
    results = model.compare_with_multiple(sample, refs, names)
    print(f"\n  Ranking (compare_with_multiple):")
    for name, score in results:
        print(f"    {name}: {score:.4f}")

    # ---- Teste 3: arquiteturas ----
    for arch in ['light', 'default', 'deep']:
        from config import SiameseConfig
        cfg_tmp = SiameseConfig(architecture=arch)
        m = SiameseNet(cfg=cfg_tmp)
        print(f"\n  [{arch:7s}] params: {m.siamese_model.count_params():,}")

    print("\n✓ TODOS OS TESTES PASSARAM!\n")