"""
Rede Neural Siamesa - Versão Otimizada
Comparação de similaridade entre datasets de malware Android
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, callbacks
import numpy as np
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


class SiameseNet:
    """
    Rede Siamesa com CNN para comparar datasets binários

    Arquitetura:
    - Encoder CNN compartilhado
    - Embedding normalizado L2
    - Distância euclidiana
    - Output: similaridade [0, 1]
    """

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (256, 100, 1),
            embedding_dim: int = 128,
            learning_rate: float = 0.0001,
            architecture: str = 'default'
    ):
        """
        Args:
            input_shape: Shape (samples, features, channels)
            embedding_dim: Dimensão do vetor embedding (64-256)
            learning_rate: Taxa de aprendizado
            architecture: 'default', 'deep', ou 'light'
        """
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.architecture = architecture

        self.encoder: Optional[Model] = None
        self.siamese_model: Optional[Model] = None
        self.history = None

        self._build_models()

    def _build_encoder_default(self, inputs: layers.Input) -> layers.Layer:
        """Arquitetura padrão (balanceada)"""
        # Bloco 1
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        # Bloco 2
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        # Bloco 3
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)

        # Dense
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        return x

    def _build_encoder_deep(self, inputs: layers.Input) -> layers.Layer:
        """Arquitetura profunda (mais capacidade)"""
        # 4 blocos convolucionais
        x = inputs
        for filters in [32, 64, 128, 256]:
            x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)

        # Dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.4)(x)

        return x

    def _build_encoder_light(self, inputs: layers.Input) -> layers.Layer:
        """Arquitetura leve (mais rápida)"""
        # 2 blocos convolucionais
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)

        # Dense
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        return x

    def _build_models(self):
        """Constrói encoder e modelo siamês"""
        print(f"\n{'=' * 70}")
        print(f"CONSTRUINDO REDE SIAMESA ({self.architecture.upper()})")
        print(f"{'=' * 70}")

        # Input
        inputs = layers.Input(shape=self.input_shape)

        # Encoder (baseado na arquitetura escolhida)
        if self.architecture == 'deep':
            x = self._build_encoder_deep(inputs)
        elif self.architecture == 'light':
            x = self._build_encoder_light(inputs)
        else:
            x = self._build_encoder_default(inputs)

        # Embedding layer
        embeddings = layers.Dense(self.embedding_dim, name='embeddings')(x)

        # Normalização L2 (importante para distância euclidiana)
        embeddings = layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1),
            name='l2_normalize'
        )(embeddings)

        # Encoder model
        self.encoder = Model(inputs=inputs, outputs=embeddings, name='encoder')

        print(f"\n✓ Encoder criado:")
        print(f"  Input: {self.input_shape}")
        print(f"  Output: {self.embedding_dim}")
        print(f"  Parâmetros: {self.encoder.count_params():,}")

        # Modelo Siamês
        input_left = layers.Input(shape=self.input_shape, name='input_left')
        input_right = layers.Input(shape=self.input_shape, name='input_right')

        # Embeddings (pesos compartilhados)
        embedding_left = self.encoder(input_left)
        embedding_right = self.encoder(input_right)

        # Distância euclidiana
        distance = layers.Lambda(
            lambda tensors: tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True)),
            name='euclidean_distance'
        )([embedding_left, embedding_right])

        # Similaridade (inverte distância)
        # Usa sigmoid: quanto menor distância, maior similaridade
        similarity = layers.Dense(1, activation='sigmoid', name='similarity')(distance)

        self.siamese_model = Model(
            inputs=[input_left, input_right],
            outputs=similarity,
            name='siamese_network'
        )

        # Compile
        self.siamese_model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )

        print(f"\n✓ Modelo Siamês criado:")
        print(f"  Parâmetros totais: {self.siamese_model.count_params():,}")
        print(f"{'=' * 70}\n")

    def train(
            self,
            pairs_left: np.ndarray,
            pairs_right: np.ndarray,
            labels: np.ndarray,
            validation_split: float = 0.2,
            epochs: int = 50,
            batch_size: int = 32,
            early_stopping_patience: int = 10,
            reduce_lr_patience: int = 5,
            verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Treina a rede siamesa

        Args:
            pairs_left: Array (n_pairs, *input_shape)
            pairs_right: Array (n_pairs, *input_shape)
            labels: Array (n_pairs,) - 1=similar, 0=diferente
            validation_split: Proporção para validação
            epochs: Número de épocas
            batch_size: Tamanho do batch
            early_stopping_patience: Paciência para early stopping
            reduce_lr_patience: Paciência para redução de LR
            verbose: Verbosidade (0, 1, ou 2)

        Returns:
            History object do Keras
        """
        print(f"\n{'=' * 70}")
        print(f"TREINAMENTO")
        print(f"{'=' * 70}")
        print(f"Pares: {len(labels)}")
        print(f"Similar: {np.sum(labels == 1)} ({np.mean(labels == 1) * 100:.1f}%)")
        print(f"Diferente: {np.sum(labels == 0)} ({np.mean(labels == 0) * 100:.1f}%)")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Validation split: {validation_split}")
        print(f"{'=' * 70}\n")

        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]

        # Treina
        self.history = self.siamese_model.fit(
            [pairs_left, pairs_right],
            labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )

        print(f"\n{'=' * 70}")
        print(f"✓ TREINAMENTO CONCLUÍDO")
        print(f"{'=' * 70}\n")

        return self.history

    def predict_similarity(
            self,
            dataset_left: np.ndarray,
            dataset_right: np.ndarray
    ) -> float:
        """
        Prediz similaridade entre dois datasets

        Args:
            dataset_left: Array shape (256, 100, 1)
            dataset_right: Array shape (256, 100, 1)

        Returns:
            Similaridade [0, 1] - 1=muito similar, 0=muito diferente
        """
        # Adiciona batch dimension se necessário
        if len(dataset_left.shape) == 3:
            dataset_left = np.expand_dims(dataset_left, axis=0)
        if len(dataset_right.shape) == 3:
            dataset_right = np.expand_dims(dataset_right, axis=0)

        similarity = self.siamese_model.predict(
            [dataset_left, dataset_right],
            verbose=0
        )

        return float(similarity[0][0])

    def get_embedding(self, dataset: np.ndarray) -> np.ndarray:
        """
        Extrai embedding de um dataset

        Args:
            dataset: Array shape (256, 100, 1)

        Returns:
            Embedding shape (embedding_dim,)
        """
        if len(dataset.shape) == 3:
            dataset = np.expand_dims(dataset, axis=0)

        embedding = self.encoder.predict(dataset, verbose=0)
        return embedding[0]

    def compare_with_multiple(
            self,
            query: np.ndarray,
            references: List[np.ndarray],
            names: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Compara query com múltiplos datasets de referência

        Args:
            query: Dataset query
            references: Lista de datasets de referência
            names: Nomes dos datasets (opcional)

        Returns:
            Lista de (nome, similaridade) ordenada por similaridade
        """
        if names is None:
            names = [f"Dataset_{i}" for i in range(len(references))]

        results = []
        for name, ref in zip(names, references):
            sim = self.predict_similarity(query, ref)
            results.append((name, sim))

        # Ordena por similaridade (maior primeiro)
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def save(self, filepath: str = 'siamese_model'):
        """Salva modelos"""
        self.encoder.save(f'{filepath}_encoder.keras')
        self.siamese_model.save(f'{filepath}_full.keras')
        print(f"✓ Modelos salvos: {filepath}_*.keras")

    def load(self, filepath: str = 'siamese_model'):
        """Carrega modelos"""
        self.encoder = keras.models.load_model(f'{filepath}_encoder.keras')
        self.siamese_model = keras.models.load_model(f'{filepath}_full.keras')
        print(f"✓ Modelos carregados: {filepath}_*.keras")

    def summary(self):
        """Mostra resumo dos modelos"""
        print("\n" + "=" * 70)
        print("ENCODER")
        print("=" * 70)
        self.encoder.summary()

        print("\n" + "=" * 70)
        print("SIAMESE MODEL")
        print("=" * 70)
        self.siamese_model.summary()


# ============================================================================
# TESTES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTE DA REDE SIAMESA V2")
    print("=" * 70)

    # Dados de teste
    np.random.seed(42)

    # Simula pares
    n_pairs = 200
    pairs_left = np.random.randint(0, 2, size=(n_pairs, 256, 100, 1)).astype(np.float32)
    pairs_right = np.random.randint(0, 2, size=(n_pairs, 256, 100, 1)).astype(np.float32)
    labels = np.random.randint(0, 2, size=(n_pairs,)).astype(np.float32)

    print(f"\n📊 Dados de teste:")
    print(f"  Pairs left: {pairs_left.shape}")
    print(f"  Pairs right: {pairs_right.shape}")
    print(f"  Labels: {labels.shape}")

    # ========================================================================
    # TESTE 1: Arquitetura Default
    # ========================================================================
    print("\n" + "=" * 70)
    print("TESTE 1: ARQUITETURA DEFAULT")
    print("=" * 70)

    model_default = SiameseNet(
        input_shape=(256, 100, 1),
        embedding_dim=128,
        architecture='default'
    )

    # Treino rápido
    history = model_default.train(
        pairs_left, pairs_right, labels,
        epochs=2,
        batch_size=32,
        verbose=0
    )

    # Teste de predição
    test_left = np.random.randint(0, 2, size=(256, 100, 1)).astype(np.float32)
    test_right = np.random.randint(0, 2, size=(256, 100, 1)).astype(np.float32)

    similarity = model_default.predict_similarity(test_left, test_right)
    print(f"✓ Similaridade: {similarity:.4f}")

    embedding = model_default.get_embedding(test_left)
    print(f"✓ Embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")

    # ========================================================================
    # TESTE 2: Comparação Múltipla
    # ========================================================================
    print("\n" + "=" * 70)
    print("TESTE 2: COMPARAÇÃO MÚLTIPLA")
    print("=" * 70)

    query = test_left
    references = [
        np.random.randint(0, 2, size=(256, 100, 1)).astype(np.float32)
        for _ in range(5)
    ]
    names = [f"Malware_Type_{i}" for i in range(5)]

    results = model_default.compare_with_multiple(query, references, names)

    print("\n✓ Ranking de similaridade:")
    for name, score in results:
        print(f"  {name}: {score:.4f}")

    # ========================================================================
    # TESTE 3: Arquiteturas Alternativas
    # ========================================================================
    print("\n" + "=" * 70)
    print("TESTE 3: COMPARAÇÃO DE ARQUITETURAS")
    print("=" * 70)

    for arch in ['light', 'default', 'deep']:
        model = SiameseNet(
            input_shape=(256, 100, 1),
            embedding_dim=128,
            architecture=arch
        )
        print(f"\n{arch.upper()}:")
        print(f"  Parâmetros: {model.siamese_model.count_params():,}")

    print("\n" + "=" * 70)
    print("✓ TODOS OS TESTES PASSARAM!")
    print("=" * 70 + "\n")