"""
SCRIPT DE TESTE COMPLETO
Testa todos os componentes do sistema passo a passo
"""

import numpy as np
import sys

print("\n" + "=" * 80)
print("TESTE COMPLETO DO SISTEMA DE DETECÇÃO DE MALWARE")
print("=" * 80)

# ============================================================================
# TESTE 1: Importações
# ============================================================================
print("\n" + "=" * 80)
print("TESTE 1: VERIFICANDO IMPORTAÇÕES")
print("=" * 80)

try:
    from standardizer import DatasetStandardizer, create_comparison_pairs

    print("✓ standardizer.py importado com sucesso")
except Exception as e:
    print(f"✗ ERRO ao importar standardizer: {e}")
    sys.exit(1)

try:
    import tensorflow as tf

    print(f"✓ TensorFlow {tf.__version__} importado com sucesso")

    from siamese import SiameseNet

    print("✓ siamese.py importado com sucesso")
except Exception as e:
    print(f"✗ ERRO ao importar siamese: {e}")
    print("\n💡 DICA: Instale TensorFlow com:")
    print("   pip install tensorflow numpy scikit-learn")
    print("\nContinuando apenas com testes do standardizer...\n")
    TENSORFLOW_AVAILABLE = False
else:
    TENSORFLOW_AVAILABLE = True

# ============================================================================
# TESTE 2: DatasetStandardizer
# ============================================================================
print("\n" + "=" * 80)
print("TESTE 2: DATASET STANDARDIZER")
print("=" * 80)

print("\n📊 Criando datasets de teste com tamanhos variados...")
np.random.seed(42)

# Simula 5 datasets com tamanhos diferentes
test_datasets = []
test_labels = []

configs = [
    (150, 50, "Pequeno"),
    (500, 120, "Grande"),
    (300, 80, "Médio"),
    (800, 45, "Largo"),
    (600, 95, "Alto")
]

for i, (n_samples, n_features, desc) in enumerate(configs):
    dataset = np.random.randint(0, 2, size=(n_samples, n_features))
    test_datasets.append(dataset)
    test_labels.append(i % 3)  # 3 classes: 0, 1, 2
    print(f"  Dataset {i + 1}: {str(dataset.shape):15s} | {desc:10s} | Classe: {i % 3}")

test_labels = np.array(test_labels)

# Teste do standardizer
print("\n🔧 Testando DatasetStandardizer...")

try:
    standardizer = DatasetStandardizer(
        target_samples=256,
        target_features=100,
        use_pca=True
    )
    print("✓ Standardizer criado")

    # Batch transform
    datasets_std = standardizer.fit_transform_batch(test_datasets, show_progress=False)
    print(f"✓ Datasets padronizados: {datasets_std.shape}")

    # Verifica binaridade
    unique_vals = np.unique(datasets_std)
    is_binary = set(unique_vals) == {0.0, 1.0}

    if is_binary:
        print("✓ Binaridade preservada (apenas 0s e 1s)")
    else:
        print(f"✗ ERRO: Valores encontrados: {unique_vals}")
        sys.exit(1)

    # Verifica shape
    expected_shape = (5, 256, 100, 1)
    if datasets_std.shape == expected_shape:
        print(f"✓ Shape correto: {expected_shape}")
    else:
        print(f"✗ ERRO: Shape esperado {expected_shape}, obtido {datasets_std.shape}")
        sys.exit(1)

except Exception as e:
    print(f"✗ ERRO no standardizer: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TESTE 3: Criação de Pares
# ============================================================================
print("\n" + "=" * 80)
print("TESTE 3: CRIAÇÃO DE PARES DE TREINO")
print("=" * 80)

try:
    print("\n🔀 Criando pares de comparação...")

    pairs_left, pairs_right, similarity = create_comparison_pairs(
        datasets_std,
        test_labels,
        n_pairs=100,
        balance_ratio=0.5
    )

    print(f"✓ Pares criados:")
    print(f"  Pairs left:  {pairs_left.shape}")
    print(f"  Pairs right: {pairs_right.shape}")
    print(f"  Labels:      {similarity.shape}")

    n_similar = np.sum(similarity == 1)
    n_different = np.sum(similarity == 0)

    print(f"\n✓ Balanceamento:")
    print(f"  Similar:    {n_similar} ({n_similar / len(similarity) * 100:.1f}%)")
    print(f"  Diferente:  {n_different} ({n_different / len(similarity) * 100:.1f}%)")

    # Verifica se há pares
    if len(pairs_left) > 0:
        print("✓ Pares gerados com sucesso")
    else:
        print("✗ ERRO: Nenhum par gerado")
        sys.exit(1)

except Exception as e:
    print(f"✗ ERRO na criação de pares: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TESTE 4: Rede Siamesa (se TensorFlow disponível)
# ============================================================================
if TENSORFLOW_AVAILABLE:
    print("\n" + "=" * 80)
    print("TESTE 4: REDE NEURAL SIAMESA")
    print("=" * 80)

    try:
        print("\n🧠 Criando rede siamesa...")

        siamese = SiameseNet(
            input_shape=(256, 100, 1),
            embedding_dim=64,  # Menor para teste rápido
            architecture='light'  # Arquitetura leve
        )
        print("✓ Rede siamesa criada")

        # Teste de treino rápido (2 épocas)
        print("\n🏋️ Testando treino (2 épocas, modo silencioso)...")

        history = siamese.train(
            pairs_left[:50],  # Apenas 50 pares para teste rápido
            pairs_right[:50],
            similarity[:50],
            validation_split=0.2,
            epochs=2,
            batch_size=16,
            verbose=0
        )
        print("✓ Treino executado com sucesso")

        # Teste de predição
        print("\n🔍 Testando predição...")

        test_dataset_1 = datasets_std[0]
        test_dataset_2 = datasets_std[1]

        sim_score = siamese.predict_similarity(test_dataset_1, test_dataset_2)
        print(f"✓ Similaridade calculada: {sim_score:.4f}")

        if 0 <= sim_score <= 1:
            print("✓ Score no intervalo correto [0, 1]")
        else:
            print(f"✗ ERRO: Score fora do intervalo: {sim_score}")
            sys.exit(1)

        # Teste de embedding
        print("\n📊 Testando extração de embedding...")

        embedding = siamese.get_embedding(test_dataset_1)
        print(f"✓ Embedding extraído: shape={embedding.shape}")

        if embedding.shape == (64,):
            print("✓ Dimensão do embedding correta")
        else:
            print(f"✗ ERRO: Dimensão esperada (64,), obtida {embedding.shape}")
            sys.exit(1)

        # Teste de comparação múltipla
        print("\n🔎 Testando comparação com múltiplos datasets...")

        query = datasets_std[0]
        references = list(datasets_std[1:])
        names = [f"Dataset_{i}" for i in range(1, 5)]

        results = siamese.compare_with_multiple(query, references, names)
        print(f"✓ Comparação realizada com {len(results)} datasets")

        print("\n  Ranking:")
        for name, score in results[:3]:
            print(f"    {name}: {score:.4f}")

    except Exception as e:
        print(f"✗ ERRO na rede siamesa: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

else:
    print("\n" + "=" * 80)
    print("TESTE 4: REDE SIAMESA - PULADO (TensorFlow não disponível)")
    print("=" * 80)

# ============================================================================
# TESTE 5: Transform de novo dataset
# ============================================================================
print("\n" + "=" * 80)
print("TESTE 5: TRANSFORMAÇÃO DE NOVO DATASET")
print("=" * 80)

try:
    print("\n📱 Simulando novo dataset (app suspeito)...")

    new_dataset = np.random.randint(0, 2, size=(700, 65))
    print(f"  Dataset original: {new_dataset.shape}")

    new_dataset_std = standardizer.transform(new_dataset)
    print(f"✓ Dataset transformado: {new_dataset_std.shape}")

    if new_dataset_std.shape == (256, 100, 1):
        print("✓ Shape correto após transformação")
    else:
        print(f"✗ ERRO: Shape incorreto")
        sys.exit(1)

except Exception as e:
    print(f"✗ ERRO na transformação: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# RESUMO FINAL
# ============================================================================
print("\n" + "=" * 80)
print("RESUMO DOS TESTES")
print("=" * 80)

tests_passed = [
    "✓ Importações",
    "✓ DatasetStandardizer",
    "✓ Criação de pares",
    "✓ Transform de novos datasets"
]

if TENSORFLOW_AVAILABLE:
    tests_passed.append("✓ Rede Siamesa")
else:
    tests_passed.append("⚠ Rede Siamesa (TensorFlow não instalado)")

print("\n" + "\n".join(tests_passed))

print("\n" + "=" * 80)
print("🎉 TODOS OS TESTES PASSARAM COM SUCESSO!")
print("=" * 80)

print("\n📚 PRÓXIMOS PASSOS:\n")
print("1. Substitua os dados simulados pelos seus datasets reais")
print("2. Execute o treinamento completo:")
print("   python examples.py")
print("3. Ajuste hiperparâmetros conforme necessário")
print("4. Veja README.md para mais exemplos e documentação")

if not TENSORFLOW_AVAILABLE:
    print("\n⚠️  ATENÇÃO: TensorFlow não está instalado!")
    print("   Instale com: pip install tensorflow")
    print("   Ou: pip install tensorflow numpy scikit-learn")

print("\n" + "=" * 80 + "\n")