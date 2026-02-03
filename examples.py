"""
EXEMPLO COMPLETO: Sistema de Detecção de Malware Android
Usando Rede Siamesa com Padronização Automática
"""

import numpy as np
from standardizer import DatasetStandardizer, create_comparison_pairs
from siamese import SiameseNet


# ============================================================================
# EXEMPLO 1: WORKFLOW COMPLETO DE TREINO
# ============================================================================

def example_1_complete_training():
    """
    Exemplo completo: desde datasets brutos até modelo treinado
    """
    print("\n" + "=" * 80)
    print("EXEMPLO 1: WORKFLOW COMPLETO DE TREINO")
    print("=" * 80)

    # ========================================================================
    # PASSO 1: Preparar Dados
    # ========================================================================
    print("\n📁 PASSO 1: Carregando datasets de malware...")

    # Simula 10 datasets de malware com tamanhos variados
    # Em uso real, você carregaria de CSVs:
    # raw_datasets = [pd.read_csv(f'malware_{i}.csv').values for i in range(10)]

    np.random.seed(42)
    raw_datasets = []
    labels = []

    malware_types = {
        0: "Banking Trojan",
        1: "Spyware",
        2: "Ransomware"
    }

    for i in range(10):
        # Tamanhos variados (como na vida real)
        n_samples = np.random.randint(500, 2000)
        n_features = np.random.randint(30, 100)

        dataset = np.random.randint(0, 2, size=(n_samples, n_features))
        raw_datasets.append(dataset)

        label = i % 3  # 3 tipos de malware
        labels.append(label)

        print(f"  Dataset {i + 1:2d}: {dataset.shape} | Tipo: {malware_types[label]}")

    labels = np.array(labels)

    # ========================================================================
    # PASSO 2: Padronizar Datasets
    # ========================================================================
    print("\n🔧 PASSO 2: Padronizando datasets...")

    standardizer = DatasetStandardizer(
        target_samples=256,
        target_features=100,
        use_pca=True  # Usa PCA para redução inteligente
    )

    # Padroniza todos datasets de uma vez
    datasets_standardized = standardizer.fit_transform_batch(raw_datasets)

    print(f"\n✓ Datasets padronizados: {datasets_standardized.shape}")

    # ========================================================================
    # PASSO 3: Criar Pares de Treino
    # ========================================================================
    print("\n🔀 PASSO 3: Criando pares para treinamento...")

    pairs_left, pairs_right, similarity_labels = create_comparison_pairs(
        datasets_standardized,
        labels,
        n_pairs=5000,  # Quanto mais pares, melhor
        balance_ratio=0.5  # 50% similar, 50% diferente
    )

    print(f"\n✓ Pares criados:")
    print(f"  Total: {len(similarity_labels)}")
    print(f"  Similar: {np.sum(similarity_labels == 1)}")
    print(f"  Diferente: {np.sum(similarity_labels == 0)}")

    # ========================================================================
    # PASSO 4: Criar e Treinar Rede Siamesa
    # ========================================================================
    print("\n🧠 PASSO 4: Criando e treinando rede siamesa...")

    siamese = SiameseNet(
        input_shape=(256, 100, 1),
        embedding_dim=128,
        learning_rate=0.0001,
        architecture='default'  # Opções: 'light', 'default', 'deep'
    )

    # Treina
    history = siamese.train(
        pairs_left, pairs_right, similarity_labels,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        early_stopping_patience=10
    )

    # ========================================================================
    # PASSO 5: Salvar Modelo
    # ========================================================================
    print("\n💾 PASSO 5: Salvando modelo...")

    siamese.save('malware_detector')

    print("\n" + "=" * 80)
    print("✓ TREINAMENTO CONCLUÍDO!")
    print("=" * 80)
    print("\nArquivos gerados:")
    print("  • malware_detector_encoder.keras")
    print("  • malware_detector_full.keras")
    print("  • best_model.keras")

    return standardizer, siamese, datasets_standardized, labels


# ============================================================================
# EXEMPLO 2: DETECTAR MALWARE EM NOVO APP
# ============================================================================

def example_2_detect_malware(standardizer, siamese, reference_datasets, reference_labels):
    """
    Exemplo: Analisar um novo app suspeito
    """
    print("\n" + "=" * 80)
    print("EXEMPLO 2: DETECÇÃO DE MALWARE EM NOVO APP")
    print("=" * 80)

    # ========================================================================
    # Novo app suspeito
    # ========================================================================
    print("\n📱 Carregando app suspeito...")

    # Em uso real:
    # new_app = pd.read_csv('suspicious_app.csv').values

    # Simulação
    new_app = np.random.randint(0, 2, size=(850, 65))
    print(f"  Shape original: {new_app.shape}")

    # ========================================================================
    # Padroniza
    # ========================================================================
    print("\n🔧 Padronizando app...")
    new_app_std = standardizer.transform(new_app)

    # ========================================================================
    # Compara com base de conhecimento
    # ========================================================================
    print("\n🔍 Comparando com base de conhecimento...")

    malware_types = {
        0: "Banking Trojan",
        1: "Spyware",
        2: "Ransomware"
    }

    reference_names = [
        f"{malware_types[label]}_v{i}"
        for i, label in enumerate(reference_labels)
    ]

    results = siamese.compare_with_multiple(
        new_app_std,
        reference_datasets,
        reference_names
    )

    # ========================================================================
    # Analisa resultados
    # ========================================================================
    print("\n" + "-" * 80)
    print("RESULTADOS DA ANÁLISE")
    print("-" * 80)

    threshold = 0.7  # Ajuste conforme necessário

    print(f"\n📊 Ranking de similaridade (threshold={threshold}):\n")

    is_malware = False
    malware_matches = []

    for name, score in results:
        status = "⚠️  MATCH" if score > threshold else "✓ OK"
        print(f"  {name:30s} | {score:.4f} | {status}")

        if score > threshold:
            is_malware = True
            malware_matches.append((name, score))

    # ========================================================================
    # Veredicto final
    # ========================================================================
    print("\n" + "=" * 80)
    if is_malware:
        print("⚠️  ALERTA: APP SUSPEITO DE MALWARE!")
        print("=" * 80)
        print("\nMatches encontrados:")
        for name, score in malware_matches:
            print(f"  • {name}: {score * 100:.1f}% de similaridade")
        print(f"\nMelhor match: {results[0][0]} ({results[0][1] * 100:.1f}%)")
    else:
        print("✓ APP APROVADO")
        print("=" * 80)
        print("\nO app NÃO é similar a malwares conhecidos.")
        print(f"Maior similaridade: {results[0][0]} ({results[0][1] * 100:.1f}%)")

    print("\n" + "=" * 80)


# ============================================================================
# EXEMPLO 3: ANÁLISE EM LOTE
# ============================================================================

def example_3_batch_analysis(standardizer, siamese, reference_datasets, reference_names):
    """
    Exemplo: Analisar múltiplos apps de uma vez
    """
    print("\n" + "=" * 80)
    print("EXEMPLO 3: ANÁLISE EM LOTE DE APPS")
    print("=" * 80)

    # ========================================================================
    # Simula lote de apps
    # ========================================================================
    print("\n📱 Carregando lote de apps...")

    apps = {
        'app_game.apk': np.random.randint(0, 2, size=(600, 45)),
        'app_banking.apk': np.random.randint(0, 2, size=(1200, 75)),
        'app_social.apk': np.random.randint(0, 2, size=(800, 55)),
        'app_utility.apk': np.random.randint(0, 2, size=(500, 40)),
        'app_photo.apk': np.random.randint(0, 2, size=(950, 68)),
    }

    print(f"  Total de apps: {len(apps)}")

    # ========================================================================
    # Analisa cada app
    # ========================================================================
    print("\n🔍 Analisando apps...\n")

    threshold = 0.7
    results_summary = []

    for app_name, app_data in apps.items():
        # Padroniza
        app_std = standardizer.transform(app_data)

        # Compara
        comparisons = siamese.compare_with_multiple(
            app_std,
            reference_datasets,
            reference_names
        )

        # Verifica se é malware
        best_match, best_score = comparisons[0]
        is_malware = best_score > threshold

        results_summary.append({
            'app': app_name,
            'is_malware': is_malware,
            'best_match': best_match,
            'score': best_score
        })

        status = "⚠️  MALWARE" if is_malware else "✓ LIMPO"
        print(f"  {app_name:20s} | {status:12s} | {best_match:25s} | {best_score:.4f}")

    # ========================================================================
    # Resumo
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESUMO DA ANÁLISE")
    print("=" * 80)

    total = len(results_summary)
    malware_count = sum(1 for r in results_summary if r['is_malware'])
    clean_count = total - malware_count

    print(f"\n  Total analisado: {total}")
    print(f"  Apps maliciosos: {malware_count} ({malware_count / total * 100:.1f}%)")
    print(f"  Apps limpos:     {clean_count} ({clean_count / total * 100:.1f}%)")

    if malware_count > 0:
        print("\n  ⚠️  Apps maliciosos detectados:")
        for r in results_summary:
            if r['is_malware']:
                print(f"    • {r['app']}: {r['score'] * 100:.1f}% similar a {r['best_match']}")

    print("\n" + "=" * 80)


# ============================================================================
# EXEMPLO 4: ANÁLISE DE EMBEDDINGS
# ============================================================================

def example_4_embedding_analysis(siamese, datasets_standardized, labels):
    """
    Exemplo: Análise de embeddings e clustering
    """
    print("\n" + "=" * 80)
    print("EXEMPLO 4: ANÁLISE DE EMBEDDINGS")
    print("=" * 80)

    print("\n🧠 Extraindo embeddings...")

    embeddings = []
    for i, dataset in enumerate(datasets_standardized):
        emb = siamese.get_embedding(dataset)
        embeddings.append(emb)
        print(f"  Dataset {i + 1}: embedding shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")

    embeddings = np.array(embeddings)

    # ========================================================================
    # Distâncias entre embeddings
    # ========================================================================
    print("\n📊 Matriz de distâncias:\n")

    from scipy.spatial.distance import cdist

    distances = cdist(embeddings, embeddings, metric='euclidean')

    print("     ", end="")
    for i in range(len(distances)):
        print(f" D{i + 1:2d}", end="")
    print()

    for i, row in enumerate(distances):
        print(f"  D{i + 1:2d}", end="")
        for val in row:
            if val == 0:
                print("  --", end="")
            else:
                print(f" {val:4.2f}", end="")
        print(f"  | Label: {labels[i]}")

    # ========================================================================
    # Análise por classe
    # ========================================================================
    print("\n📊 Distâncias médias por tipo:")

    malware_types = {
        0: "Banking Trojan",
        1: "Spyware",
        2: "Ransomware"
    }

    for label_type, name in malware_types.items():
        # Indices da classe
        indices = np.where(labels == label_type)[0]

        if len(indices) > 1:
            # Distância intra-classe (mesma classe)
            intra_dists = []
            for i in indices:
                for j in indices:
                    if i < j:
                        intra_dists.append(distances[i, j])

            # Distância inter-classe (classes diferentes)
            inter_dists = []
            for i in indices:
                for j in range(len(labels)):
                    if labels[j] != label_type:
                        inter_dists.append(distances[i, j])

            print(f"\n  {name}:")
            print(f"    Intra-classe (mesma): {np.mean(intra_dists):.4f} ± {np.std(intra_dists):.4f}")
            print(f"    Inter-classe (diff):  {np.mean(inter_dists):.4f} ± {np.std(inter_dists):.4f}")

    print("\n" + "=" * 80)


# ============================================================================
# MAIN: EXECUTA TODOS OS EXEMPLOS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SISTEMA DE DETECÇÃO DE MALWARE ANDROID")
    print("Rede Siamesa com Padronização Automática")
    print("=" * 80)

    # ========================================================================
    # Exemplo 1: Treino Completo
    # ========================================================================
    standardizer, siamese, datasets_std, labels = example_1_complete_training()

    # ========================================================================
    # Exemplo 2: Detectar Malware
    # ========================================================================
    malware_types = {
        0: "Banking Trojan",
        1: "Spyware",
        2: "Ransomware"
    }

    reference_names = [
        f"{malware_types[label]}_v{i}"
        for i, label in enumerate(labels)
    ]

    example_2_detect_malware(
        standardizer,
        siamese,
        datasets_std,
        labels
    )

    # ========================================================================
    # Exemplo 3: Análise em Lote
    # ========================================================================
    example_3_batch_analysis(
        standardizer,
        siamese,
        datasets_std,
        reference_names
    )

    # ========================================================================
    # Exemplo 4: Análise de Embeddings
    # ========================================================================
    example_4_embedding_analysis(
        siamese,
        datasets_std,
        labels
    )

    # ========================================================================
    # RESUMO FINAL
    # ========================================================================
    print("\n" + "=" * 80)
    print("✓ TODOS OS EXEMPLOS EXECUTADOS COM SUCESSO!")
    print("=" * 80)
    print("\n📚 PRÓXIMOS PASSOS:\n")
    print("1. Substitua os dados simulados pelos seus datasets reais")
    print("2. Ajuste os hiperparâmetros conforme necessário:")
    print("   - target_samples, target_features no DatasetStandardizer")
    print("   - embedding_dim, architecture no SiameseNet")
    print("   - epochs, batch_size no treinamento")
    print("3. Experimente diferentes thresholds de detecção")
    print("4. Use validação cruzada para avaliar performance")
    print("5. Implemente aprendizado incremental para novos malwares")
    print("\n" + "=" * 80 + "\n")