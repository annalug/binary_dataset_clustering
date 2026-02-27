"""
pair_generator.py — Gerador de pares para treinamento da rede siamesa

Correção principal:
  - Os labels usados para definir "similar" devem ser TIPOS DE DATASET
    (ex: 0=permissions, 1=api_calls, 2=network), não os labels de classe
    malware/benigno dos exemplos dentro de cada dataset.
    A rede aprende a distinguir ESTRUTURA de dataset, não malware vs benigno.

  - Adicionada função build_dataset_type_labels() que cria os labels de tipo
    a partir do dataset_type_map definido em CFG.pairs.

  - Parâmetros lidos de CFG.pairs por padrão.
  - Seeds locais (np.random.default_rng) para reprodutibilidade sem
    contaminar o estado global do numpy.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict

from config import CFG, PairConfig


# ============================================================================
# HELPER: labels de tipo de dataset
# ============================================================================

def build_dataset_type_labels(
    names: List[str],
    type_map: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """
    Constrói um array de labels de TIPO DE DATASET (não de malware/benigno).

    O tipo reflete a natureza das features do dataset, por exemplo:
      0 → permissions
      1 → API calls
      2 → network features
      3 → graph-based features

    Se um nome não estiver no mapa, recebe um tipo único (não agrupado).

    Args:
        names:    Lista de nomes de datasets (mesma ordem de `datasets`).
        type_map: Dicionário {nome_dataset: tipo_int}.
                  Se None, usa CFG.pairs.dataset_type_map.
                  Se vazio, cada dataset recebe seu próprio tipo (nenhum par similar).

    Returns:
        Array (n_datasets,) com inteiros representando o tipo de cada dataset.

    Exemplo:
        type_map = {
            "balanced_adroit":              0,   # permissions
            "balanced_androcrawl":          0,   # permissions
            "balanced_android_permissions": 0,   # permissions
            "balanced_drebin215":           1,   # API calls
            "balanced_defensedroid_prs":    1,   # API calls
            "balanced_kronodroid_emulator": 2,   # mixed
            "balanced_kronodroid_real":     2,   # mixed
        }
    """
    tmap = type_map if type_map is not None else CFG.pairs.dataset_type_map

    # Tipos usados; datasets sem mapeamento recebem IDs únicos
    max_known = max(tmap.values(), default=-1)
    next_id = max_known + 1

    label_arr = []
    for name in names:
        if name in tmap:
            label_arr.append(tmap[name])
        else:
            label_arr.append(next_id)
            next_id += 1

    return np.array(label_arr, dtype=np.int32)


# ============================================================================
# GERADOR DE PARES PRINCIPAL
# ============================================================================

def create_comparison_pairs(
    datasets:      np.ndarray,
    dataset_labels: np.ndarray,
    n_pairs:       Optional[int]   = None,
    balance_ratio: Optional[float] = None,
    random_state:  Optional[int]   = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cria pares (left, right, similarity) para treinamento siamês.

    ATENÇÃO: `dataset_labels` deve conter TIPOS DE DATASET (ver
    build_dataset_type_labels), não labels malware/benigno.

    Dois datasets são "similares" (label=1) quando têm o mesmo tipo.
    Dois datasets são "diferentes" (label=0) quando têm tipos distintos.

    Args:
        datasets:       Array (n_datasets, H, W, C) — datasets padronizados
        dataset_labels: Array (n_datasets,)          — tipo de cada dataset
        n_pairs:        Total de pares a gerar (padrão: CFG.pairs.n_pairs)
        balance_ratio:  Fração de pares similares (padrão: CFG.pairs.balance_ratio)
        random_state:   Seed (padrão: CFG.pairs.random_state)

    Returns:
        pairs_left:  Array (n_pairs, H, W, C)
        pairs_right: Array (n_pairs, H, W, C)
        similarity:  Array (n_pairs,)  — 1.0 = similar, 0.0 = diferente
    """
    n_pairs       = n_pairs       or CFG.pairs.n_pairs
    balance_ratio = balance_ratio if balance_ratio is not None else CFG.pairs.balance_ratio
    seed          = random_state  if random_state  is not None else CFG.pairs.random_state

    if len(datasets) < 2:
        raise ValueError("São necessários pelo menos 2 datasets para criar pares.")

    rng        = np.random.default_rng(seed)
    n_similar  = int(n_pairs * balance_ratio)
    n_different = n_pairs - n_similar

    print(f"\n🔀 Gerando {n_pairs} pares de datasets:")
    print(f"   Similares  (mesmo tipo) : {n_similar}")
    print(f"   Diferentes (tipo distinto): {n_different}")

    unique_labels = np.unique(dataset_labels)

    # Grupos com ≥ 2 membros (podem formar pares similares)
    similar_groups = [
        np.where(dataset_labels == lbl)[0]
        for lbl in unique_labels
        if np.sum(dataset_labels == lbl) >= 2
    ]
    # Todos os grupos com ≥ 1 membro
    all_groups = {lbl: np.where(dataset_labels == lbl)[0] for lbl in unique_labels}

    pairs_left, pairs_right, sim_labels = [], [], []

    # --- Pares similares ---
    if not similar_groups:
        print("  ⚠ Nenhum tipo de dataset tem ≥ 2 membros. "
              "Nenhum par similar será gerado. "
              "Preencha CFG.pairs.dataset_type_map corretamente.")
        n_different = n_pairs   # converte todos em diferentes
        n_similar   = 0

    count = 0
    max_tries = n_similar * 20
    tries = 0
    while count < n_similar and tries < max_tries:
        tries += 1
        group = similar_groups[rng.integers(len(similar_groups))]
        i1, i2 = rng.choice(group, size=2, replace=False)
        pairs_left.append(datasets[i1])
        pairs_right.append(datasets[i2])
        sim_labels.append(1.0)
        count += 1

    if count < n_similar:
        print(f"  ⚠ Apenas {count}/{n_similar} pares similares gerados "
              f"(poucos datasets por tipo).")

    # --- Pares diferentes ---
    if len(unique_labels) < 2:
        print("  ⚠ Apenas um tipo de dataset. "
              "Nenhum par diferente será gerado.")
    else:
        for _ in range(n_different):
            lbl1, lbl2 = rng.choice(unique_labels, size=2, replace=False)
            i1 = rng.choice(all_groups[lbl1])
            i2 = rng.choice(all_groups[lbl2])
            pairs_left.append(datasets[i1])
            pairs_right.append(datasets[i2])
            sim_labels.append(0.0)

    # --- Converte e embaralha ---
    pairs_left  = np.array(pairs_left,  dtype=np.float32)
    pairs_right = np.array(pairs_right, dtype=np.float32)
    sim_labels  = np.array(sim_labels,  dtype=np.float32)

    shuffle_idx = rng.permutation(len(sim_labels))
    print(f"   Total gerado: {len(sim_labels)} pares")

    return pairs_left[shuffle_idx], pairs_right[shuffle_idx], sim_labels[shuffle_idx]


# ============================================================================
# GERADOR BALANCEADO POR TIPO
# ============================================================================

def create_balanced_pairs(
    datasets:      np.ndarray,
    dataset_labels: np.ndarray,
    pairs_per_type: Optional[int] = None,
    random_state:   Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cria pares com balanceamento por tipo de dataset.

    Para cada tipo: gera pairs_per_type//2 pares similares
                    e   pairs_per_type//2 pares diferentes.

    Args:
        datasets:       Array (n_datasets, H, W, C)
        dataset_labels: Array (n_datasets,) — tipos de dataset
        pairs_per_type: Pares por tipo (padrão: CFG.pairs.pairs_per_class)
        random_state:   Seed

    Returns:
        pairs_left, pairs_right, similarity
    """
    pairs_per_type = pairs_per_type or CFG.pairs.pairs_per_class
    seed           = random_state   if random_state is not None else CFG.pairs.random_state
    rng            = np.random.default_rng(seed)

    unique_labels = np.unique(dataset_labels)
    n_types       = len(unique_labels)

    print(f"\n🎯 Pares balanceados por tipo de dataset:")
    print(f"   Tipos encontrados: {n_types}")
    print(f"   Pares por tipo   : {pairs_per_type}")

    all_left, all_right, all_sim = [], [], []

    for lbl in unique_labels:
        same  = np.where(dataset_labels == lbl)[0]
        other = np.where(dataset_labels != lbl)[0]

        n_sim  = pairs_per_type // 2
        n_diff = pairs_per_type // 2

        # similares
        for _ in range(n_sim):
            if len(same) >= 2:
                i1, i2 = rng.choice(same, size=2, replace=False)
                all_left.append(datasets[i1])
                all_right.append(datasets[i2])
                all_sim.append(1.0)

        # diferentes
        for _ in range(n_diff):
            if len(other) > 0:
                i1 = rng.choice(same)
                i2 = rng.choice(other)
                all_left.append(datasets[i1])
                all_right.append(datasets[i2])
                all_sim.append(0.0)

    pairs_left  = np.array(all_left,  dtype=np.float32)
    pairs_right = np.array(all_right, dtype=np.float32)
    similarity  = np.array(all_sim,   dtype=np.float32)

    idx = rng.permutation(len(similarity))
    print(f"   Total gerado: {len(similarity)} pares")
    print(f"   Similares  : {int(np.sum(similarity == 1))}")
    print(f"   Diferentes : {int(np.sum(similarity == 0))}")

    return pairs_left[idx], pairs_right[idx], similarity[idx]


# ============================================================================
# TESTES
# ============================================================================

if __name__ == "__main__":
    from config import CFG

    print("\n" + "=" * 70)
    print("TESTE DO GERADOR DE PARES")
    print("=" * 70)

    np.random.seed(42)
    n_ds = 10
    shape = CFG.siamese.input_shape
    datasets = np.random.randint(0, 2, size=(n_ds, *shape)).astype(np.float32)

    # Simula 3 tipos de dataset
    names        = [f"Dataset_{i}" for i in range(n_ds)]
    type_map     = {f"Dataset_{i}": i % 3 for i in range(n_ds)}
    ds_labels    = build_dataset_type_labels(names, type_map)

    print(f"\n  Dataset labels (tipos): {ds_labels}")
    print(f"  Tipos únicos          : {np.unique(ds_labels)}")

    # Teste 1: pares básicos
    pl, pr, sl = create_comparison_pairs(datasets, ds_labels, n_pairs=50)
    print(f"\n  Teste 1 — create_comparison_pairs:")
    print(f"    left shape : {pl.shape}")
    print(f"    similares  : {int(np.sum(sl == 1))} / {len(sl)}")
    print(f"    diferentes : {int(np.sum(sl == 0))} / {len(sl)}")

    # Teste 2: pares balanceados
    pl2, pr2, sl2 = create_balanced_pairs(datasets, ds_labels, pairs_per_type=20)
    print(f"\n  Teste 2 — create_balanced_pairs:")
    print(f"    total      : {len(sl2)}")
    print(f"    similares  : {int(np.sum(sl2 == 1))}")
    print(f"    diferentes : {int(np.sum(sl2 == 0))}")

    # Teste 3: aviso quando type_map está vazio
    print(f"\n  Teste 3 — type_map vazio (deve avisar):")
    empty_labels = build_dataset_type_labels(names, {})
    pl3, pr3, sl3 = create_comparison_pairs(datasets, empty_labels, n_pairs=20)
    print(f"    pares gerados: {len(sl3)}")

    print("\n✓ TODOS OS TESTES DO GERADOR DE PARES CONCLUÍDOS\n")