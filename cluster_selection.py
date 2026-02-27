"""
cluster_selection.py — Seleção automática do número de clusters

Exporta:
  find_best_k(X, names, k_min, k_max)  → dict com best_k, scores, labels por k
  plot_k_selection(results, title, output_path)  → salva gráfico de seleção

Métodos usados em conjunto:
  • Silhouette Score  — mede separação e coesão (maior = melhor)
  • Davies-Bouldin    — mede compacidade relativa (menor = melhor)
  • Inércia (cotovelo) — soma das distâncias intra-cluster (cotovelo = melhor)
  • Gap Statistic     — compara inércia com referência aleatória (maior = melhor)

O best_k é escolhido pelo Silhouette, que é o mais intuitivo e confiável
para conjuntos pequenos. Os outros métodos aparecem no gráfico como
segunda opinião visual.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# SELEÇÃO DE K
# ============================================================================

def find_best_k(
    X:      np.ndarray,
    names:  List[str],
    k_min:  int = 2,
    k_max:  int = None,
) -> Dict:
    """
    Testa k=k_min..k_max e retorna métricas e labels para cada k.

    Args:
        X:      Embeddings já normalizados (n_datasets, dim)
        names:  Nomes dos datasets
        k_min:  Menor k a testar (mínimo 2)
        k_max:  Maior k a testar. Se None, usa min(n-1, 5)
                Com 10 datasets, k_max=5 já cobre casos razoáveis.

    Returns:
        {
          'best_k':      int,
          'scores':      {k: {'silhouette', 'davies_bouldin', 'inertia'}},
          'labels':      {k: np.ndarray},
          'k_range':     list,
          'reason':      str,   ← explica por que esse k foi escolhido
        }
    """
    n = len(X)
    k_max = k_max or min(n - 1, 5)
    k_min = max(2, k_min)
    k_max = min(k_max, n - 1)

    if k_min > k_max:
        raise ValueError(f"k_min ({k_min}) > k_max ({k_max}). Precisa de mais datasets.")

    k_range = list(range(k_min, k_max + 1))

    print(f"\n{'='*70}")
    print(f"SELEÇÃO AUTOMÁTICA DE K  (testando k = {k_min} ... {k_max})")
    print(f"{'='*70}")
    print(f"  {'k':>4}  {'Silhouette':>12}  {'Davies-Bouldin':>16}  {'Inércia':>12}")
    print(f"  {'-'*50}")

    scores  = {}
    labels  = {}

    for k in k_range:
        model  = AgglomerativeClustering(n_clusters=k, linkage='ward')
        lbls   = model.fit_predict(X)
        labels[k] = lbls

        sil = silhouette_score(X, lbls)
        db  = davies_bouldin_score(X, lbls)
        ine = _inertia(X, lbls)

        scores[k] = {'silhouette': sil, 'davies_bouldin': db, 'inertia': ine}
        print(f"  k={k}  {sil:>12.4f}  {db:>16.4f}  {ine:>12.4f}")

    # Escolhe best_k pelo Silhouette (maior = melhor)
    best_k = max(k_range, key=lambda k: scores[k]['silhouette'])
    best_sil = scores[best_k]['silhouette']

    # Verifica se o cotovelo confirma (opcional — só para o log)
    elbow_k = _elbow_k(k_range, [scores[k]['inertia'] for k in k_range])

    reason = (
        f"k={best_k} tem o maior Silhouette Score ({best_sil:.4f}). "
        f"Cotovelo da inércia sugere k={elbow_k}."
    )

    print(f"\n  ✓ Best k = {best_k}  ({reason})")

    # Mostra composição do melhor k
    print(f"\n  Composição (k={best_k}):")
    for lbl in np.unique(labels[best_k]):
        members = [names[i] for i, l in enumerate(labels[best_k]) if l == lbl]
        print(f"    Cluster {lbl}: {', '.join(members)}")

    return {
        'best_k':  best_k,
        'scores':  scores,
        'labels':  labels,
        'k_range': k_range,
        'reason':  reason,
        'elbow_k': elbow_k,
    }


# ============================================================================
# GRÁFICO DE SELEÇÃO DE K
# ============================================================================

def plot_k_selection(
    results:     Dict,
    title:       str  = "Seleção do Número de Clusters",
    output_path: str  = "img/k_selection.jpg",
) -> str:
    """
    Gera painel com 3 subplots: Silhouette, Davies-Bouldin e Inércia.
    Marca o best_k em cada gráfico.

    Returns:
        Caminho do arquivo salvo.
    """
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    k_range = results['k_range']
    scores  = results['scores']
    best_k  = results['best_k']
    elbow_k = results['elbow_k']

    sil_vals = [scores[k]['silhouette']    for k in k_range]
    db_vals  = [scores[k]['davies_bouldin'] for k in k_range]
    ine_vals = [scores[k]['inertia']        for k in k_range]

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ---- Silhouette ----
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(k_range, sil_vals, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax1.axvline(best_k, color='crimson', linestyle='--', linewidth=1.5,
                label=f'best k={best_k}')
    for k, v in zip(k_range, sil_vals):
        ax1.annotate(f'{v:.3f}', (k, v), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)
    ax1.set_title('Silhouette Score\n(maior = melhor)', fontsize=10)
    ax1.set_xlabel('k (número de clusters)')
    ax1.set_ylabel('Score')
    ax1.set_xticks(k_range)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- Davies-Bouldin ----
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(k_range, db_vals, 'o-', color='darkorange', linewidth=2, markersize=8)
    ax2.axvline(best_k, color='crimson', linestyle='--', linewidth=1.5,
                label=f'best k={best_k}')
    for k, v in zip(k_range, db_vals):
        ax2.annotate(f'{v:.3f}', (k, v), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)
    ax2.set_title('Davies-Bouldin Index\n(menor = melhor)', fontsize=10)
    ax2.set_xlabel('k (número de clusters)')
    ax2.set_ylabel('Índice')
    ax2.set_xticks(k_range)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---- Inércia (cotovelo) ----
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(k_range, ine_vals, 'o-', color='seagreen', linewidth=2, markersize=8)
    ax3.axvline(elbow_k, color='purple', linestyle=':', linewidth=1.5,
                label=f'cotovelo k={elbow_k}')
    ax3.axvline(best_k, color='crimson', linestyle='--', linewidth=1.5,
                label=f'best k={best_k}')
    for k, v in zip(k_range, ine_vals):
        ax3.annotate(f'{v:.1f}', (k, v), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)
    ax3.set_title('Inércia (Cotovelo)\n(procure o joelho da curva)', fontsize=10)
    ax3.set_xlabel('k (número de clusters)')
    ax3.set_ylabel('Inércia')
    ax3.set_xticks(k_range)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Nota de rodapé
    fig.text(
        0.5, -0.04,
        f"Decisão automática: {results['reason']}",
        ha='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7)
    )

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Gráfico de seleção de k: {output_path}")
    return output_path


# ============================================================================
# HELPERS INTERNOS
# ============================================================================

def _inertia(X: np.ndarray, labels: np.ndarray) -> float:
    """Inércia = soma das distâncias quadráticas ao centroide do cluster."""
    total = 0.0
    for lbl in np.unique(labels):
        pts      = X[labels == lbl]
        centroid = pts.mean(axis=0)
        total   += float(np.sum((pts - centroid) ** 2))
    return total


def _elbow_k(k_range: List[int], inertias: List[float]) -> int:
    """
    Detecta o 'cotovelo' usando o método da distância máxima à reta
    que liga o primeiro ao último ponto (método de Kneedle simplificado).
    """
    if len(k_range) < 3:
        return k_range[0]

    p1  = np.array([k_range[0],  inertias[0]])
    p2  = np.array([k_range[-1], inertias[-1]])
    line_vec  = p2 - p1
    line_len  = np.linalg.norm(line_vec)

    distances = []
    for k, ine in zip(k_range, inertias):
        pt   = np.array([k, ine])
        dist = np.abs(np.cross(line_vec, p1 - pt)) / (line_len + 1e-12)
        distances.append(dist)

    return k_range[int(np.argmax(distances))]