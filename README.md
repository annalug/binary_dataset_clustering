# 🛡️ Rede Siamesa para Detecção de Malware Android

Sistema de deep learning para comparar similaridade entre datasets binários de malware Android usando redes neurais siamesas.

## 🚀 Quick Start

```python
from standardizer import DatasetStandardizer, create_comparison_pairs
from siamese import SiameseNet
import numpy as np

# 1. Padronizar datasets
standardizer = DatasetStandardizer(target_samples=256, target_features=100)
datasets_std = standardizer.fit_transform_batch(raw_datasets)

# 2. Criar pares de treino
pairs_l, pairs_r, labels = create_comparison_pairs(datasets_std, malware_labels, n_pairs=5000)

# 3. Treinar rede siamesa
siamese = SiameseNet(embedding_dim=128)
siamese.train(pairs_l, pairs_r, labels, epochs=50)

# 4. Detectar malware em novo app
new_app_std = standardizer.transform(new_app)
similarity = siamese.predict_similarity(new_app_std, known_malware)
print(f"Similaridade: {similarity:.2f}")  # 0.0-1.0
```

## 📋 Formato dos Dados

### Input Esperado
- **Datasets brutos**: Matrizes binárias (0s e 1s)
- **Shape**: `(n_amostras, n_features)` - tamanhos variados OK!
  - `n_amostras`: 500-2000
  - `n_features`: 30-100

### Output Padronizado
- **Shape fixo**: `(256, 100, 1)` para todos datasets
- **Método**: PCA ou truncamento automático
- **Binaridade**: Preservada (0/1)

### Exemplo CSV

```csv
perm_INTERNET,perm_SMS,api_exec,api_crypto,...
1,0,1,0,...
0,1,0,1,...
1,1,1,0,...
```

## 🏗️ Arquitetura

### DatasetStandardizer
Padronização inteligente de datasets com tamanhos variados:

```python
DatasetStandardizer(
    target_samples=256,      # Linhas fixas
    target_features=100,     # Colunas fixas
    use_pca=True,           # PCA vs truncamento
    min_variance_ratio=0.90  # Variância mínima preservada
)
```

**Estratégias:**
- Amostras: padding com zeros ou amostragem
- Features: PCA inteligente ou truncamento
- Mantém dados binários após transformação

### SiameseNet
Rede neural com encoder CNN compartilhado:

```python
SiameseNet(
    input_shape=(256, 100, 1),
    embedding_dim=128,        # 64-256
    architecture='default'    # 'light', 'default', 'deep'
)
```

**Arquiteturas disponíveis:**
- `'light'`: 2 blocos conv, ~100K parâmetros (rápida)
- `'default'`: 3 blocos conv, ~500K parâmetros (balanceada)
- `'deep'`: 4 blocos conv, ~2M parâmetros (alta capacidade)

## 📖 Exemplos de Uso

### 1. Treinar Modelo

```python
import pandas as pd
from standardizer import DatasetStandardizer, create_comparison_pairs
from siamese import SiameseNet

# Carregar datasets de malware
raw_datasets = [
    pd.read_csv(f'malware_{i}.csv').values  # Apenas 0s e 1s!
    for i in range(10)
]
labels = [0, 0, 1, 1, 2, 2, 0, 1, 2, 1]  # 0=Trojan, 1=Spyware, 2=Ransomware

# Padronizar
standardizer = DatasetStandardizer()
datasets_std = standardizer.fit_transform_batch(raw_datasets)

# Criar pares
pairs_l, pairs_r, sim_labels = create_comparison_pairs(
    datasets_std, labels, n_pairs=5000
)

# Treinar
siamese = SiameseNet()
siamese.train(pairs_l, pairs_r, sim_labels, epochs=50)
siamese.save('my_detector')
```

### 2. Detectar Malware em Novo App

```python
# Carregar modelo
standardizer = DatasetStandardizer()
siamese = SiameseNet()
siamese.load('my_detector')

# Novo app suspeito
new_app = pd.read_csv('suspicious_app.csv').values
new_app_std = standardizer.transform(new_app)

# Comparar com base de conhecimento
results = siamese.compare_with_multiple(
    new_app_std,
    known_malware_datasets,
    ['Trojan_A', 'Spyware_X', 'Ransomware_Z']
)

# Analisar
threshold = 0.7
for name, score in results:
    if score > threshold:
        print(f"⚠️  MALWARE: {name} ({score*100:.1f}% similar)")
```

### 3. Análise em Lote

```python
apps = {
    'app1.apk': pd.read_csv('app1.csv').values,
    'app2.apk': pd.read_csv('app2.csv').values,
    'app3.apk': pd.read_csv('app3.csv').values,
}

for app_name, app_data in apps.items():
    app_std = standardizer.transform(app_data)
    similarity = siamese.predict_similarity(app_std, known_malware)
    
    status = "MALWARE" if similarity > 0.7 else "LIMPO"
    print(f"{app_name}: {status} ({similarity:.2f})")
```

## ⚙️ Hiperparâmetros

### Recomendações por Tamanho de Dataset

**Dataset Pequeno** (< 5 datasets, < 1000 pares):
```python
standardizer = DatasetStandardizer(use_pca=False)  # Truncamento
siamese = SiameseNet(
    embedding_dim=64,
    architecture='light'
)
# epochs=30, batch_size=16
```

**Dataset Médio** (5-20 datasets, 1000-5000 pares):
```python
standardizer = DatasetStandardizer(use_pca=True)
siamese = SiameseNet(
    embedding_dim=128,
    architecture='default'
)
# epochs=50, batch_size=32
```

**Dataset Grande** (> 20 datasets, > 5000 pares):
```python
standardizer = DatasetStandardizer(use_pca=True)
siamese = SiameseNet(
    embedding_dim=256,
    architecture='deep'
)
# epochs=100, batch_size=64
```

## 🎯 Threshold de Detecção

Ajuste baseado em sua tolerância a falsos positivos/negativos:

| Threshold | Comportamento | Uso Recomendado |
|-----------|---------------|-----------------|
| 0.5-0.6 | Muito sensível | Triagem inicial |
| 0.7-0.8 | Balanceado | Uso geral |
| 0.9+ | Muito restritivo | Alta segurança |

**Dica**: Plote curva ROC no conjunto de validação para escolher threshold ideal.

## 📊 Métricas

O modelo reporta:
- **Accuracy**: Acurácia geral
- **Precision**: Precisão na detecção
- **Recall**: Cobertura na detecção
- **AUC**: Área sob curva ROC
- **F1-Score**: Média harmônica precision/recall

## 🔄 Aprendizado Incremental

Para adicionar novos malwares descobertos:

```python
# 1. Carregar modelo existente
siamese.load('my_detector')

# 2. Adicionar novos datasets
all_datasets = old_datasets + new_datasets
all_labels = old_labels + new_labels

# 3. Re-padronizar
datasets_std = standardizer.fit_transform_batch(all_datasets)

# 4. Criar novos pares
pairs_l, pairs_r, labels = create_comparison_pairs(datasets_std, all_labels)

# 5. Re-treinar (fine-tuning)
siamese.train(pairs_l, pairs_r, labels, epochs=20)
siamese.save('my_detector_v2')
```

## 🐛 Troubleshooting

### Problema: Baixa acurácia
**Soluções:**
- ✓ Aumentar `n_pairs` (mais dados de treino)
- ✓ Aumentar `epochs`
- ✓ Usar arquitetura `'deep'`
- ✓ Verificar balanceamento (50% similar, 50% diferente)

### Problema: Overfitting
**Soluções:**
- ✓ Reduzir `epochs` ou usar early stopping
- ✓ Usar arquitetura `'light'`
- ✓ Aumentar dropout na rede
- ✓ Adicionar mais dados de treino

### Problema: Treino lento
**Soluções:**
- ✓ Reduzir `batch_size`
- ✓ Usar arquitetura `'light'`
- ✓ Reduzir `target_samples` e `target_features`
- ✓ Usar GPU (instalar tensorflow-gpu)

### Problema: Memória insuficiente
**Soluções:**
- ✓ Reduzir `batch_size` (16 ou 8)
- ✓ Reduzir `n_pairs`
- ✓ Processar datasets em lotes menores

## 📁 Estrutura de Arquivos

```
malware_siamese_v2/
├── standardizer.py    # Padronização de datasets
├── siamese.py         # Rede neural siamesa
├── examples.py        # Exemplos de uso completos
└── README.md          # Esta documentação
```

## 📦 Dependências

```bash
pip install tensorflow numpy scikit-learn pandas matplotlib
```

**Versões recomendadas:**
- Python 3.8+
- TensorFlow 2.10+
- NumPy 1.23+
- Scikit-learn 1.2+

## 📚 Referências

1. **Siamese Networks**: Koch et al. (2015) - "Siamese Neural Networks for One-shot Image Recognition"
2. **Malware Detection**: Saxe & Berlin (2015) - "Deep Neural Network Based Malware Detection"
3. **DREBIN**: Arp et al. (2014) - Dataset público de malware Android

## 💡 Dicas Práticas

### Extração de Features
Para criar datasets binários a partir de APKs:

```python
# Usando Androguard
from androguard.core.apk import APK

apk = APK('app.apk')

# Extrai features binárias
features = {
    'INTERNET': 1 if 'INTERNET' in apk.get_permissions() else 0,
    'SEND_SMS': 1 if 'SEND_SMS' in apk.get_permissions() else 0,
    # ... mais features
}
```

**Features recomendadas:**
- Permissões perigosas (30-50 features)
- APIs suspeitas (20-30 features)
- Receivers, services, providers (10-20 features)

### Balanceamento de Classes
Se classes desbalanceadas:

```python
# Ajuste balance_ratio
pairs_l, pairs_r, labels = create_comparison_pairs(
    datasets_std,
    malware_labels,
    n_pairs=5000,
    balance_ratio=0.6  # 60% similar, 40% diferente
)
```

### Validação Cruzada
Para avaliar melhor:

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(datasets_std)):
    print(f"\nFold {fold+1}/5")
    
    # Treina no fold
    train_datasets = datasets_std[train_idx]
    train_labels = labels[train_idx]
    
    # ... criar pares e treinar
```

## 📧 Suporte

Para dúvidas ou problemas:
1. Consulte `examples.py` para exemplos completos
2. Verifique troubleshooting acima
3. Ajuste hiperparâmetros gradualmente

---

**Desenvolvido para pesquisa em segurança Android** 🛡️