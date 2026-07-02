# ⚽ Módulo 3 — Previsão de Resultados de Futebol (ML)

Projeto pessoal de Ciência de Dados para treinar o **pipeline completo** (dado bruto →
análise → modelo → explicação → produto) antes do Módulo 3 do Inteli.

## O que ele faz

Dado **dois times antes de um jogo**, o modelo estima a **probabilidade** de cada
resultado: **vitória do mandante · empate · vitória do visitante**. É uma
**classificação probabilística multiclasse (3 classes)** — nunca uma certeza, sempre
uma probabilidade.

## Como funciona (etapas do pipeline)

1. **Dados** — `results.csv`, +48 mil jogos internacionais desde 1872 (times, placar, torneio, mando).
2. **Alvo (target)** — coluna `resultado` (`casa`/`empate`/`fora`) derivada do placar.
   ⚠️ Nada que só existe **depois** do jogo (o placar) pode virar feature → isso é *data leakage*.
3. **Features (X)** — só o que existe **antes do apito**: forma recente, saldo de gols
   histórico, retrospecto no confronto direto, mando de campo.
4. **Treino/teste** — separação temporal para não avaliar no que o modelo já viu (evita overfitting).
5. **Modelos** — Random Forest e Gradient Boosting, comparados contra baseline burra ("sempre casa").
6. **Avaliação** — acurácia é enganosa (a casa vence muito); usar matriz de confusão, precisão e recall.
7. **Explainability** — SHAP, para explicar **por que** o modelo previu cada resultado.
8. **Produto** — modelo salvo em `.joblib` e servido via **FastAPI** (recebe dois times → devolve probabilidades).

## Estrutura de pastas

```
modulo3-futebol/
├── data/raw/          # CSVs originais do Kaggle (NÃO versionado)
├── data/processed/    # dados tratados (NÃO versionado)
├── notebooks/         # 01_eda.ipynb, ...
├── src/features/      # feature engineering: forma, saldo, mando
├── src/model/         # treino e avaliação
├── src/api/           # FastAPI que serve o modelo
├── models/            # .joblib treinados (NÃO versionado)
├── documents/         # análise de negócio, gráficos
└── dashboard/         # backend + frontend do dashboard
```

## Setup no WSL (Ubuntu)

Este projeto roda no WSL. O ambiente virtual já foi criado em `.venv` (Python 3.14).

```bash
# Ativar o venv
cd /mnt/c/Users/admin/Documents/modulo3-futebol
source .venv/bin/activate

# (se recriar o venv do zero) instalar dependências
pip install -r requirements.txt
```

> ℹ️ Este ambiente não tinha `ensurepip`; o pip foi bootstrapado via `get-pip.py`.
> Para recriar: `python3 -m venv --without-pip .venv && curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python -`

## Dataset

Kaggle: **`martj42/international-football-results-from-1872-to-2017`**
(arquivos: `results.csv`, `goalscorers.csv`, `shootouts.csv`).

✅ **Já baixados** para `data/raw/` via `kagglehub` (download público, sem credencial).
`results.csv` = 49.498 jogos × 9 colunas.

Para rebaixar do zero (a pasta `data/raw/` é ignorada pelo git):

```bash
source .venv/bin/activate
python - <<'PY'
import shutil, kagglehub
from pathlib import Path
p = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2017")
for f in ("results.csv", "goalscorers.csv", "shootouts.csv"):
    shutil.copy(Path(p) / f, Path("data/raw") / f)
print("CSVs em data/raw/")
PY
```

## Abrir o notebook de EDA

```bash
source .venv/bin/activate
jupyter notebook notebooks/01_eda.ipynb
# ou:  jupyter lab
```

## Roadmap por sprints

- **Sprint 2 · EDA** — `notebooks/01_eda.ipynb` (ponto de partida) ⬅️ você está aqui
- **Sprint 3** — feature engineering (`src/features/`)
- **Sprint 4** — treino + avaliação (`src/model/`)
- **Sprint 5** — explainability (SHAP) + API (`src/api/`) + dashboard
