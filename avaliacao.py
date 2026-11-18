"""Protocolo de avaliação comum a todos os classificadores (OBS 1 a 4 do TC1).

Os modelos seguem a interface do scikit-learn: fit(X, y) e predict(X), com y codificado em inteiros.
"""
import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RAIZ = Path(__file__).parent
DADOS = RAIZ / "wall+following+robot+navigation+data"
ATRIBUTOS = ["SD_front", "SD_left", "SD_right", "SD_back"]
SPLITS = {"20/80": 0.2, "30/70": 0.3, "50/50": 0.5, "70/30": 0.7, "80/20": 0.8}
RODADAS = 100
STATS = {"mean": "média", "std": "desvio", "max": "máxima", "min": "mínima", "median": "mediana"}


def carregar(processada):
    """Retorna X, y (inteiros) e os nomes das classes."""
    if processada:
        df = pd.read_csv(DADOS / "sensor_readings_4_processed.csv")
    else:
        df = pd.read_csv(DADOS / "sensor_readings_4.data", header=None, names=ATRIBUTOS + ["Class"])
    classes, y = np.unique(df["Class"], return_inverse=True)
    return df[ATRIBUTOS].to_numpy(), y, classes


def separar(y, frac, rng):
    """Divisão estratificada: mesma proporção de cada classe no treino e no teste."""
    tr = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:round(frac * np.sum(y == c))]
                         for c in np.unique(y)])
    return tr, np.setdiff1d(np.arange(len(y)), tr)


def avaliar(modelos, X, y, classes, rodadas=RODADAS, seed=42):
    """Executa `rodadas` treino-teste independentes por split. Na mesma rodada, todos os modelos usam a mesma divisão."""
    rng = np.random.default_rng(seed)
    k = len(classes)
    registros, confusoes = [], {}
    for split, frac in SPLITS.items():
        for r in range(rodadas):
            tr, te = separar(y, frac, rng)
            for nome, base in modelos.items():
                modelo = copy.deepcopy(base)
                t0 = time.perf_counter()
                modelo.fit(X[tr], y[tr])
                t1 = time.perf_counter()
                pred = modelo.predict(X[te])
                t2 = time.perf_counter()

                cm = np.bincount(y[te] * k + pred, minlength=k * k).reshape(k, k)
                confusoes[split, nome, r] = cm
                registros.append({
                    "split": split, "modelo": nome, "rodada": r,
                    "acuracia": cm.trace() / cm.sum(),
                    "t_treino_ms": (t1 - t0) * 1e3, "t_teste_ms": (t2 - t1) * 1e3,
                    **dict(zip(classes, cm.diagonal() / cm.sum(axis=1))),  # taxa de acerto por classe
                })
    return pd.DataFrame(registros), confusoes


def estatisticas(res):
    """OBS 1 e 2: estatísticas da acurácia e tempos médios por split e modelo."""
    g = res.groupby(["split", "modelo"], sort=False)
    return g["acuracia"].agg(list(STATS)).rename(columns=STATS).join(g[["t_treino_ms", "t_teste_ms"]].mean())


def matrizes_confusao(res, confusoes, classes, splits=SPLITS):
    """OBS 2: matrizes de confusão da melhor e da pior rodada de cada modelo em cada split."""
    combos = [(s, m) for s in splits for m in res["modelo"].unique()]
    fig, axes = plt.subplots(len(combos), 2, figsize=(11, 4.5 * len(combos)), squeeze=False)
    for linha, (split, nome) in zip(axes, combos):
        sub = res[(res["split"] == split) & (res["modelo"] == nome)]
        for ax, (rotulo, i) in zip(linha, [("Melhor", sub["acuracia"].idxmax()), ("Pior", sub["acuracia"].idxmin())]):
            sns.heatmap(confusoes[split, nome, sub.loc[i, "rodada"]], annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set(title=f"{nome} | {split} | {rotulo} rodada (acc = {sub.loc[i, 'acuracia']:.4f})",
                   xlabel="Predito", ylabel="Real")
    fig.tight_layout()


def grafico_splits(res):
    """OBS 3: acurácia média ± desvio-padrão em função da separação treino/teste."""
    plt.figure(figsize=(10, 5))
    sns.pointplot(res, x="split", y="acuracia", hue="modelo", errorbar="sd",
                  dodge=0.5 if res["modelo"].nunique() > 1 else False)
    plt.title("Acurácia (média ± desvio-padrão) por separação treino/teste")


def por_classe(res, classes):
    """OBS 4: estatísticas da taxa de acerto de cada classe, por split e modelo."""
    longo = res.melt(id_vars=["split", "modelo"], value_vars=list(classes), var_name="classe", value_name="acerto")
    return longo.groupby(["classe", "split", "modelo"], sort=False)["acerto"].agg(list(STATS)).rename(columns=STATS)


def grafico_por_classe(res, classes):
    """OBS 4: distribuição da taxa de acerto por classe em cada split."""
    longo = res.melt(id_vars=["split", "modelo"], value_vars=list(classes), var_name="classe", value_name="acerto")
    sns.catplot(longo, x="classe", y="acerto", hue="modelo", col="split", col_wrap=2, kind="box", height=4, aspect=1.6)


def resumo(res, classes):
    """Conclusões numéricas explícitas da avaliação."""
    media = res.groupby(["split", "modelo"], sort=False)["acuracia"].mean()
    (split, modelo), acc = media.idxmax(), media.max()
    print(f"Melhor configuração: {modelo} com split {split} -> acurácia média {acc:.4f}")

    tempos = res.groupby("modelo", sort=False)[["t_treino_ms", "t_teste_ms"]].mean()
    print("\nTempo médio por rodada (ms):")
    for nome, t in tempos.iterrows():
        print(f"  {nome}: treino {t['t_treino_ms']:.3f} | teste {t['t_teste_ms']:.3f}")

    print("\nEfeito do split (acurácia média 20/80 -> 80/20):")
    for nome, s in media.unstack("split").iterrows():
        print(f"  {nome}: {s['20/80']:.4f} -> {s['80/20']:.4f} ({s['80/20'] - s['20/80']:+.4f})")

    acerto = res[list(classes)].mean().sort_values()
    print("\nTaxa de acerto média por classe (todas as rodadas):")
    for classe, v in acerto.items():
        print(f"  {classe}: {v:.4f}")
    print(f"Classe mais fácil: {acerto.index[-1]} | mais difícil: {acerto.index[0]}")


def salvar(res, versao, classificador):
    """Grava os resultados por rodada em resultados/<versao>/<classificador>.csv (usado em comparacao.ipynb)."""
    destino = RAIZ / "resultados" / versao / f"{classificador}.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(destino, index=False)
