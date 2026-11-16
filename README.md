# Trabalho Computacional 1 — Reconhecimento de Padrões

Classificação da ação de um robô móvel na tarefa de navegação por seguimento de paredes.

## Base de dados

[Wall-Following Robot Navigation Data](https://archive.ics.uci.edu/dataset/194/wall+following+robot+navigation+data) (UCI).

Os dados foram coletados com o robô SCITOS G5 percorrendo uma sala no sentido horário, seguindo a parede, a 9 amostras por segundo. O robô tem 24 sensores de ultrassom ao redor do corpo. Este trabalho usa a versão com **4 atributos** (`sensor_readings_4.data`): cada atributo é a menor leitura entre os sensores de um arco de 60°.

| Atributo | Descrição |
|---|---|
| `SD_front` | menor distância na frente do robô |
| `SD_left` | menor distância à esquerda |
| `SD_right` | menor distância à direita |
| `SD_back` | menor distância atrás |

As distâncias estão em metros, e 5,0 m é o alcance máximo do sensor.

**Classes** (ação do robô), com 5456 instâncias no total:

| Classe | Instâncias | % |
|---|---|---|
| Move-Forward | 2205 | 40,4 |
| Sharp-Right-Turn | 2097 | 38,4 |
| Slight-Right-Turn | 826 | 15,1 |
| Slight-Left-Turn | 328 | 6,0 |

Segundo os autores, o problema não é linearmente separável.

## Pré-processamento

Feito em `data_process.ipynb`, que gera `sensor_readings_4_processed.csv`. O resultado tem 5406 instâncias.

- A base não tem valores ausentes.
- As 50 linhas duplicadas foram removidas.
- Os atributos foram normalizados com z-score (média 0, desvio-padrão 1).
- Outliers e leituras saturadas em 5,0 m foram mantidos, porque são medições reais.

## Estrutura

```
data_analysis.ipynb          análise descritiva da base original
data_process.ipynb           pré-processamento e análise da base processada
classification/              classificadores em numpy, base original
classification_processed/    classificadores em numpy, base processada
classification_sklearn/      classificadores em scikit-learn, base processada
wall+following+robot+navigation+data/   arquivos da base
```

Cada pasta de classificação tem um notebook por classificador:
- 1-NN com distância de Minkowski (m ∈ {1/2, 2/3, 1, 3/2, 2, 5/2})
- distância mínima ao centroide
- distância mínima ao centroide, versão robusta
- máxima correlação

Cada classificador é avaliado em 100 rodadas para cada divisão treino/teste (20/80, 30/70, 50/50, 70/30 e 80/20). Os notebooks registram os tempos de treino e de teste, as estatísticas de acurácia, as matrizes de confusão da melhor e da pior rodada e o desempenho por classe.

## Como executar

Requer Python 3.12 e [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Execute `data_process.ipynb` antes dos notebooks em `classification_processed/` e `classification_sklearn/`, porque são eles que leem a base processada.
