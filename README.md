# Inclui+ Vision

Detecção de recursos e barreiras de acessibilidade em fotos e em vídeo em tempo real, com um modelo YOLO11 customizado. Projeto da atividade *Applied Computer Vision | Enxergando o mundo real*, no contexto da plataforma Inclui+.

**Integrantes:** _nomes e RMs_

> **Estado do repositório.** O código, o notebook de treino e os testes estão prontos. O modelo ainda **não foi treinado**: faltam as imagens (passo 1), o treino no Colab (passo 2) e os números da seção [Resultados](#resultados). O arquivo `weights/best.pt` só existe depois do treino.

## O que o modelo detecta

| Classe | Descrição | Categoria no Inclui+ |
|---|---|---|
| `rampa` | Rampa de acesso | Mobilidade |
| `escada` | Escada ou degrau (possível barreira) | Mobilidade |
| `corrimao` | Corrimão em escada ou rampa | Mobilidade |
| `elevador` | Porta ou cabine de elevador | Mobilidade |
| `barra_apoio` | Barra de apoio em banheiro adaptado | Mobilidade |
| `vaga_pcd` | Vaga de estacionamento PcD sinalizada | Mobilidade |
| `piso_tatil` | Piso tátil direcional ou de alerta | Visão |
| `simbolo_acessibilidade` | Símbolo Internacional de Acesso | Geral |

Além das detecções, os scripts emitem um alerta de barreira quando há **escada sem rampa** na cena.

Com pouco tempo, comece com `rampa`, `escada`, `corrimao` e `piso_tatil`: basta deixar só essas quatro em `classes`, no `dataset/fontes.yaml`.

## Estrutura

```
inclui-vision/
├── notebooks/treinamento.ipynb   # passos 3 e 4, no Google Colab
├── dataset/
│   ├── fontes.yaml               # fontes de imagens e mapa de classes de cada uma
│   ├── data.yaml                 # classes e pastas, no formato do Ultralytics
│   └── README.md                 # onde achar datasets públicos e como fotografar
├── src/
│   ├── categorias.py             # classes, categorias do Inclui+ e regra de barreira
│   ├── preparar_dataset.py       # une as fontes, unifica classes e divide em 70/20/10
│   ├── ver_amostras.py           # grade de imagens anotadas, para conferir o dataset
│   ├── avaliar.py                # métricas no teste, matriz de confusão e exemplos
│   ├── realtime.py               # detecção na webcam ou na câmera do celular
│   ├── relatorio.py              # relatório de acessibilidade de uma foto
│   └── util_dataset.py           # leitura de datasets no formato YOLO
├── tests/                        # testes que rodam sem GPU e sem o modelo
├── weights/best.pt               # modelo treinado (depois do passo 2)
├── results/                      # métricas, gráficos e exemplos (depois do passo 2)
├── docs/roteiro_video.md         # roteiro do vídeo de demonstração
└── requirements.txt
```

## Como rodar

Todos os comandos partem da raiz do repositório.

```bash
python -m venv .venv
source .venv/bin/activate          # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Montar o dataset

1. Baixe os datasets públicos e anote as fotos do grupo. O [`dataset/README.md`](dataset/README.md) lista os datasets já conferidos e dá as orientações para as fotos.
2. Exporte cada fonte no formato **YOLOv11** (ou YOLOv8) e extraia em `dataset/brutos/<nome-da-fonte>/`.
3. Ajuste o `dataset/fontes.yaml`: uma entrada por fonte, com o mapa das classes dela para as nossas. Apague as fontes que você não baixou.
4. Rode a preparação e confira o resultado:

```bash
python src/preparar_dataset.py --config dataset/fontes.yaml --saida dataset --limpar
python src/ver_amostras.py --n 12 --saida results/amostras_treino.jpg
```

O primeiro comando imprime quantas imagens cada classe tem em cada conjunto e avisa quais estão abaixo da meta de 300. O segundo salva uma grade com as anotações desenhadas: se uma rampa aparecer como escada, o mapa de classes está errado.

O que a preparação faz:

- traduz os nomes de classe de cada fonte (`stairs`, `upstairs` e `downstairs` viram `escada`) e descarta as classes fora do escopo;
- divide em 70% treino, 20% validação e 10% teste, com semente fixa, sorteando pela classe mais rara de cada foto para que mesmo as classes pequenas apareçam nos três conjuntos;
- mantém no mesmo conjunto as cópias aumentadas de uma mesma foto (os arquivos `.rf.<hash>` do Roboflow), para a foto não vazar do treino para o teste;
- grava também `data_publico.yaml`, com o treino só de fontes públicas e a mesma validação e o mesmo teste, para o experimento sobre o valor das fotos próprias.

### 2. Treinar e avaliar

Abra [`notebooks/treinamento.ipynb`](notebooks/treinamento.ipynb) no Google Colab, ative a GPU T4 e siga as células. O notebook prepara o dataset, treina o `yolo11n` a partir dos pesos do COCO, avalia no conjunto de teste e roda os dois experimentos do relatório.

No fim, traga para o repositório o `weights/best.pt` e a pasta `results/`.

Para reavaliar no computador um modelo já treinado:

```bash
python src/avaliar.py --pesos weights/best.pt --data dataset/data.yaml --split test
```

### 3. Detecção em tempo real

```bash
python src/realtime.py                                   # webcam do computador
python src/realtime.py --fonte 1                         # câmera do celular via DroidCam ou Iriun
python src/realtime.py --fonte gravacao.mp4 --gravar demo.mp4
```

A janela mostra as bounding boxes, um resumo por categoria do Inclui+, o FPS e o alerta de escada sem rampa. A tecla `q` encerra.

Uma detecção continua valendo por 1 segundo depois de sumir (`--janela`), para o painel não piscar e o alerta não disparar só porque a rampa saiu do enquadramento por um instante. Com `--janela 0` a análise é quadro a quadro.

O terceiro comando processa um vídeo gravado com o celular e salva o resultado anotado. É o jeito mais prático de gravar a demonstração em locais onde não dá para levar o notebook.

### 4. Relatório de uma foto

```bash
python src/relatorio.py foto.jpg --saida results/exemplos/foto.jpg
```

```
Relatório de acessibilidade — Inclui+
  [Mobilidade] corrimao: 1
  [Mobilidade] escada: 2
  [Visão] piso_tatil: 1
  ATENÇÃO, barreira: Escada sem rampa alternativa
```

Com `--json` a saída é estruturada, no formato que a plataforma consumiria.

### 5. Testes

```bash
python -m unittest discover -s tests -v
```

Cobrem a preparação do dataset, as categorias, a regra de barreira, o painel do vídeo e as tabelas de métricas. Não precisam de GPU, de imagens reais nem do modelo treinado.

## Resultados

_A preencher depois do treino, com o conteúdo de `results/metricas.md` e de `results/comparacao.csv`._

| Modelo | mAP@50 | mAP@50-95 | Precisão | Recall | FPS (T4) |
|---|---:|---:|---:|---:|---:|
| `yolo11n` | | | | | |
| `yolo11s` | | | | | |
| `yolo11n`, só datasets públicos | | | | | |

Figuras que o treino gera em `results/`: `results.png` (curvas por época), `confusion_matrix.png`, as curvas de precisão e recall e os exemplos com bounding boxes em `exemplos/`.

## Limitações conhecidas

- **Anotações incompletas ao unir datasets.** Cada dataset público só anota as próprias classes: um dataset de escadas não marca os corrimãos que aparecem nas fotos. Para o modelo, esses corrimãos viram fundo. O ideal é completar as anotações no Roboflow, pelo menos na validação e no teste.
- **O alerta vale só para o que a câmera viu.** Uma rampa fora do enquadramento não é considerada, e um elevador ao lado da escada não cancela o alerta.
- **O modelo detecta presença, não conformidade.** Ele não mede a inclinação da rampa nem a largura da porta; uma rampa íngreme demais conta como rampa.
- **Classes com poucas imagens e iluminação ruim** tendem a ter recall menor. A tabela por classe mostra onde isso acontece.

## Créditos e licenças

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), licença AGPL-3.0.
- Datasets públicos do Roboflow Universe, a maioria sob CC BY 4.0, que exige atribuição. Liste aqui os que forem usados de fato:

| Dataset | Autor | Licença | Link |
|---|---|---|---|
| | | | |
