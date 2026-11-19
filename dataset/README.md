# Dataset

As imagens não ficam no Git. Esta pasta guarda a receita: de onde elas vêm (`fontes.yaml`) e como viram o dataset de treino (`src/preparar_dataset.py`).

```
dataset/
├── fontes.yaml       # fontes e mapa de classes (você edita)
├── data.yaml         # gerado pela preparação
├── brutos/           # um export YOLO por fonte (você baixa)
└── train/ valid/ test/   # gerados pela preparação
```

## Datasets públicos

Busca feita no Roboflow Universe em 19/11/2026. Antes de usar qualquer um, abra a página, olhe as imagens e confira a licença.

### Conferidos

| Para as classes | Dataset | Classes de origem | Imagens | Licença |
|---|---|---|---:|---|
| `escada`, `corrimao` | [Stairs \| Handrail Detector](https://universe.roboflow.com/itlabroom/stairs-handrail-detector) (ITLabRoom) | `downstairs`, `handrail`, `upstairs` | 451 | CC BY 4.0 |
| `escada` | [Stairs](https://universe.roboflow.com/barrierfreenavigation/stairs-u437q) (BarrierFreeNavigation) | `stairs` | 110 | CC BY 4.0 |
| `rampa`, `escada`, `simbolo_acessibilidade` | [accessibility object detection](https://universe.roboflow.com/barrierfree/accessibility-object-detection-lrrth) (Barrierfree) | rampas, escadas, símbolos de acessibilidade, pessoas e buracos na via | não informado na página | não informada na página |
| `piso_tatil` | [tactile_paving](https://universe.roboflow.com/tactile-paving-detection) (tactile paving detection) | não informadas na página | 439 | não informada na página |

Os dois primeiros já estão no `fontes.yaml`. Para os outros dois, pegue os nomes exatos das classes no `data.yaml` do export antes de escrever o mapa.

### Candidatos que apareceram na busca e ainda não foram abertos

- Piso tátil: [tactile_pavement](https://universe.roboflow.com/work-rdjfc/tactile_pavement) e [tactile-paving-segmentation](https://universe.roboflow.com/thesis-jjmi5/tactile-paving-segmentation). O segundo é de segmentação; a preparação converte os polígonos em caixas.
- Escada: [YOLO_stairs](https://universe.roboflow.com/hohyoo/yolo_stairs).
- Entradas de estabelecimentos: [entrance accessibility 2](https://universe.roboflow.com/practicalai-junur/entrance-accessibility-2).
- Elevador: [elevator_door](https://universe.roboflow.com/elevator-panel-detection/elevator_door-riuxl) e [elevator](https://universe.roboflow.com/lmw/elevator-mmec5).
- Banheiro (possível fonte de `barra_apoio`): [Bathroom small item](https://universe.roboflow.com/udrestroom/bathroom-small-item).

### O que a busca não encontrou

- **`vaga_pcd`.** Os resultados para "disabled parking" são, pelos nomes das classes (`disabled_parking`, `highpass`, `blackbox`, `navigator`), datasets de adesivos e aparelhos no para-brisa de carros, não de vagas pintadas no chão. Exemplos: [Disabled Sign Detection](https://universe.roboflow.com/bekhzod-olimov/disabled-sign-detection) e [Disabled Parking Sign](https://universe.roboflow.com/disabled-parking-sign/disabled-parking-sign-k66on).
- **`barra_apoio`.** Nenhum dataset específico.

Essas duas classes, e provavelmente `rampa`, dependem das fotos do grupo. É mais um motivo para começar pelo escopo de quatro classes.

## Fotos próprias

São o diferencial do projeto: fazem o modelo aprender o contexto brasileiro (piso tátil amarelo, placas no padrão da NBR 9050).

- **Onde:** estações de metrô, shoppings, calçadas, banheiros e estacionamentos da faculdade.
- **Quantas:** a meta é de 300 a 800 imagens por classe, somando as fontes públicas.
- **Variedade:** ângulos e distâncias diferentes, ambientes internos e externos, dia e noite, recursos parcialmente cobertos.
- **Negativas:** fotos de corredores, calçadas e fachadas sem nenhum recurso. Vão em uma pasta à parte e não precisam de anotação.
- **Privacidade:** evite enquadrar rostos e placas de veículos, ou borre antes de subir as fotos. Confira também se o projeto no Roboflow está público ou privado.
- **Autorização:** alguns locais, como metrô e shoppings, podem exigir permissão para fotografar.

## Anotação

1. Suba as fotos para um projeto de *Object Detection* no Roboflow (ou no CVAT) e crie as classes com os nomes finais: `rampa`, `escada`, `corrimao`, `elevador`, `barra_apoio`, `vaga_pcd`, `piso_tatil`, `simbolo_acessibilidade`.
2. Desenhe uma caixa justa em volta de cada recurso. Anote **todos** os recursos da foto, inclusive os pequenos e os parcialmente cobertos.
3. Gere uma versão **sem augmentation** e sem divisão especial: a divisão em 70/20/10 e a augmentation são feitas depois, pela preparação e pelo treino.
4. Exporte no formato YOLOv11 e extraia em `dataset/brutos/fotos-proprias/`.

Se sobrar tempo, importe também os datasets públicos para o mesmo projeto e complete as anotações que faltam (por exemplo, os corrimãos das fotos de um dataset que só marca escadas), pelo menos nas imagens de validação e de teste.
