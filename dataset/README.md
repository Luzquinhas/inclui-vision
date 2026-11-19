# Dataset

As imagens não ficam no Git. Esta pasta guarda a receita: de onde elas vêm (`fontes.yaml`) e como viram o dataset de treino (`src/preparar_dataset.py`).

```
dataset/
├── fontes.yaml       # fontes e mapa de classes (você edita)
├── data.yaml         # gerado pela preparação
├── brutos/           # um export YOLO por fonte (você baixa)
└── train/ valid/ test/   # gerados pela preparação
```

## Datasets usados no modelo entregue

Quatro datasets públicos do Roboflow Universe, todos sob CC BY 4.0. Baixe cada um no formato YOLOv11 e extraia na pasta indicada.

| Pasta em `brutos/` | Dataset | Classes de origem usadas | Classe no projeto | Imagens no export |
|---|---|---|---|---:|
| `stairs-handrail` | [Stairs \| Handrail Detector](https://universe.roboflow.com/itlabroom/stairs-handrail-detector) (ITLabRoom) | `downstairs`, `handrail`, `upstairs` | `escada`, `corrimao` | 451 |
| `stairs-barrierfree` | [Stairs](https://universe.roboflow.com/barrierfreenavigation/stairs-u437q) (BarrierFreeNavigation) | `stairs` | `escada` | 150 |
| `detector-rampas` | [detector-rampas](https://universe.roboflow.com/lockytobebe/detector-rampas/dataset/2), versão 2 (lockytobebe) | `rampa` e duas variantes do mesmo nome | `rampa` | 2.230 |
| `braille-block` | [Braille Block Detect](https://universe.roboflow.com/yourking/braille-block-detect/dataset/3), versão 3 (yourking) | `braille_block` | `piso_tatil` | 787 |

Classes de origem descartadas: `no_braille_block`, que marca onde não há piso tátil (260 caixas), `braille_block_unusable` (6 caixas) e uma classe sem nome no dataset de rampas (156 caixas). Imagens que ficam sem nenhuma caixa são deixadas de fora.

Depois da preparação, o dataset tem 2.382 imagens de treino, 330 de validação e 165 de teste.

O dataset de rampas é bem maior que os outros e tem muitas fotos parecidas do mesmo local. Os exports do Roboflow já trazem cópias aumentadas; a preparação mantém as cópias de uma foto no mesmo conjunto e deixa só uma delas na validação e no teste.

## Como ampliar

### Classes que ficaram de fora

Candidatos que apareceram na busca no Roboflow Universe em 19/11/2026. Antes de usar qualquer um, abra a página, olhe as imagens e confira a licença.

- `rampa`, `escada` e `simbolo_acessibilidade`: [Disabilities](https://universe.roboflow.com/testworkspace-pzhbu/disabilities), com 201 imagens e as classes `ramp`, `stairs` e `disability_sign`, CC BY 4.0.
- `elevador`: [elevator](https://universe.roboflow.com/lmw/elevator-mmec5), com 3.589 imagens de portas (`open`, `close`, `middle`), CC BY 4.0. É muito maior que as outras fontes.
- `vaga_pcd`: nada adequado. Os resultados para "disabled parking" são, pelos nomes das classes, datasets de adesivos e aparelhos no para-brisa de carros, não de vagas pintadas no chão.
- `barra_apoio`: nenhum dataset específico.

### Fotos próprias

O modelo entregue não usa fotos do grupo. Elas são o próximo passo natural: fazem o modelo aprender o contexto brasileiro (piso tátil amarelo, placas no padrão da NBR 9050).

- **Onde:** estações de metrô, shoppings, calçadas, banheiros e estacionamentos da faculdade.
- **Quantas:** a meta é de 300 a 800 imagens por classe, somando as fontes públicas. Hoje só `corrimao` está abaixo disso, com 231 no treino.
- **Variedade:** ângulos e distâncias diferentes, ambientes internos e externos, dia e noite, recursos parcialmente cobertos.
- **Negativas:** fotos de corredores, calçadas e fachadas sem nenhum recurso. Vão em uma pasta à parte e não precisam de anotação.
- **Privacidade:** evite enquadrar rostos e placas de veículos, ou borre antes de subir as fotos. Confira também se o projeto no Roboflow está público ou privado.
- **Autorização:** alguns locais, como metrô e shoppings, podem exigir permissão para fotografar.

### Anotação das fotos próprias

1. Suba as fotos para um projeto de *Object Detection* no Roboflow (ou no CVAT) e crie as classes com os nomes finais: `rampa`, `escada`, `corrimao` e `piso_tatil`, mais as que forem acrescentadas ao `fontes.yaml`.
2. Desenhe uma caixa justa em volta de cada recurso. Anote **todos** os recursos da foto, inclusive os pequenos e os parcialmente cobertos.
3. Gere uma versão **sem augmentation** e sem divisão especial: a divisão em 70/20/10 e a augmentation são feitas depois, pela preparação e pelo treino.
4. Exporte no formato YOLOv11, extraia em `dataset/brutos/fotos-proprias/` e acrescente ao `fontes.yaml` uma fonte com `origem: proprio`. Fotos anotadas com os nomes finais não precisam de mapa.

Vale também importar os datasets públicos para o mesmo projeto e completar as anotações que faltam (por exemplo, os corrimãos das fotos de um dataset que só marca escadas), pelo menos nas imagens de validação e de teste.
