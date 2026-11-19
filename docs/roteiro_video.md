# Roteiro do vídeo de demonstração (3 a 5 minutos)

Grave em locais reais (faculdade, metrô, shopping). A demonstração convence muito mais do que imagens da internet na tela.

## Antes de gravar

- [ ] `weights/best.pt` no lugar e `python src/realtime.py` abrindo a câmera.
- [ ] Celular funcionando como câmera (DroidCam ou Iriun), ou o plano B: gravar com o celular e processar depois com `python src/realtime.py --fonte gravacao.mp4 --gravar demo.mp4`.
- [ ] Percurso ensaiado: escada, rampa com corrimão e piso tátil, de preferência perto uns dos outros.
- [ ] Uma foto de fachada ou de entrada de estabelecimento para o `relatorio.py`.
- [ ] Números de `results/metricas.md` à mão.

## Cenas

| # | Duração | Na tela | Fala |
|---|---|---|---|
| 1 | 30 s | Título do projeto e a tela do Inclui+ | O Censo 2022 contou 14,4 milhões de pessoas com deficiência no Brasil. Hoje o Inclui+ depende de descrições manuais dos lugares. A proposta é a foto virar a evidência. |
| 2 | 45 s | Os datasets no Roboflow Universe; a grade `results/amostras_treino.jpg`; o notebook no Colab | Juntamos quatro datasets públicos, unificamos as classes e dividimos em treino, validação e teste. O YOLO11n parte dos pesos do COCO e aprende as quatro classes em 100 épocas. |
| 3 | 30 s | `results/results.png`, a tabela por classe e `confusion_matrix.png` | mAP@50 de 0,797 no conjunto de teste. As classes mais fortes são piso tátil (0,963) e rampa (0,928). A mais fraca é corrimão (0,511): tem menos imagens e é um objeto fino. |
| 4 | 1 a 2 min | `realtime.py` em tela cheia, câmera na mão | Ver o passo a passo abaixo. |
| 5 | 30 s | Terminal rodando `python src/relatorio.py foto.jpg` e a imagem anotada | A partir de uma única foto enviada pelo usuário, o sistema devolve o que encontrou, já nas categorias da plataforma. |
| 6 | 20 s | Tabela "Parte do Inclui+ × como a visão computacional ajuda" | Isso fortalece a colaboração, dá lastro ao selo e mede o KPI de recursos novos com fotos de antes e depois. |

### Cena 4, passo a passo

1. Aponte para uma **escada**. Aparecem a caixa, a linha `Mobilidade: escada` e o alerta em vermelho de escada sem rampa.
2. Vire para uma **rampa com corrimão**. Duas detecções em Mobilidade, e o alerta some.
3. Mostre o **piso tátil**: `Visao: piso_tatil`.
4. Volte ao **corrimão** e mostre que a detecção falha ou oscila: é a classe mais fraca do modelo.
5. Deixe o contador de FPS visível em algum momento e cite o valor.

Diga por que o modelo erra onde erra: ele foi treinado só com imagens públicas, sem fotos de locais brasileiros. Isso prepara a seção de limitações do relatório.
