"""Monta uma grade de imagens com as anotações desenhadas, para conferir o dataset antes de treinar.

Um mapa de classes errado (por exemplo, rampa trocada com escada) aparece aqui em segundos,
e não só depois de horas de treino. A grade também serve de figura para o relatório.

Uso (a partir da raiz do repositório):
    python src/ver_amostras.py                              # 12 imagens do treino
    python src/ver_amostras.py --split val --n 8 --saida results/amostras_val.jpg
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from categorias import sem_acento
from util_dataset import imagens_do_conjunto, ler_nomes_de_yaml, ler_rotulos

CORES = [(66, 135, 245), (60, 200, 80), (40, 170, 240), (220, 100, 60),
         (180, 80, 200), (60, 60, 230), (0, 215, 255), (200, 200, 60)]


def desenhar_rotulos(imagem, caixas, nomes: dict):
    """Desenha as caixas (classe, xc, yc, w, h normalizados) e o nome de cada classe."""
    altura, largura = imagem.shape[:2]
    espessura = max(2, round(min(altura, largura) / 250))
    escala = max(0.5, min(altura, largura) / 900)
    for classe, xc, yc, w, h in caixas:
        x1, y1 = int((xc - w / 2) * largura), int((yc - h / 2) * altura)
        x2, y2 = int((xc + w / 2) * largura), int((yc + h / 2) * altura)
        cor = CORES[classe % len(CORES)]
        cv2.rectangle(imagem, (x1, y1), (x2, y2), cor, espessura)
        texto = sem_acento(str(nomes.get(classe, classe)))
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, espessura)
        topo = max(y1 - th - 8, 0)
        cv2.rectangle(imagem, (x1, topo), (x1 + tw + 8, topo + th + 8), cor, -1)
        cv2.putText(imagem, texto, (x1 + 4, topo + th + 3), cv2.FONT_HERSHEY_SIMPLEX,
                    escala, (255, 255, 255), espessura, cv2.LINE_AA)
    return imagem


def encaixar(imagem, lado: int):
    """Redimensiona mantendo a proporção e completa com cinza até um quadrado."""
    altura, largura = imagem.shape[:2]
    fator = lado / max(altura, largura)
    nova = cv2.resize(imagem, (max(1, round(largura * fator)), max(1, round(altura * fator))))
    quadro = np.full((lado, lado, 3), 114, dtype=np.uint8)
    y0, x0 = (lado - nova.shape[0]) // 2, (lado - nova.shape[1]) // 2
    quadro[y0:y0 + nova.shape[0], x0:x0 + nova.shape[1]] = nova
    return quadro


def montar_grade(imagens: list, colunas: int = 4, lado: int = 480):
    blocos = [encaixar(img, lado) for img in imagens]
    while len(blocos) % colunas:
        blocos.append(np.full((lado, lado, 3), 114, dtype=np.uint8))
    linhas = [np.hstack(blocos[i:i + colunas]) for i in range(0, len(blocos), colunas)]
    return np.vstack(linhas)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Inclui+ Vision: confere as anotações do dataset.")
    p.add_argument("--data", default="dataset/data.yaml", type=Path)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--colunas", type=int, default=4)
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--saida", default="results/amostras_treino.jpg", type=Path)
    args = p.parse_args(argv)

    nomes = ler_nomes_de_yaml(args.data)
    candidatas = sorted(imagens_do_conjunto(args.data, args.split))
    random.Random(args.semente).shuffle(candidatas)

    prontas = []
    for caminho in candidatas:
        imagem = cv2.imread(str(caminho))
        if imagem is None:
            print(f"AVISO: não consegui abrir {caminho}", file=sys.stderr)
            continue
        prontas.append(desenhar_rotulos(imagem, ler_rotulos(caminho), nomes))
        if len(prontas) == args.n:
            break
    if not prontas:
        print("ERRO: nenhuma imagem encontrada nesse conjunto.", file=sys.stderr)
        return 1

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.saida), montar_grade(prontas, args.colunas))
    print(f"Grade com {len(prontas)} imagens salva em {args.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
