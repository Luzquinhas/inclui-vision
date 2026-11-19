"""Relatório de acessibilidade de uma foto, agrupado pelas categorias do Inclui+.

Uso (a partir da raiz do repositório):
    python src/relatorio.py foto.jpg
    python src/relatorio.py foto.jpg --saida results/exemplos/foto.jpg
    python src/relatorio.py foto.jpg --json        # saída estruturada, para integrar à plataforma
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from categorias import agrupar_por_categoria, barreiras


def montar_relatorio(contagem) -> dict:
    """Transforma {classe: quantidade} no relatório estruturado."""
    contagem = {classe: int(n) for classe, n in dict(contagem).items() if n > 0}
    return {
        "total_deteccoes": sum(contagem.values()),
        "categorias": agrupar_por_categoria(contagem),
        "barreiras": barreiras(contagem),
    }


def formatar(relatorio: dict) -> str:
    linhas = ["Relatório de acessibilidade — Inclui+"]
    if not relatorio["categorias"]:
        linhas.append("  Nenhum recurso de acessibilidade detectado na imagem.")
    for nome_categoria, itens in relatorio["categorias"].items():
        for classe, n in itens.items():
            linhas.append(f"  [{nome_categoria}] {classe}: {n}")
    for barreira in relatorio["barreiras"]:
        linhas.append(f"  ATENÇÃO, barreira: {barreira}")
    return "\n".join(linhas)


def ler_argumentos(argv=None):
    p = argparse.ArgumentParser(description="Inclui+ Vision: relatório de uma foto.")
    p.add_argument("imagem")
    p.add_argument("--pesos", default="weights/best.pt", help="modelo treinado")
    p.add_argument("--conf", type=float, default=0.5, help="confiança mínima")
    p.add_argument("--saida", default="resultado.jpg", help="imagem com as bounding boxes")
    p.add_argument("--json", action="store_true", help="imprime o relatório em JSON")
    return p.parse_args(argv)


def main(argv=None):
    args = ler_argumentos(argv)
    from ultralytics import YOLO  # importado aqui para o resto do módulo não depender dele

    if hasattr(sys.stdout, "reconfigure"):  # acentos no terminal do Windows
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    modelo = YOLO(args.pesos)
    resultado = modelo(args.imagem, conf=args.conf, verbose=False)[0]
    contagem = Counter(resultado.names[int(c)] for c in resultado.boxes.cls.tolist())
    relatorio = montar_relatorio(contagem)

    resultado.save(args.saida)
    if args.json:
        relatorio["imagem"] = args.imagem
        relatorio["imagem_anotada"] = args.saida
        print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    else:
        print(formatar(relatorio))
        print(f"  Imagem anotada: {args.saida}")


if __name__ == "__main__":
    main()
