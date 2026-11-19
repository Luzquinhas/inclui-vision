"""Passo 4: avalia o modelo no conjunto de teste e salva o que o relatório precisa.

Gera em --saida:
    metricas.md, metricas.json, metricas_por_classe.csv
    confusion_matrix.png, confusion_matrix_normalized.png e as curvas de precisão e recall
    exemplos/    imagens do conjunto avaliado com as bounding boxes previstas
e atualiza results/comparacao.csv, uma linha por modelo avaliado (yolo11n, yolo11s...).

Uso (a partir da raiz do repositório):
    python src/avaliar.py --pesos weights/best.pt --data dataset/data.yaml
    python src/avaliar.py --pesos runs/exp_s/weights/best.pt --rotulo yolo11s --saida results/yolo11s
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

from util_dataset import imagens_do_conjunto

COLUNAS_COMPARACAO = ["rotulo", "conjunto", "mAP50", "mAP50-95", "precisao", "recall",
                      "ms_por_imagem", "fps", "parametros_M", "pesos"]
GRAFICOS = ("confusion_matrix*.png", "*PR_curve.png", "*F1_curve.png", "*P_curve.png", "*R_curve.png")


def tabela_por_classe(metrics) -> list[dict]:
    """Precisão, recall e mAP de cada classe que aparece no conjunto avaliado."""
    linhas = []
    for posicao, id_classe in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(posicao)
        linhas.append({
            "classe": metrics.names[int(id_classe)],
            "precisao": float(p), "recall": float(r),
            "mAP50": float(ap50), "mAP50-95": float(ap),
        })
    return linhas


def resumo_geral(metrics) -> dict:
    velocidade = dict(getattr(metrics, "speed", {}) or {})
    ms = sum(float(velocidade.get(k, 0.0)) for k in ("preprocess", "inference", "postprocess"))
    return {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precisao": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "ms_por_imagem": ms,
        "fps": 1000.0 / ms if ms > 0 else 0.0,
    }


def para_markdown(geral: dict, por_classe: list[dict], rotulo: str, conjunto: str) -> str:
    linhas = [
        f"### Métricas gerais ({rotulo}, conjunto de {conjunto})",
        "",
        "| mAP@50 | mAP@50-95 | Precisão | Recall | Tempo por imagem | FPS |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {geral['mAP50']:.3f} | {geral['mAP50-95']:.3f} | {geral['precisao']:.3f} | "
        f"{geral['recall']:.3f} | {geral['ms_por_imagem']:.1f} ms | {geral['fps']:.0f} |",
        "",
        "### Métricas por classe",
        "",
        "| Classe | Precisão | Recall | mAP@50 | mAP@50-95 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for l in por_classe:
        linhas.append(f"| `{l['classe']}` | {l['precisao']:.3f} | {l['recall']:.3f} | "
                      f"{l['mAP50']:.3f} | {l['mAP50-95']:.3f} |")
    return "\n".join(linhas) + "\n"


def atualizar_comparacao(arquivo: Path, linha: dict) -> list[dict]:
    """Grava a linha deste modelo no CSV de comparação, substituindo a de mesmo rótulo."""
    linhas = []
    if arquivo.exists():
        with arquivo.open(newline="", encoding="utf-8") as f:
            linhas = [l for l in csv.DictReader(f) if l.get("rotulo") != linha["rotulo"]]
    linhas.append({c: linha.get(c, "") for c in COLUNAS_COMPARACAO})
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    with arquivo.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS_COMPARACAO)
        w.writeheader()
        w.writerows(linhas)
    return linhas


def copiar_graficos(origem: Path, destino: Path) -> list[str]:
    arquivos = {a.name: a for padrao in GRAFICOS for a in Path(origem).glob(padrao)}
    for nome, arquivo in sorted(arquivos.items()):
        shutil.copy2(arquivo, destino / nome)
    return sorted(arquivos)


def salvar_exemplos(modelo, imagens: list[Path], destino: Path, n: int, conf: float, semente: int) -> int:
    destino.mkdir(parents=True, exist_ok=True)
    amostra = sorted(imagens)
    random.Random(semente).shuffle(amostra)
    amostra = amostra[:n]
    if not amostra:
        return 0
    for resultado in modelo.predict(source=[str(p) for p in amostra], conf=conf, verbose=False):
        resultado.save(filename=str(destino / f"{Path(resultado.path).stem}.jpg"))
    return len(amostra)


def ler_argumentos(argv=None):
    p = argparse.ArgumentParser(description="Inclui+ Vision: avaliação do modelo.")
    p.add_argument("--pesos", default="weights/best.pt")
    p.add_argument("--data", default="dataset/data.yaml")
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--saida", default="results", type=Path)
    p.add_argument("--rotulo", default="inclui_vision", help="nome deste modelo na tabela de comparação")
    p.add_argument("--comparacao", default="results/comparacao.csv", type=Path)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--exemplos", type=int, default=12, help="quantas imagens anotadas salvar")
    p.add_argument("--conf", type=float, default=0.5, help="confiança mínima nos exemplos")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = ler_argumentos(argv)
    from ultralytics import YOLO  # importado aqui para o resto do módulo não depender dele

    if hasattr(sys.stdout, "reconfigure"):  # acentos no terminal do Windows
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    saida = args.saida.resolve()
    saida.mkdir(parents=True, exist_ok=True)
    data = Path(args.data).resolve()

    modelo = YOLO(args.pesos)
    metrics = modelo.val(data=str(data), split=args.split, imgsz=args.imgsz, batch=args.batch,
                         plots=True, project=str(saida / "validacao"), name=args.rotulo,
                         exist_ok=True, verbose=False)

    geral = resumo_geral(metrics)
    por_classe = tabela_por_classe(metrics)
    nomes_conjunto = {"test": "teste", "val": "validação", "train": "treino"}
    texto = para_markdown(geral, por_classe, args.rotulo, nomes_conjunto[args.split])

    (saida / "metricas.md").write_text(texto, encoding="utf-8")
    (saida / "metricas.json").write_text(
        json.dumps({"rotulo": args.rotulo, "conjunto": args.split, "geral": geral,
                    "por_classe": por_classe}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (saida / "metricas_por_classe.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["classe", "precisao", "recall", "mAP50", "mAP50-95"])
        w.writeheader()
        w.writerows(por_classe)

    parametros = sum(p.numel() for p in modelo.model.parameters()) / 1e6
    atualizar_comparacao(args.comparacao, {
        "rotulo": args.rotulo, "conjunto": args.split, "pesos": args.pesos,
        "mAP50": f"{geral['mAP50']:.4f}", "mAP50-95": f"{geral['mAP50-95']:.4f}",
        "precisao": f"{geral['precisao']:.4f}", "recall": f"{geral['recall']:.4f}",
        "ms_por_imagem": f"{geral['ms_por_imagem']:.2f}", "fps": f"{geral['fps']:.1f}",
        "parametros_M": f"{parametros:.2f}",
    })

    graficos = copiar_graficos(Path(metrics.save_dir), saida)
    n = salvar_exemplos(modelo, imagens_do_conjunto(data, args.split), saida / "exemplos",
                        args.exemplos, args.conf, semente=0)

    print(texto)
    print(f"Gráficos copiados para {saida}: {', '.join(graficos) or 'nenhum'}")
    print(f"{n} exemplos com bounding boxes em {saida / 'exemplos'}")
    print(f"Tabela de comparação atualizada: {args.comparacao}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
