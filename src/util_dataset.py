"""Funções pequenas para ler datasets no formato YOLO, usadas pelos outros scripts."""
from __future__ import annotations

from pathlib import Path

import yaml

EXT_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def listar_imagens(pasta: Path) -> list[Path]:
    """Todas as imagens dentro de uma pasta, em ordem estável."""
    return sorted(p for p in Path(pasta).rglob("*")
                  if p.is_file() and p.suffix.lower() in EXT_IMAGEM)


def caminho_rotulo(imagem: Path) -> Path:
    """Arquivo de rótulo de uma imagem: troca a última pasta 'images' por 'labels'.

    É a mesma regra do Ultralytics. Se a imagem não está numa pasta 'images',
    o rótulo é procurado ao lado dela.
    """
    partes = list(Path(imagem).parts)
    for i in range(len(partes) - 2, -1, -1):
        if partes[i] == "images":
            partes[i] = "labels"
            return Path(*partes).with_suffix(".txt")
    return Path(imagem).with_suffix(".txt")


def ler_nomes_de_yaml(arquivo: Path) -> dict[int, str]:
    """Classes de um data.yaml, como {id: nome}. Aceita lista ou dicionário."""
    dados = yaml.safe_load(Path(arquivo).read_text(encoding="utf-8")) or {}
    nomes = dados.get("names") if isinstance(dados, dict) else None
    if isinstance(nomes, dict):
        return {int(k): str(v) for k, v in nomes.items()}
    if isinstance(nomes, list):
        return {i: str(n) for i, n in enumerate(nomes)}
    return {}


def imagens_do_conjunto(data_yaml: Path, conjunto: str) -> list[Path]:
    """Imagens de um conjunto ('train', 'val' ou 'test') descrito em um data.yaml."""
    data_yaml = Path(data_yaml).resolve()
    dados = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    raiz = Path(dados.get("path") or data_yaml.parent)
    if not raiz.is_absolute():
        raiz = raiz.resolve() if raiz.exists() else (data_yaml.parent / raiz).resolve()

    entrada = dados.get(conjunto)
    if not entrada:
        raise ValueError(f"O conjunto '{conjunto}' não está definido em {data_yaml}.")
    if isinstance(entrada, (list, tuple)):
        entradas = list(entrada)
    else:
        entradas = [entrada]

    imagens: list[Path] = []
    for item in entradas:
        caminho = Path(item)
        if not caminho.is_absolute():
            caminho = raiz / caminho
        if caminho.suffix.lower() == ".txt":
            for linha in caminho.read_text(encoding="utf-8").splitlines():
                linha = linha.strip()
                if not linha:
                    continue
                # Caminhos que começam com ./ são relativos ao próprio arquivo .txt.
                alvo = Path(linha)
                imagens.append(alvo if alvo.is_absolute() else caminho.parent / alvo)
        elif caminho.is_dir():
            imagens.extend(listar_imagens(caminho))
        elif caminho.is_file():
            imagens.append(caminho)
    return imagens


def ler_rotulos(imagem: Path) -> list[tuple[int, float, float, float, float]]:
    """Caixas (classe, x_centro, y_centro, largura, altura) de uma imagem já preparada."""
    arquivo = caminho_rotulo(imagem)
    if not arquivo.exists():
        return []
    caixas = []
    for linha in arquivo.read_text(encoding="utf-8").splitlines():
        partes = linha.split()
        if len(partes) == 5:
            caixas.append((int(float(partes[0])), *(float(v) for v in partes[1:])))
    return caixas
