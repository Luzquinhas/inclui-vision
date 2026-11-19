"""Passos 1 e 2: une os datasets, unifica as classes e divide em treino, validação e teste.

Entrada: pastas no formato YOLO (o export "YOLOv8" ou "YOLOv11" do Roboflow, ou o do CVAT),
listadas em dataset/fontes.yaml junto com o mapa de classes de cada uma.

Saída, dentro de --saida:
    train/ valid/ test/    images/ e labels/ com os IDs de classe já unificados
    data.yaml              treino com todas as fontes
    data_publico.yaml      mesma validação e mesmo teste, treino só com as fontes públicas
    manifesto.csv          de onde veio cada imagem e em que conjunto caiu
    resumo.md              imagens e caixas por classe, pronto para colar no relatório

Uso (a partir da raiz do repositório):
    python src/preparar_dataset.py --config dataset/fontes.yaml --saida dataset --limpar
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from categorias import CLASSES as CLASSES_CONHECIDAS
from util_dataset import caminho_rotulo, ler_nomes_de_yaml, listar_imagens

CONJUNTOS = ("train", "valid", "test")
NOME_CONJUNTO = {"train": "treino", "valid": "validação", "test": "teste"}
CHAVE_DIVISAO = {"train": "treino", "valid": "validacao", "test": "teste"}
SUFIXO_ROBOFLOW = re.compile(r"\.rf\.[0-9a-zA-Z]+$")
META_MINIMA = 300  # a proposta pede de 300 a 800 imagens por classe


class ErroDeConfiguracao(Exception):
    """Problema no fontes.yaml ou nas pastas de origem; a mensagem diz como corrigir."""


@dataclass
class Item:
    imagem: Path
    fonte: str
    origem: str
    grupo: str
    caixas: list
    conjunto: str = ""
    destino: str = ""


@dataclass
class Estatisticas:
    imagens_lidas: int = 0
    imagens_descartadas: int = 0
    linhas_invalidas: int = 0
    caixas_descartadas: Counter = field(default_factory=Counter)
    classes_sem_mapa: list = field(default_factory=list)


# --------------------------------------------------------------------------- leitura

def ler_nomes(pasta: Path, declarados=None) -> dict[int, str]:
    """Classes de uma fonte: do fontes.yaml, do data.yaml dela ou do classes.txt (CVAT)."""
    if declarados:
        return {i: str(n) for i, n in enumerate(declarados)}
    # O data.yaml costuma estar na raiz do export; se o .zip tinha uma pasta a mais, procura dentro.
    candidatos = sorted(pasta.glob("*.yaml")) + sorted(pasta.glob("*.yml")) + sorted(pasta.rglob("data.yaml"))
    for arquivo in candidatos:
        nomes = ler_nomes_de_yaml(arquivo)
        if nomes:
            return nomes
    for nome_arquivo in ("classes.txt", "obj.names"):
        achados = sorted(pasta.rglob(nome_arquivo))
        if achados:
            linhas = achados[0].read_text(encoding="utf-8").splitlines()
            return {i: l.strip() for i, l in enumerate(l for l in linhas if l.strip())}
    raise ErroDeConfiguracao(
        f"Não achei as classes de '{pasta}'. Deixe o data.yaml do export nessa pasta "
        "ou declare 'nomes: [...]' para essa fonte no fontes.yaml."
    )


def montar_remap(nomes: dict[int, str], mapa, classes: list[str], fonte: str = ""):
    """Traduz os IDs de classe da fonte para os IDs finais.

    Devolve ({id_origem: id_final ou None}, [classes da fonte que ficaram sem destino]).
    None significa que as caixas dessa classe são descartadas.
    """
    alvo = {c: i for i, c in enumerate(classes)}
    mapa_norm = {str(k).strip().lower(): v for k, v in (mapa or {}).items()}
    nomes_norm = {n.strip().lower() for n in nomes.values()}

    desconhecidas = sorted(set(mapa_norm) - nomes_norm)
    if desconhecidas:
        raise ErroDeConfiguracao(
            f"Fonte '{fonte}': o mapa cita classes que não existem nela: {desconhecidas}. "
            f"As classes dessa fonte são: {sorted(nomes.values())}."
        )

    remap: dict[int, int | None] = {}
    sem_mapa: list[str] = []
    for id_origem, nome in nomes.items():
        chave = nome.strip().lower()
        if chave in mapa_norm:
            destino = mapa_norm[chave]
        elif chave in alvo:  # o nome já é o final
            destino = chave
        else:
            destino = None
            sem_mapa.append(nome)

        if destino is None:
            remap[id_origem] = None
            continue
        destino = str(destino).strip()
        if destino in alvo:
            remap[id_origem] = alvo[destino]
        elif destino in CLASSES_CONHECIDAS:
            remap[id_origem] = None  # classe fora do escopo escolhido em 'classes'
        else:
            raise ErroDeConfiguracao(
                f"Fonte '{fonte}': '{nome}' aponta para '{destino}', que não está em "
                f"'classes' ({classes}). Confira se o nome foi digitado certo."
            )
    return remap, sem_mapa


def ler_caixas(texto: str, remap: dict):
    """Lê um arquivo de rótulo YOLO e devolve as caixas já com os IDs finais.

    Aceita caixas (classe + 4 valores) e polígonos de segmentação (classe + pares x y),
    que viram a caixa que envolve o polígono.
    Devolve (caixas, contagem de caixas descartadas por id de origem, linhas inválidas).
    """
    caixas, descartadas, invalidas = [], Counter(), 0
    for linha in texto.splitlines():
        partes = linha.split()
        if not partes:
            continue
        try:
            id_origem = int(float(partes[0]))
            valores = [float(v) for v in partes[1:]]
        except ValueError:
            invalidas += 1
            continue

        if len(valores) == 4:
            xc, yc, w, h = valores
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
        elif len(valores) >= 6 and len(valores) % 2 == 0:
            xs, ys = valores[0::2], valores[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            invalidas += 1
            continue

        x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
        if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6 or id_origem not in remap:
            invalidas += 1
            continue

        destino = remap[id_origem]
        if destino is None:
            descartadas[id_origem] += 1
            continue
        caixas.append((destino, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return caixas, descartadas, invalidas


def grupo_de(imagem: Path, pasta: Path) -> str:
    """Chave que mantém juntas as cópias aumentadas de uma mesma foto.

    O Roboflow nomeia cada cópia como '<original>.rf.<hash>.jpg'. Se elas fossem
    divididas ao acaso, a mesma foto apareceria no treino e no teste.
    """
    if SUFIXO_ROBOFLOW.search(imagem.stem):
        return SUFIXO_ROBOFLOW.sub("", imagem.stem)
    return imagem.relative_to(pasta).as_posix()


def carregar_fonte(cfg: dict, base: Path, classes: list[str], manter_vazias: bool):
    if "pasta" not in cfg:
        raise ErroDeConfiguracao("Toda fonte precisa do campo 'pasta'.")
    pasta = Path(cfg["pasta"])
    if not pasta.is_absolute():
        pasta = (base / pasta).resolve()
    nome = str(cfg.get("nome") or pasta.name)
    origem = str(cfg.get("origem", "publico")).strip().lower()
    if origem not in ("publico", "proprio"):
        raise ErroDeConfiguracao(f"Fonte '{nome}': 'origem' deve ser 'publico' ou 'proprio'.")
    if not pasta.is_dir():
        raise ErroDeConfiguracao(
            f"Fonte '{nome}': a pasta '{pasta}' não existe. Coloque o export nela "
            "ou apague essa fonte do fontes.yaml."
        )

    so_negativos = bool(cfg.get("negativos", False))
    estat = Estatisticas()
    nomes: dict[int, str] = {}
    remap: dict = {}
    if not so_negativos:
        nomes = ler_nomes(pasta, cfg.get("nomes"))
        remap, estat.classes_sem_mapa = montar_remap(nomes, cfg.get("mapa"), classes, nome)

    itens: list[Item] = []
    for imagem in listar_imagens(pasta):
        estat.imagens_lidas += 1
        rotulo = caminho_rotulo(imagem)
        caixas, tinha_rotulo = [], False
        if not so_negativos and rotulo.exists():
            texto = rotulo.read_text(encoding="utf-8", errors="replace")
            tinha_rotulo = bool(texto.strip())
            caixas, descartadas, invalidas = ler_caixas(texto, remap)
            estat.linhas_invalidas += invalidas
            for id_origem, n in descartadas.items():
                estat.caixas_descartadas[nomes.get(id_origem, str(id_origem))] += n
        # Imagem que só tinha classes fora do escopo não serve como exemplo negativo:
        # ela pode conter recursos que a fonte original simplesmente não anotou.
        if tinha_rotulo and not caixas and not manter_vazias:
            estat.imagens_descartadas += 1
            continue
        itens.append(Item(imagem, nome, origem, grupo_de(imagem, pasta), caixas))
    return itens, estat


# --------------------------------------------------------------------------- divisão

def dividir(itens: list[Item], fracoes: dict, semente) -> None:
    """Sorteia o conjunto de cada imagem sem separar as cópias de uma mesma foto.

    O sorteio é feito por fonte e, dentro dela, pela classe mais rara de cada foto,
    para que as classes com poucas imagens também apareçam na validação e no teste.
    """
    frequencia: Counter = Counter()
    for item in itens:
        for classe in {c[0] for c in item.caixas}:
            frequencia[classe] += 1

    grupos: dict = defaultdict(list)  # (fonte, grupo) -> imagens
    for item in itens:
        grupos[(item.fonte, item.grupo)].append(item)

    estratos: dict = defaultdict(list)  # (fonte, classe mais rara) -> grupos
    for chave, membros in grupos.items():
        presentes = {c[0] for m in membros for c in m.caixas}
        rara = min(presentes, key=lambda c: (frequencia[c], c)) if presentes else -1
        estratos[(chave[0], rara)].append(chave)

    for estrato in sorted(estratos):
        chaves = sorted(estratos[estrato])
        random.Random(f"{semente}:{estrato[0]}:{estrato[1]}").shuffle(chaves)
        n = len(chaves)
        n_teste = round(n * fracoes["test"])
        n_val = round(n * fracoes["valid"])
        if n >= 3:  # estratos pequenos ainda aparecem na validação e no teste
            n_teste = max(n_teste, 1) if fracoes["test"] > 0 else 0
            n_val = max(n_val, 1) if fracoes["valid"] > 0 else 0
        if fracoes["train"] > 0:
            while n_teste + n_val >= n and (n_teste or n_val):
                if n_val >= n_teste:
                    n_val -= 1
                else:
                    n_teste -= 1
        for i, chave in enumerate(chaves):
            conjunto = "test" if i < n_teste else "valid" if i < n_teste + n_val else "train"
            for item in grupos[chave]:
                item.conjunto = conjunto


def enxugar_avaliacao(itens: list[Item]):
    """Na validação e no teste fica só uma imagem por grupo.

    As outras são cópias aumentadas da mesma foto: inflariam o conjunto sem trazer
    informação nova, e a proposta pede augmentation apenas no treino.
    """
    vistos, mantidos, removidas = set(), [], 0
    for item in sorted(itens, key=lambda i: (i.fonte, i.imagem.as_posix())):
        if item.conjunto in ("valid", "test"):
            chave = (item.fonte, item.grupo)
            if chave in vistos:
                removidas += 1
                continue
            vistos.add(chave)
        mantidos.append(item)
    return mantidos, removidas


# --------------------------------------------------------------------------- escrita

def slug(texto: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", texto).strip("-").lower() or "fonte"


def gravar(itens: list[Item], saida: Path) -> None:
    for conjunto in CONJUNTOS:
        (saida / conjunto / "images").mkdir(parents=True, exist_ok=True)
        (saida / conjunto / "labels").mkdir(parents=True, exist_ok=True)

    usados: set[str] = set()
    for item in itens:
        base = f"{slug(item.fonte)}__{item.imagem.stem}"
        extensao = item.imagem.suffix.lower()
        nome, k = base, 1
        while nome.lower() in usados:
            k += 1
            nome = f"{base}_{k}"
        usados.add(nome.lower())
        item.destino = f"{nome}{extensao}"

        shutil.copy2(item.imagem, saida / item.conjunto / "images" / item.destino)
        linhas = "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in item.caixas)
        (saida / item.conjunto / "labels" / f"{nome}.txt").write_text(linhas, encoding="utf-8")


def escrever_yaml(arquivo: Path, classes: list[str], treino: str, comentario: str) -> None:
    conteudo = {
        "train": treino,
        "val": "valid/images",
        "test": "test/images",
        "names": {i: c for i, c in enumerate(classes)},
    }
    texto = f"# {comentario}\n# Sem a chave 'path', os caminhos valem a partir da pasta deste arquivo.\n"
    texto += yaml.safe_dump(conteudo, sort_keys=False, allow_unicode=True)
    arquivo.write_text(texto, encoding="utf-8")


def escrever_manifesto(itens: list[Item], classes: list[str], arquivo: Path) -> None:
    with arquivo.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "conjunto", "fonte", "origem", "caixas", "classes"])
        for item in sorted(itens, key=lambda i: (i.conjunto, i.destino)):
            presentes = sorted({classes[c[0]] for c in item.caixas})
            w.writerow([item.destino, item.conjunto, item.fonte, item.origem,
                        len(item.caixas), " ".join(presentes)])


def contar(itens: list[Item], classes: list[str]) -> dict:
    cont = {
        "imagens": {c: Counter() for c in CONJUNTOS},
        "caixas": {c: Counter() for c in CONJUNTOS},
        "total": Counter(),
        "negativas": Counter(),
    }
    for item in itens:
        ids = [c[0] for c in item.caixas]
        cont["total"][item.conjunto] += 1
        if not ids:
            cont["negativas"][item.conjunto] += 1
        for i in set(ids):
            cont["imagens"][item.conjunto][i] += 1
        for i in ids:
            cont["caixas"][item.conjunto][i] += 1
    return cont


def resumo_markdown(cont: dict, classes: list[str]) -> str:
    linhas = [
        "| Classe | Imagens no treino | Imagens na validação | Imagens no teste | Caixas (total) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for i, classe in enumerate(classes):
        por_conjunto = [cont["imagens"][c][i] for c in CONJUNTOS]
        caixas = sum(cont["caixas"][c][i] for c in CONJUNTOS)
        linhas.append(f"| `{classe}` | {por_conjunto[0]} | {por_conjunto[1]} | {por_conjunto[2]} | {caixas} |")
    neg = [cont["negativas"][c] for c in CONJUNTOS]
    tot = [cont["total"][c] for c in CONJUNTOS]
    linhas.append(f"| Sem nenhum recurso (negativas) | {neg[0]} | {neg[1]} | {neg[2]} | 0 |")
    todas = sum(sum(cont["caixas"][c].values()) for c in CONJUNTOS)
    linhas.append(f"| **Total de imagens** | **{tot[0]}** | **{tot[1]}** | **{tot[2]}** | **{todas}** |")
    return "\n".join(linhas) + "\n"


def avisos(cont: dict, classes: list[str]) -> list[str]:
    treino = {c: cont["imagens"]["train"][i] for i, c in enumerate(classes)}
    msgs = []
    sem_treino = [c for c in classes if treino[c] == 0]
    if sem_treino:
        msgs.append("Sem nenhuma imagem no treino (o modelo não aprende estas classes): "
                    + ", ".join(sem_treino) + ".")
    poucas = [f"{c} ({treino[c]})" for c in classes if 0 < treino[c] < META_MINIMA]
    if poucas:
        msgs.append(f"Abaixo da meta de {META_MINIMA} a 800 imagens no treino: " + ", ".join(poucas) + ".")
    for conjunto in ("valid", "test"):
        faltam = [c for i, c in enumerate(classes) if treino[c] and cont["imagens"][conjunto][i] == 0]
        if faltam:
            msgs.append(f"Sem imagens no conjunto de {NOME_CONJUNTO[conjunto]} (ficam sem métrica): "
                        + ", ".join(faltam) + ".")
    if not cont["negativas"]["train"]:
        msgs.append("Não há imagens negativas no treino; a proposta pede algumas para reduzir falsos positivos.")
    return msgs


# --------------------------------------------------------------------------- principal

def ler_config(arquivo: Path) -> dict:
    if not arquivo.exists():
        raise ErroDeConfiguracao(f"Arquivo de configuração não encontrado: {arquivo}")
    cfg = yaml.safe_load(arquivo.read_text(encoding="utf-8")) or {}

    classes = [str(c).strip() for c in cfg.get("classes") or []]
    if not classes or len(set(classes)) != len(classes):
        raise ErroDeConfiguracao("'classes' deve ser uma lista sem repetições, com ao menos uma classe.")

    divisao = cfg.get("divisao") or {"treino": 0.7, "validacao": 0.2, "teste": 0.1}
    try:
        fracoes = {c: float(divisao[CHAVE_DIVISAO[c]]) for c in CONJUNTOS}
    except (KeyError, TypeError, ValueError):
        raise ErroDeConfiguracao("'divisao' precisa de 'treino', 'validacao' e 'teste'.") from None
    if abs(sum(fracoes.values()) - 1) > 1e-6 or min(fracoes.values()) < 0:
        raise ErroDeConfiguracao("As frações de 'divisao' devem ser positivas e somar 1.")

    fontes = cfg.get("fontes") or []
    if not fontes:
        raise ErroDeConfiguracao("Nenhuma fonte listada em 'fontes'.")
    return {"classes": classes, "fracoes": fracoes, "semente": cfg.get("semente", 42), "fontes": fontes}


def preparar(config: Path, saida: Path, limpar: bool = False, manter_vazias: bool = False) -> dict:
    cfg = ler_config(config)
    classes = cfg["classes"]

    # As pastas train/, valid/ e test/ da saída são apagadas e recriadas a cada execução,
    # então nenhuma fonte pode estar dentro delas (nem contê-las).
    for cfg_fonte in cfg["fontes"]:
        pasta = Path(str(cfg_fonte.get("pasta", "")))
        pasta = pasta.resolve() if pasta.is_absolute() else (config.parent / pasta).resolve()
        for conjunto in CONJUNTOS:
            destino = (saida / conjunto).resolve()
            if pasta == destino or destino in pasta.parents or pasta in destino.parents:
                raise ErroDeConfiguracao(
                    f"A fonte '{pasta}' se mistura com a pasta de saída '{destino}'. "
                    "Deixe cada export em uma subpasta própria, por exemplo dataset/brutos/<nome>."
                )

    existentes = [saida / c for c in CONJUNTOS if (saida / c).exists() and any((saida / c).rglob("*.*"))]
    if existentes and not limpar:
        raise ErroDeConfiguracao(
            f"Já existe um dataset preparado em '{saida}'. Rode de novo com --limpar para substituir."
        )
    for conjunto in CONJUNTOS:  # apaga só o que este script gera, nunca as fontes
        shutil.rmtree(saida / conjunto, ignore_errors=True)
    for antigo in ("data_publico.yaml", "train_publico.txt"):
        (saida / antigo).unlink(missing_ok=True)

    itens: list[Item] = []
    por_fonte = {}
    for cfg_fonte in cfg["fontes"]:
        lidos, estat = carregar_fonte(cfg_fonte, config.parent, classes, manter_vazias)
        nome = lidos[0].fonte if lidos else str(cfg_fonte.get("nome") or Path(cfg_fonte["pasta"]).name)
        if nome in por_fonte:
            raise ErroDeConfiguracao(f"Duas fontes com o mesmo nome '{nome}'. Use o campo 'nome' para diferenciar.")
        por_fonte[nome] = estat
        itens.extend(lidos)
    if not itens:
        raise ErroDeConfiguracao("Nenhuma imagem aproveitada. Confira as pastas e os mapas de classes.")

    dividir(itens, cfg["fracoes"], cfg["semente"])
    itens, removidas = enxugar_avaliacao(itens)

    saida.mkdir(parents=True, exist_ok=True)
    gravar(itens, saida)
    escrever_yaml(saida / "data.yaml", classes, "train/images",
                  "Inclui+ Vision: treino com todas as fontes (gerado por preparar_dataset.py)")
    escrever_manifesto(itens, classes, saida / "manifesto.csv")

    treino = [i for i in itens if i.conjunto == "train"]
    publicos = [i for i in treino if i.origem == "publico"]
    tem_variante = bool(publicos) and len(publicos) < len(treino)
    if tem_variante:
        lista = "".join(f"./train/images/{i.destino}\n" for i in sorted(publicos, key=lambda i: i.destino))
        (saida / "train_publico.txt").write_text(lista, encoding="utf-8")
        escrever_yaml(saida / "data_publico.yaml", classes, "train_publico.txt",
                      "Inclui+ Vision: treino só com fontes públicas; validação e teste iguais ao data.yaml")

    cont = contar(itens, classes)
    (saida / "resumo.md").write_text(resumo_markdown(cont, classes), encoding="utf-8")
    return {
        "classes": classes, "contagem": cont, "fontes": por_fonte, "itens": itens,
        "removidas_avaliacao": removidas, "avisos": avisos(cont, classes),
        "treino_publico": len(publicos) if tem_variante else None,
    }


def imprimir(resultado: dict, saida: Path) -> None:
    print("Fontes lidas:")
    for nome, estat in resultado["fontes"].items():
        print(f"  {nome}: {estat.imagens_lidas} imagens")
        if estat.classes_sem_mapa:
            print(f"    classes sem destino (caixas descartadas): {', '.join(estat.classes_sem_mapa)}")
        if estat.caixas_descartadas:
            detalhe = ", ".join(f"{c}: {n}" for c, n in sorted(estat.caixas_descartadas.items()))
            print(f"    caixas descartadas por classe de origem: {detalhe}")
        if estat.imagens_descartadas:
            print(f"    {estat.imagens_descartadas} imagens ficaram sem nenhuma caixa e foram deixadas de fora")
        if estat.linhas_invalidas:
            print(f"    {estat.linhas_invalidas} linhas de rótulo inválidas ignoradas")
    if resultado["removidas_avaliacao"]:
        print(f"Cópias aumentadas removidas da validação e do teste: {resultado['removidas_avaliacao']}")
    print("\n" + resumo_markdown(resultado["contagem"], resultado["classes"]))
    if resultado["treino_publico"] is not None:
        print(f"Variante só com fontes públicas: {resultado['treino_publico']} imagens de treino "
              f"(data_publico.yaml).")
    for aviso in resultado["avisos"]:
        print(f"AVISO: {aviso}")
    print(f"\nDataset pronto em {saida}. Treine com data={saida / 'data.yaml'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Une, unifica e divide os datasets do Inclui+ Vision.")
    p.add_argument("--config", default="dataset/fontes.yaml", type=Path)
    p.add_argument("--saida", default="dataset", type=Path)
    p.add_argument("--limpar", action="store_true", help="substitui um dataset já preparado")
    p.add_argument("--manter-vazias", action="store_true",
                   help="mantém como negativas as imagens que perderam todas as caixas no mapa de classes")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):  # acentos no terminal do Windows
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        resultado = preparar(args.config.resolve(), args.saida.resolve(), args.limpar, args.manter_vazias)
    except ErroDeConfiguracao as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        return 1
    imprimir(resultado, args.saida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
