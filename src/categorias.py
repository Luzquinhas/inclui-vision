"""Classes do modelo, categorias do Inclui+ e regras de barreira.

Fica separado dos scripts para que realtime.py e relatorio.py usem as mesmas
definições, e para poder ser testado sem câmera nem modelo.
"""
from __future__ import annotations

import unicodedata

# A ordem é o ID de cada classe no data.yaml.
CLASSES = [
    "rampa",
    "escada",
    "corrimao",
    "elevador",
    "barra_apoio",
    "vaga_pcd",
    "piso_tatil",
    "simbolo_acessibilidade",
]

CATEGORIAS = {
    "rampa": "Mobilidade",
    "escada": "Mobilidade",
    "corrimao": "Mobilidade",
    "elevador": "Mobilidade",
    "barra_apoio": "Mobilidade",
    "vaga_pcd": "Mobilidade",
    "piso_tatil": "Visão",
    "simbolo_acessibilidade": "Geral",
}

DESCRICOES = {
    "rampa": "Rampa de acesso",
    "escada": "Escada ou degrau",
    "corrimao": "Corrimão",
    "elevador": "Elevador",
    "barra_apoio": "Barra de apoio",
    "vaga_pcd": "Vaga PcD",
    "piso_tatil": "Piso tátil",
    "simbolo_acessibilidade": "Símbolo de acessibilidade",
}

# Ordem em que as categorias aparecem no painel e no relatório.
ORDEM_CATEGORIAS = ["Mobilidade", "Visão", "Geral", "Outros"]


def categoria(classe: str) -> str:
    """Categoria do Inclui+ de uma classe; 'Outros' se o modelo tiver uma classe nova."""
    return CATEGORIAS.get(classe, "Outros")


def sem_acento(texto: str) -> str:
    """Remove acentos. O cv2.putText só desenha ASCII e trocaria 'Visão' por 'Vis??o'."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def barreiras(classes) -> list[str]:
    """Barreiras deduzidas do conjunto de classes detectadas.

    Regra da proposta: há escada e nenhuma rampa foi vista.
    Vale só para o que apareceu na imagem (ou na janela de tempo do vídeo):
    uma rampa fora do enquadramento não é considerada.
    """
    presentes = set(classes)
    achadas = []
    if "escada" in presentes and "rampa" not in presentes:
        achadas.append("Escada sem rampa alternativa")
    return achadas


def agrupar_por_categoria(contagem) -> dict[str, dict[str, int]]:
    """Agrupa {classe: quantidade} pelas categorias do Inclui+.

    >>> agrupar_por_categoria({"escada": 2, "piso_tatil": 1})
    {'Mobilidade': {'escada': 2}, 'Visão': {'piso_tatil': 1}}
    """
    grupos: dict[str, dict[str, int]] = {}
    for nome_categoria in ORDEM_CATEGORIAS:
        itens = {
            classe: int(n)
            for classe, n in sorted(dict(contagem).items())
            if n > 0 and categoria(classe) == nome_categoria
        }
        if itens:
            grupos[nome_categoria] = itens
    return grupos
