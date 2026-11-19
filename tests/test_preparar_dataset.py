"""Testes da preparação do dataset. Rodam sem GPU e sem o Ultralytics:

    python -m unittest discover -s tests -v
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import preparar_dataset as prep  # noqa: E402
from util_dataset import imagens_do_conjunto, ler_rotulos  # noqa: E402

CLASSES = ["rampa", "escada", "corrimao", "elevador", "barra_apoio",
           "vaga_pcd", "piso_tatil", "simbolo_acessibilidade"]


def imagem_falsa(caminho: Path, rotulo=None):
    """Cria um arquivo com cara de imagem e, se pedido, o rótulo YOLO dele."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_bytes(b"\xff\xd8\xff\xd9")
    if rotulo is not None:
        pasta = Path(str(caminho.parent).replace("images", "labels"))
        pasta.mkdir(parents=True, exist_ok=True)
        (pasta / (caminho.name.rsplit(".", 1)[0] + ".txt")).write_text(rotulo, encoding="utf-8")


def montar_cenario(raiz: Path, classes=None, fontes_extra=None) -> Path:
    brutos = raiz / "dataset" / "brutos"

    # Fonte pública no estilo do Roboflow: 20 fotos, cada uma com 3 cópias aumentadas.
    publica = brutos / "publica"
    (publica / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (publica / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\n"
        "names: ['downstairs', 'handrail', 'person', 'upstairs']\n", encoding="utf-8")
    for i in range(20):
        for copia in ("aaa111", "bbb222", "ccc333"):
            rotulo = "0 0.5 0.5 0.4 0.4\n1 0.2 0.5 0.1 0.6\n2 0.8 0.8 0.1 0.1\n"
            if i % 2:
                rotulo = "3 0.5 0.5 0.4 0.4\n"
            imagem_falsa(publica / "train" / "images" / f"foto{i:02d}_jpg.rf.{copia}.jpg", rotulo)
    # Só pessoas: deve ser descartada. Sem rótulo: é um exemplo negativo.
    imagem_falsa(publica / "train" / "images" / "so_pessoa_jpg.rf.ddd444.jpg", "2 0.5 0.5 0.2 0.2\n")
    imagem_falsa(publica / "train" / "images" / "vazia_jpg.rf.eee555.jpg", "")

    # Fotos próprias, já com os nomes finais; uma delas anotada como polígono.
    propria = brutos / "propria"
    propria.mkdir(parents=True, exist_ok=True)
    (propria / "data.yaml").write_text(
        yaml.safe_dump({"names": {0: "rampa", 1: "piso_tatil", 2: "elevador"}}), encoding="utf-8")
    for i in range(10):
        imagem_falsa(propria / "images" / f"IMG_{i:04d}.jpg", f"{i % 3} 0.5 0.5 0.3 0.3\n")
    imagem_falsa(propria / "images" / "poligono.png", "1 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8\n")

    negativas = brutos / "negativas"
    for i in range(5):
        imagem_falsa(negativas / f"rua_{i}.jpg")

    config = {
        "classes": classes or CLASSES,
        "fontes": [
            {"nome": "publica", "pasta": "brutos/publica", "origem": "publico",
             "mapa": {"upstairs": "escada", "downstairs": "escada", "handrail": "corrimao"}},
            {"nome": "propria", "pasta": "brutos/propria", "origem": "proprio"},
            {"nome": "negativas", "pasta": "brutos/negativas", "origem": "proprio", "negativos": True},
        ] + (fontes_extra or []),
    }
    arquivo = raiz / "dataset" / "fontes.yaml"
    arquivo.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return arquivo


def ler_manifesto(saida: Path) -> list[dict]:
    with (saida / "manifesto.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


class PrepararDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self.tmp.name)
        self.config = montar_cenario(self.raiz)
        self.saida = self.raiz / "dataset"

    def tearDown(self):
        self.tmp.cleanup()

    def test_divide_e_remapeia(self):
        r = prep.preparar(self.config, self.saida)
        cont = r["contagem"]
        total = sum(cont["total"].values())
        self.assertGreater(cont["total"]["train"], cont["total"]["valid"])
        self.assertGreater(cont["total"]["valid"], 0)
        self.assertGreater(cont["total"]["test"], 0)

        # Só existem IDs das classes finais, e 'person' sumiu.
        ids = set()
        for conjunto in ("train", "valid", "test"):
            for txt in (self.saida / conjunto / "labels").glob("*.txt"):
                ids |= {int(l.split()[0]) for l in txt.read_text().splitlines() if l.strip()}
        self.assertEqual(ids, {0, 1, 2, 3, 6})  # rampa, escada, corrimao, elevador, piso_tatil

        # A imagem que só tinha pessoa foi descartada; as negativas ficaram.
        nomes = [l["arquivo"] for l in ler_manifesto(self.saida)]
        self.assertFalse(any("so_pessoa" in n for n in nomes))
        self.assertEqual(sum(cont["negativas"].values()), 6)  # 5 da pasta + a 'vazia' da pública
        self.assertEqual(len(nomes), total)
        self.assertEqual(r["fontes"]["publica"].imagens_descartadas, 1)
        self.assertEqual(r["fontes"]["publica"].classes_sem_mapa, ["person"])

    def test_copias_aumentadas_nao_vazam(self):
        prep.preparar(self.config, self.saida)
        onde = {}
        por_grupo_avaliacao = {}
        for linha in ler_manifesto(self.saida):
            if linha["fonte"] != "publica":
                continue
            grupo = linha["arquivo"].split(".rf.")[0]
            onde.setdefault(grupo, set()).add(linha["conjunto"])
            if linha["conjunto"] != "train":
                por_grupo_avaliacao[grupo] = por_grupo_avaliacao.get(grupo, 0) + 1
        self.assertTrue(all(len(c) == 1 for c in onde.values()), "uma foto caiu em dois conjuntos")
        self.assertTrue(all(n == 1 for n in por_grupo_avaliacao.values()),
                        "validação e teste devem ter uma cópia por foto")

    def test_poligono_vira_caixa(self):
        prep.preparar(self.config, self.saida)
        achado = [p for c in ("train", "valid", "test")
                  for p in (self.saida / c / "images").glob("propria__poligono.png")]
        self.assertEqual(len(achado), 1)
        (classe, xc, yc, w, h), = ler_rotulos(achado[0])
        self.assertEqual(classe, 6)  # piso_tatil
        for obtido, esperado in zip((xc, yc, w, h), (0.3, 0.5, 0.4, 0.6)):
            self.assertAlmostEqual(obtido, esperado, places=5)

    def test_yaml_e_variante_publica(self):
        r = prep.preparar(self.config, self.saida)
        data = yaml.safe_load((self.saida / "data.yaml").read_text(encoding="utf-8"))
        self.assertNotIn("path", data)
        self.assertEqual(data["names"], dict(enumerate(CLASSES)))

        treino = imagens_do_conjunto(self.saida / "data.yaml", "train")
        publico = imagens_do_conjunto(self.saida / "data_publico.yaml", "train")
        self.assertEqual(len(publico), r["treino_publico"])
        self.assertLess(len(publico), len(treino))
        self.assertTrue(all(p.exists() and p.name.startswith("publica__") for p in publico))
        # Validação e teste são os mesmos nas duas variantes.
        for conjunto in ("val", "test"):
            self.assertEqual(imagens_do_conjunto(self.saida / "data.yaml", conjunto),
                             imagens_do_conjunto(self.saida / "data_publico.yaml", conjunto))

    def test_reproduzivel_e_exige_limpar(self):
        prep.preparar(self.config, self.saida)
        primeiro = ler_manifesto(self.saida)
        with self.assertRaises(prep.ErroDeConfiguracao):
            prep.preparar(self.config, self.saida)
        prep.preparar(self.config, self.saida, limpar=True)
        self.assertEqual(primeiro, ler_manifesto(self.saida))
        # As fontes continuam intactas depois de limpar.
        self.assertTrue((self.saida / "brutos" / "publica" / "data.yaml").exists())

    def test_escopo_reduzido(self):
        config = montar_cenario(self.raiz, classes=["rampa", "escada", "corrimao", "piso_tatil"])
        r = prep.preparar(config, self.saida)
        self.assertEqual(r["classes"], ["rampa", "escada", "corrimao", "piso_tatil"])
        self.assertGreater(r["fontes"]["propria"].caixas_descartadas["elevador"], 0)
        ids = set()
        for conjunto in ("train", "valid", "test"):
            for txt in (self.saida / conjunto / "labels").glob("*.txt"):
                ids |= {int(l.split()[0]) for l in txt.read_text().splitlines() if l.strip()}
        self.assertEqual(ids, {0, 1, 2, 3})

    def test_erros_de_configuracao(self):
        nomes = {0: "stairs", 1: "ramp"}
        with self.assertRaisesRegex(prep.ErroDeConfiguracao, "não existem nela"):
            prep.montar_remap(nomes, {"escadaria": "escada"}, CLASSES, "x")
        with self.assertRaisesRegex(prep.ErroDeConfiguracao, "digitado certo"):
            prep.montar_remap(nomes, {"stairs": "escadas"}, CLASSES, "x")
        remap, sem_mapa = prep.montar_remap(nomes, {"Stairs": "escada"}, CLASSES, "x")
        self.assertEqual(remap, {0: 1, 1: None})
        self.assertEqual(sem_mapa, ["ramp"])

    def test_fonte_dentro_da_saida(self):
        config = montar_cenario(self.raiz, fontes_extra=[
            {"nome": "errada", "pasta": ".", "origem": "publico", "negativos": True}])
        with self.assertRaisesRegex(prep.ErroDeConfiguracao, "se mistura"):
            prep.preparar(config, self.saida)
        self.assertTrue((self.saida / "brutos" / "publica" / "data.yaml").exists())

    def test_linhas_de_rotulo(self):
        remap = {0: 1, 1: None}
        caixas, descartadas, invalidas = prep.ler_caixas(
            "0 0.5 0.5 0.2 0.2\n"      # válida
            "1 0.5 0.5 0.2 0.2\n"      # classe descartada
            "0 0.95 0.5 0.2 0.2\n"     # sai da imagem: é cortada na borda
            "0 0.5 0.5 0 0.2\n"        # largura zero
            "7 0.5 0.5 0.2 0.2\n"      # id que não existe na fonte
            "lixo\n", remap)
        self.assertEqual(len(caixas), 2)
        self.assertEqual(descartadas[1], 1)
        self.assertEqual(invalidas, 3)
        _, xc, _, w, _ = caixas[1]
        self.assertAlmostEqual(xc + w / 2, 1.0)


if __name__ == "__main__":
    unittest.main()
