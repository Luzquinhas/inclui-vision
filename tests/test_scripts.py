"""Testes da lógica dos scripts que não depende do modelo treinado.

    python -m unittest discover -s tests -v
"""
import csv
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import avaliar  # noqa: E402
import relatorio  # noqa: E402
from categorias import (CATEGORIAS, CLASSES, DESCRICOES, agrupar_por_categoria,  # noqa: E402
                        barreiras, categoria, sem_acento)

try:
    import cv2  # noqa: F401
    import numpy as np
    TEM_OPENCV = True
except ImportError:  # pragma: no cover
    TEM_OPENCV = False


class Categorias(unittest.TestCase):
    def test_toda_classe_tem_categoria_e_descricao(self):
        self.assertEqual(set(CLASSES), set(CATEGORIAS))
        self.assertEqual(set(CLASSES), set(DESCRICOES))
        self.assertEqual(categoria("piso_tatil"), "Visão")
        self.assertEqual(categoria("classe_nova"), "Outros")

    def test_barreira_escada_sem_rampa(self):
        self.assertEqual(barreiras({"escada", "corrimao"}), ["Escada sem rampa alternativa"])
        self.assertEqual(barreiras({"escada", "rampa"}), [])
        self.assertEqual(barreiras({"piso_tatil"}), [])
        self.assertEqual(barreiras([]), [])

    def test_sem_acento(self):
        self.assertEqual(sem_acento("Visão: piso tátil — ATENÇÃO"), "Visao: piso tatil  ATENCAO")
        self.assertTrue(sem_acento("Mobilidade: corrimão").isascii())

    def test_agrupar(self):
        grupos = agrupar_por_categoria(Counter(piso_tatil=1, escada=2, simbolo_acessibilidade=1, rampa=0))
        self.assertEqual(list(grupos), ["Mobilidade", "Visão", "Geral"])
        self.assertEqual(grupos["Mobilidade"], {"escada": 2})


class Relatorio(unittest.TestCase):
    def test_relatorio_com_barreira(self):
        rel = relatorio.montar_relatorio(Counter(escada=2, corrimao=1, piso_tatil=1))
        self.assertEqual(rel["total_deteccoes"], 4)
        self.assertEqual(rel["barreiras"], ["Escada sem rampa alternativa"])
        texto = relatorio.formatar(rel)
        self.assertIn("[Mobilidade] escada: 2", texto)
        self.assertIn("[Visão] piso_tatil: 1", texto)
        self.assertIn("ATENÇÃO, barreira: Escada sem rampa alternativa", texto)

    def test_relatorio_vazio(self):
        texto = relatorio.formatar(relatorio.montar_relatorio(Counter()))
        self.assertIn("Nenhum recurso", texto)
        self.assertNotIn("barreira", texto)


class MetricasFalsas:
    """Imita o objeto devolvido por model.val() do Ultralytics."""
    names = {0: "rampa", 1: "escada", 2: "corrimao", 6: "piso_tatil"}
    speed = {"preprocess": 1.0, "inference": 8.0, "loss": 0.0, "postprocess": 1.0}
    save_dir = "."

    def __init__(self):
        valores = {1: (0.9, 0.8, 0.85, 0.6), 6: (0.7, 0.5, 0.65, 0.4)}
        self.box = SimpleNamespace(
            ap_class_index=[1, 6],
            class_result=lambda i: valores[[1, 6][i]],
            map50=0.75, map=0.5, mp=0.8, mr=0.65,
        )


class Avaliar(unittest.TestCase):
    def test_tabelas(self):
        m = MetricasFalsas()
        por_classe = avaliar.tabela_por_classe(m)
        self.assertEqual([l["classe"] for l in por_classe], ["escada", "piso_tatil"])
        self.assertEqual(por_classe[1]["recall"], 0.5)
        geral = avaliar.resumo_geral(m)
        self.assertAlmostEqual(geral["ms_por_imagem"], 10.0)
        self.assertAlmostEqual(geral["fps"], 100.0)
        texto = avaliar.para_markdown(geral, por_classe, "yolo11n", "teste")
        self.assertIn("| 0.750 | 0.500 | 0.800 | 0.650 | 10.0 ms | 100 |", texto)
        self.assertIn("| `piso_tatil` | 0.700 | 0.500 | 0.650 | 0.400 |", texto)

    def test_comparacao_substitui_mesmo_rotulo(self):
        with tempfile.TemporaryDirectory() as tmp:
            arquivo = Path(tmp) / "results" / "comparacao.csv"
            avaliar.atualizar_comparacao(arquivo, {"rotulo": "yolo11n", "mAP50": "0.70"})
            avaliar.atualizar_comparacao(arquivo, {"rotulo": "yolo11s", "mAP50": "0.80"})
            avaliar.atualizar_comparacao(arquivo, {"rotulo": "yolo11n", "mAP50": "0.72"})
            with arquivo.open(encoding="utf-8") as f:
                linhas = {l["rotulo"]: l["mAP50"] for l in csv.DictReader(f)}
            self.assertEqual(linhas, {"yolo11s": "0.80", "yolo11n": "0.72"})


@unittest.skipUnless(TEM_OPENCV, "precisa de opencv-python e numpy")
class TempoReal(unittest.TestCase):
    def test_memoria(self):
        import realtime
        memoria = realtime.Memoria(janela=1.0)
        self.assertEqual(memoria.atualizar({"escada"}, 0.0), {"escada"})
        self.assertEqual(memoria.atualizar({"rampa"}, 0.5), {"escada", "rampa"})
        self.assertEqual(memoria.atualizar(set(), 1.2), {"rampa"})   # a escada expirou
        self.assertEqual(memoria.atualizar(set(), 2.0), set())
        quadro_a_quadro = realtime.Memoria(janela=0)
        quadro_a_quadro.atualizar({"escada"}, 0.0)
        self.assertEqual(quadro_a_quadro.atualizar({"rampa"}, 0.04), {"rampa"})

    def test_painel(self):
        import realtime
        linhas = realtime.linhas_do_painel({"piso_tatil", "escada", "simbolo_acessibilidade"})
        self.assertEqual(linhas, ["Mobilidade: escada", "Visao: piso_tatil", "Geral: simbolo_acessibilidade"])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        realtime.desenhar_painel(frame, {"escada"}, barreiras({"escada"}), fps=31.4)
        self.assertGreater(int(frame[:, :, 1].sum()), 0)   # texto verde do resumo
        self.assertGreater(int(frame[:, :, 2].sum()), 0)   # alerta em vermelho

    def test_janela_cabe_na_tela(self):
        import realtime
        vertical = np.zeros((1920, 1080, 3), dtype=np.uint8)
        self.assertEqual(realtime.ajustar_para_tela(vertical, 720, 1280).shape[:2], (720, 405))
        pequeno = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertIs(realtime.ajustar_para_tela(pequeno, 720, 1280), pequeno)   # nunca amplia
        self.assertEqual(realtime.girar(vertical, 90).shape[:2], (1080, 1920))
        self.assertIs(realtime.girar(vertical, 0), vertical)
        # O painel cresce com o quadro: em vídeo de celular o texto ocupa mais pixels.
        a, b = pequeno.copy(), vertical.copy()
        realtime.desenhar_painel(a, {"escada"}, [], fps=30)
        realtime.desenhar_painel(b, {"escada"}, [], fps=30)
        self.assertGreater(int((b[:, :, 1] > 0).sum()), int((a[:, :, 1] > 0).sum()))

    def test_grade_de_amostras(self):
        import ver_amostras
        imagens = []
        for lado in ((300, 400), (600, 200), (128, 128)):
            img = np.full((*lado, 3), 200, dtype=np.uint8)
            imagens.append(ver_amostras.desenhar_rotulos(img, [(1, 0.5, 0.5, 0.4, 0.4)], {1: "escada"}))
        grade = ver_amostras.montar_grade(imagens, colunas=2, lado=240)
        self.assertEqual(grade.shape, (480, 480, 3))


if __name__ == "__main__":
    unittest.main()
