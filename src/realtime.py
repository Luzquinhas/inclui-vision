"""Passo 5: detecção em tempo real com a webcam ou a câmera do celular.

Uso (a partir da raiz do repositório):
    python src/realtime.py                              # webcam padrão
    python src/realtime.py --fonte 1                    # outra câmera (DroidCam e Iriun costumam ser 1 ou 2)
    python src/realtime.py --fonte http://192.168.0.10:4747/video
    python src/realtime.py --fonte gravacao.mp4 --gravar demo.mp4

A tecla q encerra.
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2

from categorias import barreiras, categoria, sem_acento, ORDEM_CATEGORIAS

VERDE = (0, 255, 0)
VERMELHO = (0, 0, 255)
BRANCO = (255, 255, 255)
FONTE = cv2.FONT_HERSHEY_SIMPLEX


class Memoria:
    """Mantém uma classe ativa por alguns instantes depois da última detecção.

    Sem isso o painel pisca a cada quadro em que o modelo perde o objeto, e o
    alerta de escada dispara só porque a rampa saiu do enquadramento por um instante.
    Com janela 0 o comportamento é quadro a quadro, como no script da proposta.
    """

    def __init__(self, janela: float):
        self.janela = max(0.0, janela)
        self._visto: dict[str, float] = {}

    def atualizar(self, classes, agora: float) -> set[str]:
        for classe in classes:
            self._visto[classe] = agora
        self._visto = {c: t for c, t in self._visto.items() if agora - t <= self.janela}
        return set(self._visto)


def linhas_do_painel(ativos) -> list[str]:
    """Texto do resumo, uma linha por classe, na ordem das categorias do Inclui+."""
    ordenadas = sorted(ativos, key=lambda c: (ORDEM_CATEGORIAS.index(categoria(c)), c))
    return [sem_acento(f"{categoria(c)}: {c}") for c in ordenadas]


def desenhar_painel(frame, ativos, alertas, fps=None):
    """Escreve o resumo das detecções e os alertas no canto do quadro (altera o próprio frame)."""
    linhas = [(texto, VERDE) for texto in linhas_do_painel(ativos)]
    linhas += [(sem_acento(f"ATENCAO: {a}"), VERMELHO) for a in alertas]
    if linhas:
        largura = max(cv2.getTextSize(t, FONTE, 0.7, 2)[0][0] for t, _ in linhas)
        fundo = frame.copy()
        cv2.rectangle(fundo, (0, 0), (largura + 20, 30 * len(linhas) + 12), (0, 0, 0), -1)
        cv2.addWeighted(fundo, 0.55, frame, 0.45, 0, frame)
    y = 30
    for texto, cor in linhas:
        cv2.putText(frame, texto, (10, y), FONTE, 0.7, cor, 2, cv2.LINE_AA)
        y += 30
    if fps is not None:
        altura, largura_frame = frame.shape[:2]
        cv2.putText(frame, f"{fps:.0f} FPS", (largura_frame - 110, altura - 15),
                    FONTE, 0.6, BRANCO, 2, cv2.LINE_AA)
    return frame


def ler_argumentos(argv=None):
    p = argparse.ArgumentParser(description="Inclui+ Vision: detecção em tempo real.")
    p.add_argument("--pesos", default="weights/best.pt", help="modelo treinado")
    p.add_argument("--fonte", default="0",
                   help="índice da câmera, URL do celular ou arquivo de vídeo")
    p.add_argument("--conf", type=float, default=0.5, help="confiança mínima")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--janela", type=float, default=1.0,
                   help="segundos que uma detecção continua valendo (0 = quadro a quadro)")
    p.add_argument("--gravar", default=None, help="salva o vídeo anotado neste arquivo .mp4")
    p.add_argument("--sem-janela", action="store_true",
                   help="não abre a janela (para processar um vídeo no Colab)")
    return p.parse_args(argv)


def main(argv=None):
    args = ler_argumentos(argv)
    from ultralytics import YOLO  # importado aqui para o resto do módulo não depender dele

    modelo = YOLO(args.pesos)

    fonte = int(args.fonte) if args.fonte.isdigit() else args.fonte
    ao_vivo = isinstance(fonte, int) or fonte.lower().startswith(("http", "rtsp"))
    cap = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        sys.exit(f"Não consegui abrir a fonte de vídeo '{args.fonte}'. "
                 "Confira o índice da câmera ou o caminho do arquivo.")

    fps_fonte = cap.get(cv2.CAP_PROP_FPS) or 30.0
    memoria = Memoria(args.janela)
    gravador = None
    fps = 0.0
    quadros = 0
    anterior = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            quadros += 1

            resultado = modelo(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            anotado = resultado.plot()

            agora = time.perf_counter()
            instantaneo = 1.0 / max(agora - anterior, 1e-6)
            fps = instantaneo if fps == 0 else 0.9 * fps + 0.1 * instantaneo
            anterior = agora

            # Em arquivo de vídeo o relógio é o do próprio vídeo, não o do computador.
            relogio = agora if ao_vivo else quadros / fps_fonte
            detectados = {resultado.names[int(c)] for c in resultado.boxes.cls.tolist()}
            ativos = memoria.atualizar(detectados, relogio)
            desenhar_painel(anotado, ativos, barreiras(ativos), fps)

            if args.gravar:
                # Ao vivo, espera alguns quadros para medir o FPS real antes de abrir o arquivo.
                if gravador is None and (not ao_vivo or quadros >= 15):
                    altura, largura = anotado.shape[:2]
                    taxa = max(fps, 1.0) if ao_vivo else fps_fonte
                    gravador = cv2.VideoWriter(args.gravar, cv2.VideoWriter_fourcc(*"mp4v"),
                                               taxa, (largura, altura))
                if gravador is not None:
                    gravador.write(anotado)

            if not args.sem_janela:
                cv2.imshow("Inclui+ Vision", anotado)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if gravador is not None:
            gravador.release()
        if not args.sem_janela:
            cv2.destroyAllWindows()

    if args.gravar:
        print(f"Vídeo salvo em {args.gravar} ({quadros} quadros).")


if __name__ == "__main__":
    main()
