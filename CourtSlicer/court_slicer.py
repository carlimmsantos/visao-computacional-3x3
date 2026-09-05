"""
CourtSlicer
===========

Ferramenta para revisar vídeos de partidas e marcar intervalos de tempo morto
que depois serão removidos pelo script `Corte_videos/remover_trechos.py`.

REQUISITOS
----------
- Python
- mpv
- ffmpeg / ffprobe

No Ubuntu/Pop!_OS:

    sudo apt install mpv ffmpeg

A instalação do ambiente virtual do projeto pode ser feita com:

    cd CourtSlicer
    bash install.sh


COMO EXECUTAR
-------------
Entre na pasta CourtSlicer:

    cd CourtSlicer

Depois execute:

    .venv/bin/court-slicer \
      "../Corte_videos/partidas_cortadas/<PASTA_DO_TORNEIO>/<VIDEO>.mp4"

Exemplo:

    .venv/bin/court-slicer \
      "../Corte_videos/partidas_cortadas/Torneio 3x3 - LPB/Jogo11_IMPERIAL_vs_MNC.mp4"


CONTROLES
---------
- A       : voltar 5 segundos
- D       : avançar 5 segundos
- Espaço  : play / pause
- F       : marcar início ou fim de um intervalo
- +       : aumentar velocidade
- -       : diminuir velocidade
- Ctrl+Z  : desfazer última marcação
- Q / Esc : sair

Também é possível usar os botões da interface.

A barra de progresso é clicável:
- clique em qualquer ponto para ir diretamente até aquele momento do vídeo;
- também é possível arrastar o marcador normalmente.

A velocidade pode ser alterada pelo seletor da interface, até 4x.


COMO MARCAR OS CORTES
---------------------
As marcações devem ser feitas sempre em pares:

    F -> início do trecho que deve ser removido
    F -> fim do trecho que deve ser removido

Exemplo:

    00:04:17 -> início de timeout
    00:04:53 -> fim do timeout

A interface mostra automaticamente:

    1. 00:04:17 -> 00:04:53

Se apenas o início tiver sido marcado, será mostrado:

    1. 00:04:17 -> aguardando fim

O botão também alterna automaticamente entre:

    "Marcar início"
    "Marcar fim"


O QUE DEVE SER MARCADO
----------------------
Marcar apenas pausas reais / tempo morto, por exemplo:

- timeout;
- pausa técnica;
- atendimento a jogador lesionado;
- jogador caído quando o jogo realmente é interrompido;
- discussão prolongada com arbitragem;
- interrupção longa da partida;
- bola que sai e demora significativamente para voltar.

Não marcar automaticamente:

- toda falta;
- toda bola fora;
- reposição rápida;
- pequenos momentos de organização da posse;
- situações em que o jogo continua normalmente.


RESULTADO
---------
Ao fechar o CourtSlicer, os intervalos marcados são exibidos no terminal
também no formato utilizado pelo CSV de cortes:

    video,inicio,fim,motivo

Exemplo:

    Jogo11_IMPERIAL_vs_MNC.mp4,00:04:17,00:04:53,tempo_morto

Copie os intervalos necessários para:

    Corte_videos/templates/cortes_internos.csv

ou para o CSV específico da sua dupla.


IMPORTANTE
----------
O CourtSlicer serve para localizar e registrar os timestamps.

A remoção final dos trechos deve ser feita com:

    cd Corte_videos

    python remover_trechos.py \
      --csv templates/cortes_internos.csv \
      --videos_dir partidas_cortadas \
      --output_dir partidas_limpas

Os vídeos em `partidas_cortadas/` são usados como fonte original.

Os vídeos resultantes são gerados em:

    partidas_limpas/

Não commitar arquivos .mp4 no repositório.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import tkinter as tk
from tkinter import ttk


PROJECT_ROOT = Path(__file__).resolve().parent
VIDEOS_ENTRADA = PROJECT_ROOT / "videos_entrada"

MIN_RATE = 0.25
MAX_RATE = 4.0
RATE_STEP = 1.25


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Assista ao vídeo e marque intervalos para remoção."
    )

    parser.add_argument(
        "video",
        help=(
            "Arquivo de vídeo. Se informar apenas o nome, "
            "procura em videos_entrada/."
        ),
    )

    return parser.parse_args()


def resolve_video_path(video_arg: str) -> Path:
    given = Path(video_arg)

    if given.exists():
        return given.resolve()

    candidate = VIDEOS_ENTRADA / given.name

    if candidate.exists():
        return candidate.resolve()

    return given.resolve()


# =============================================================================
# Helpers
# =============================================================================

def fmt_seconds(seconds: float | int | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"

    total = int(seconds)

    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def fmt_csv_time(seconds: float) -> str:
    total = int(seconds)

    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# =============================================================================
# MPV IPC
# =============================================================================

class MPVController:

    def __init__(self, video_path: Path):
        self.video_path = video_path

        self.sock_path = (
            Path(tempfile.gettempdir())
            / f"courtslicer_mpv_{os.getpid()}.sock"
        )

        self.proc = None
        self.request_id = 0


    def start(self, wid: int):

        if self.sock_path.exists():
            self.sock_path.unlink()

        cmd = [
            "mpv",

            # vídeo dentro do Canvas do Tk
            f"--wid={wid}",

            # NO SEU SISTEMA isto corrige a imagem verde
            "--deinterlace=yes",

            # evita conflito entre teclas do mpv e CourtSlicer
            "--input-default-bindings=no",
            "--input-vo-keyboard=no",

            "--force-window=yes",
            "--keep-open=yes",

            # mpv controlado via socket
            f"--input-ipc-server={self.sock_path}",

            # remove controles/OSD próprios
            "--osd-level=0",

            str(self.video_path),
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        deadline = time.time() + 5.0

        while time.time() < deadline:

            if self.sock_path.exists():
                return

            time.sleep(0.05)

        raise RuntimeError(
            "O mpv não criou o socket IPC. "
            "Confirme se 'mpv' está instalado."
        )


    def stop(self):

        try:
            self.command(["quit"])
        except Exception:
            pass

        if self.proc and self.proc.poll() is None:

            try:
                self.proc.terminate()
            except Exception:
                pass

        if self.sock_path.exists():

            try:
                self.sock_path.unlink()
            except Exception:
                pass


    def _send(self, payload: dict):

        message = json.dumps(payload).encode("utf-8") + b"\n"

        with socket.socket(
            socket.AF_UNIX,
            socket.SOCK_STREAM,
        ) as sock:

            sock.settimeout(1.0)
            sock.connect(str(self.sock_path))
            sock.sendall(message)

            data = b""

            while b"\n" not in data:

                chunk = sock.recv(4096)

                if not chunk:
                    break

                data += chunk

        if not data:
            return {}

        line = data.splitlines()[0]

        try:
            return json.loads(line.decode("utf-8"))

        except Exception:
            return {}


    def command(self, command: list):

        self.request_id += 1

        return self._send(
            {
                "command": command,
                "request_id": self.request_id,
            }
        )


    def get_property(self, name):

        response = self.command(
            ["get_property", name]
        )

        return response.get("data")


    def set_property(self, name, value):

        return self.command(
            ["set_property", name, value]
        )


    def current_time(self):

        value = self.get_property("time-pos")

        if value is None:
            return 0.0

        return float(value)


    def duration(self):

        value = self.get_property("duration")

        if value is None:
            return 0.0

        return float(value)


    def pause_toggle(self):

        self.command(
            ["cycle", "pause"]
        )


    def seek_relative(self, seconds):

        self.command(
            [
                "seek",
                seconds,
                "relative",
                "exact",
            ]
        )


    def seek_absolute(self, seconds):

        self.command(
            [
                "seek",
                seconds,
                "absolute",
                "exact",
            ]
        )


    def set_speed(self, speed):

        self.set_property(
            "speed",
            speed,
        )


# =============================================================================
# GUI
# =============================================================================

class CourtSlicerApp(tk.Tk):

    def __init__(self, video_path: Path):

        super().__init__()

        self.video_path = video_path

        self.mpv = MPVController(
            video_path
        )

        # marcações em segundos
        self.marks: list[float] = []

        self.rate = 1.0

        self._quitting = False
        self._slider_dragging = False

        # ---------------------------------------------------------------------
        # Window
        # ---------------------------------------------------------------------

        self.title(
            f"CourtSlicer — {video_path.name}"
        )

        self.geometry("1180x820")
        self.minsize(850, 650)

        self.configure(
            bg="#151515"
        )

        self.resizable(
            True,
            True,
        )

        # ---------------------------------------------------------------------
        # Style
        # ---------------------------------------------------------------------

        self._setup_style()
        self._build_ui()
        self._bind_keys()

        # Canvas precisa existir antes do --wid
        self.update_idletasks()

        wid = self.video_canvas.winfo_id()

        try:

            self.mpv.start(wid)

        except Exception as exc:

            print(
                f"[ERRO] Não foi possível iniciar mpv: {exc}"
            )

            self.destroy()
            return

        self.after(
            200,
            self._update_status,
        )


    # =========================================================================
    # Style
    # =========================================================================

    def _setup_style(self):

        style = ttk.Style(self)

        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            "Timeline.Horizontal.TScale",
            background="#202020",
            troughcolor="#404040",
        )

        style.configure(
            "Speed.TCombobox",
            padding=4,
        )


    def make_button(
        self,
        parent,
        text,
        command,
        width=None,
        prominent=False,
    ):

        if prominent:
            bg = "#3f6ea8"
            active = "#527fba"

        else:
            bg = "#303030"
            active = "#444444"

        return tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            bg=bg,
            fg="#f2f2f2",
            activebackground=active,
            activeforeground="white",
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=7,
            cursor="hand2",
            font=("Sans", 10),
        )


    # =========================================================================
    # Build UI
    # =========================================================================

    def _build_ui(self):

        # ---------------------------------------------------------------------
        # Video
        # ---------------------------------------------------------------------

        self.video_container = tk.Frame(
            self,
            bg="black",
        )

        self.video_container.pack(
            fill=tk.BOTH,
            expand=True,
            padx=8,
            pady=(8, 0),
        )

        self.video_canvas = tk.Canvas(
            self.video_container,
            bg="black",
            highlightthickness=0,
        )

        self.video_canvas.pack(
            fill=tk.BOTH,
            expand=True,
        )

        # ---------------------------------------------------------------------
        # Bottom panel
        # ---------------------------------------------------------------------

        panel = tk.Frame(
            self,
            bg="#202020",
        )

        panel.pack(
            fill=tk.X,
            padx=8,
            pady=8,
        )

        # ---------------------------------------------------------------------
        # Timeline
        # ---------------------------------------------------------------------

        timeline_row = tk.Frame(
            panel,
            bg="#202020",
        )

        timeline_row.pack(
            fill=tk.X,
            padx=10,
            pady=(10, 4),
        )

        self.current_label = tk.Label(
            timeline_row,
            text="00:00",
            bg="#202020",
            fg="#eeeeee",
            font=("Monospace", 10),
            width=8,
        )

        self.current_label.pack(
            side=tk.LEFT,
        )

        self.seek_var = tk.DoubleVar(
            value=0
        )

        self.seek_slider = ttk.Scale(
            timeline_row,
            from_=0,
            to=1000,
            orient=tk.HORIZONTAL,
            variable=self.seek_var,
            style="Timeline.Horizontal.TScale",
        )

        self.seek_slider.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=8,
        )

        self.total_label = tk.Label(
            timeline_row,
            text="00:00",
            bg="#202020",
            fg="#eeeeee",
            font=("Monospace", 10),
            width=8,
        )

        self.total_label.pack(
            side=tk.RIGHT,
        )

        # Clique direto na timeline
        self.seek_slider.bind(
            "<Button-1>",
            self._timeline_click,
        )

        # Arrastar normalmente
        self.seek_slider.bind(
            "<ButtonPress-1>",
            self._timeline_press,
            add="+",
        )

        self.seek_slider.bind(
            "<ButtonRelease-1>",
            self._timeline_release,
        )

        # ---------------------------------------------------------------------
        # Controls
        # ---------------------------------------------------------------------

        controls = tk.Frame(
            panel,
            bg="#202020",
        )

        controls.pack(
            fill=tk.X,
            padx=10,
            pady=5,
        )

        self.make_button(
            controls,
            "< 5s",
            self._rewind,
        ).pack(
            side=tk.LEFT,
            padx=(0, 5),
        )

        self.make_button(
            controls,
            "Play / Pause",
            self._pause_toggle,
        ).pack(
            side=tk.LEFT,
            padx=5,
        )

        self.make_button(
            controls,
            "5s >",
            self._forward,
        ).pack(
            side=tk.LEFT,
            padx=5,
        )

        # ---------------------------------------------------------------------
        # Speed
        # ---------------------------------------------------------------------

        tk.Label(
            controls,
            text="Velocidade",
            bg="#202020",
            fg="#bbbbbb",
            font=("Sans", 10),
        ).pack(
            side=tk.LEFT,
            padx=(22, 6),
        )

        self.speed_combo = ttk.Combobox(
            controls,
            width=7,
            state="readonly",
            style="Speed.TCombobox",
            values=[
                "0.25x",
                "0.5x",
                "0.75x",
                "1x",
                "1.25x",
                "1.5x",
                "2x",
                "3x",
                "4x",
            ],
        )

        self.speed_combo.set("1x")

        self.speed_combo.bind(
            "<<ComboboxSelected>>",
            self._speed_selected,
        )

        self.speed_combo.pack(
            side=tk.LEFT,
            padx=3,
        )

        # ---------------------------------------------------------------------
        # Mark controls
        # ---------------------------------------------------------------------

        self.mark_button = self.make_button(
            controls,
            "Marcar início",
            self._add_mark,
            prominent=True,
        )

        self.mark_button.pack(
            side=tk.LEFT,
            padx=(25, 5),
        )

        self.make_button(
            controls,
            "Desfazer",
            self._undo_mark,
        ).pack(
            side=tk.LEFT,
            padx=5,
        )

        self.make_button(
            controls,
            "Limpar",
            self._clear_marks,
        ).pack(
            side=tk.LEFT,
            padx=5,
        )

        # ---------------------------------------------------------------------
        # Cuts display
        # ---------------------------------------------------------------------

        cuts_frame = tk.Frame(
            panel,
            bg="#181818",
        )

        cuts_frame.pack(
            fill=tk.X,
            padx=10,
            pady=(8, 5),
        )

        title_row = tk.Frame(
            cuts_frame,
            bg="#181818",
        )

        title_row.pack(
            fill=tk.X,
            padx=8,
            pady=(6, 2),
        )

        tk.Label(
            title_row,
            text="Intervalos marcados",
            bg="#181818",
            fg="#eeeeee",
            font=("Sans", 10, "bold"),
        ).pack(
            side=tk.LEFT,
        )

        self.count_label = tk.Label(
            title_row,
            text="0 intervalos",
            bg="#181818",
            fg="#888888",
            font=("Sans", 9),
        )

        self.count_label.pack(
            side=tk.RIGHT,
        )

        self.cuts_var = tk.StringVar(
            value="Nenhum intervalo marcado."
        )

        self.cuts_label = tk.Label(
            cuts_frame,
            textvariable=self.cuts_var,
            bg="#181818",
            fg="#bbbbbb",
            font=("Monospace", 10),
            anchor="w",
            justify=tk.LEFT,
            padx=8,
            pady=6,
        )

        self.cuts_label.pack(
            fill=tk.X,
        )

        # ---------------------------------------------------------------------
        # Footer / shortcuts
        # ---------------------------------------------------------------------

        self.status_var = tk.StringVar(
            value="Carregando vídeo..."
        )

        self.status_label = tk.Label(
            panel,
            textvariable=self.status_var,
            bg="#202020",
            fg="#aaaaaa",
            font=("Monospace", 9),
            anchor="w",
            padx=10,
        )

        self.status_label.pack(
            fill=tk.X,
            pady=(3, 2),
        )

        shortcuts = (
            "A: -5s   D: +5s   Espaço: play/pause   "
            "F: marcar   +/-: velocidade   Q/Esc: sair"
        )

        tk.Label(
            panel,
            text=shortcuts,
            bg="#202020",
            fg="#686868",
            font=("Monospace", 9),
            anchor="w",
            padx=10,
        ).pack(
            fill=tk.X,
            pady=(0, 8),
        )


    # =========================================================================
    # Keyboard
    # =========================================================================

    def _bind_keys(self):

        self.bind_all(
            "<KeyPress-a>",
            lambda e: self._rewind(),
        )

        self.bind_all(
            "<KeyPress-d>",
            lambda e: self._forward(),
        )

        self.bind_all(
            "<KeyPress-space>",
            lambda e: self._pause_toggle(),
        )

        self.bind_all(
            "<KeyPress-f>",
            lambda e: self._add_mark(),
        )

        self.bind_all(
            "<KeyPress-plus>",
            lambda e: self._speed_up(),
        )

        self.bind_all(
            "<KeyPress-equal>",
            lambda e: self._speed_up(),
        )

        self.bind_all(
            "<KeyPress-minus>",
            lambda e: self._speed_down(),
        )

        self.bind_all(
            "<KeyPress-underscore>",
            lambda e: self._speed_down(),
        )

        self.bind_all(
            "<Control-z>",
            lambda e: self._undo_mark(),
        )

        self.bind_all(
            "<KeyPress-q>",
            lambda e: self._quit(),
        )

        self.bind_all(
            "<Escape>",
            lambda e: self._quit(),
        )

        self.protocol(
            "WM_DELETE_WINDOW",
            self._quit,
        )


    # =========================================================================
    # Playback
    # =========================================================================

    def _rewind(self):

        try:
            self.mpv.seek_relative(-5)
        except Exception:
            pass


    def _forward(self):

        try:
            self.mpv.seek_relative(5)
        except Exception:
            pass


    def _pause_toggle(self):

        try:
            self.mpv.pause_toggle()
        except Exception:
            pass


    # =========================================================================
    # Speed
    # =========================================================================

    def _set_speed(self, speed):

        self.rate = max(
            MIN_RATE,
            min(MAX_RATE, speed),
        )

        try:
            self.mpv.set_speed(self.rate)
        except Exception:
            pass

        self.speed_combo.set(
            f"{self.rate:g}x"
        )


    def _speed_selected(self, event=None):

        try:

            speed = float(
                self.speed_combo.get().replace("x", "")
            )

        except ValueError:
            return

        self._set_speed(speed)


    def _speed_up(self):

        self._set_speed(
            self.rate * RATE_STEP
        )


    def _speed_down(self):

        self._set_speed(
            self.rate / RATE_STEP
        )


    # =========================================================================
    # Timeline
    # =========================================================================

    def _seek_using_event(self, event):

        try:
            duration = self.mpv.duration()
        except Exception:
            return

        if duration <= 0:
            return

        width = self.seek_slider.winfo_width()

        if width <= 1:
            return

        # pequena compensação pelo tamanho visual do knob do ttk.Scale
        margin = 8

        usable_width = max(
            1,
            width - (margin * 2),
        )

        x = event.x - margin

        fraction = x / usable_width

        fraction = max(
            0.0,
            min(1.0, fraction),
        )

        target = duration * fraction

        self.seek_var.set(
            fraction * 1000
        )

        try:
            self.mpv.seek_absolute(target)
        except Exception:
            pass


    def _timeline_click(self, event):

        self._slider_dragging = True

        self._seek_using_event(event)


    def _timeline_press(self, event):

        self._slider_dragging = True


    def _timeline_release(self, event):

        self._seek_using_event(event)

        self._slider_dragging = False


    # =========================================================================
    # Marks
    # =========================================================================

    def _add_mark(self):

        try:
            current = self.mpv.current_time()

        except Exception:
            return

        # evita repetição praticamente no mesmo instante
        if self.marks:

            if abs(
                current - self.marks[-1]
            ) < 0.20:

                return

        self.marks.append(
            current
        )

        print(
            f"[MARCAÇÃO] {fmt_seconds(current)} "
            f"({current:.3f}s)"
        )

        self._refresh_marks()


    def _undo_mark(self):

        if not self.marks:
            return

        removed = self.marks.pop()

        print(
            f"[DESFEITO] {fmt_seconds(removed)}"
        )

        self._refresh_marks()


    def _clear_marks(self):

        if not self.marks:
            return

        self.marks.clear()

        print(
            "[INFO] Todas as marcações foram removidas."
        )

        self._refresh_marks()


    def _refresh_marks(self):

        # Próximo botão alterna automaticamente
        if len(self.marks) % 2 == 0:

            self.mark_button.config(
                text="Marcar início"
            )

        else:

            self.mark_button.config(
                text="Marcar fim"
            )

        lines = []

        complete_pairs = (
            len(self.marks) // 2
        )

        for i in range(
            complete_pairs
        ):

            start = self.marks[
                i * 2
            ]

            end = self.marks[
                i * 2 + 1
            ]

            duration = (
                end - start
            )

            lines.append(
                f"{i + 1}.  "
                f"{fmt_seconds(start)}  ->  "
                f"{fmt_seconds(end)}"
                f"    ({duration:.1f}s)"
            )

        # marcador sem par
        if len(self.marks) % 2 == 1:

            start = self.marks[-1]

            lines.append(
                f"{complete_pairs + 1}.  "
                f"{fmt_seconds(start)}  ->  "
                f"aguardando fim"
            )

        if not lines:

            self.cuts_var.set(
                "Nenhum intervalo marcado."
            )

        else:

            self.cuts_var.set(
                "\n".join(lines)
            )

        if complete_pairs == 1:

            count = "1 intervalo"

        else:

            count = (
                f"{complete_pairs} intervalos"
            )

        if len(self.marks) % 2 == 1:

            count += " + 1 aberto"

        self.count_label.config(
            text=count
        )


    # =========================================================================
    # Status
    # =========================================================================

    def _update_status(self):

        if self._quitting:
            return

        try:

            current = self.mpv.current_time()
            duration = self.mpv.duration()

        except Exception:

            self.after(
                400,
                self._update_status,
            )

            return

        self.current_label.config(
            text=fmt_seconds(current)
        )

        self.total_label.config(
            text=fmt_seconds(duration)
        )

        if (
            duration > 0
            and not self._slider_dragging
        ):

            fraction = (
                current / duration
            )

            self.seek_var.set(
                fraction * 1000
            )

        next_mark = (
            "fim"
            if len(self.marks) % 2
            else "início"
        )

        self.status_var.set(
            f"{fmt_seconds(current)} / "
            f"{fmt_seconds(duration)}"
            f"    Velocidade: {self.rate:g}x"
            f"    Próxima marcação: {next_mark}"
        )

        self.after(
            200,
            self._update_status,
        )


    # =========================================================================
    # Quit
    # =========================================================================

    def _quit(self):

        if self._quitting:
            return

        self._quitting = True

        try:
            self.mpv.stop()
        except Exception:
            pass

        self.destroy()


# =============================================================================
# Result
# =============================================================================

def print_result(
    video_path: Path,
    marks: list[float],
):

    print()
    print("=" * 70)
    print("RESULTADO DAS MARCAÇÕES")
    print("=" * 70)

    if not marks:

        print(
            "Nenhum intervalo foi marcado."
        )

        return

    complete_pairs = (
        len(marks) // 2
    )

    for i in range(
        complete_pairs
    ):

        start = marks[
            i * 2
        ]

        end = marks[
            i * 2 + 1
        ]

        print(
            f"{i + 1:02d}. "
            f"{fmt_seconds(start)} -> "
            f"{fmt_seconds(end)}"
        )

    if len(marks) % 2:

        print()
        print(
            "[AVISO] Existe um intervalo sem marcação de fim:"
        )

        print(
            f"       {fmt_seconds(marks[-1])} -> ???"
        )

    print()
    print("-" * 70)
    print("LINHAS PARA cortes_internos.csv")
    print("-" * 70)

    for i in range(
        complete_pairs
    ):

        start = marks[
            i * 2
        ]

        end = marks[
            i * 2 + 1
        ]

        print(
            f"{video_path.name},"
            f"{fmt_csv_time(start)},"
            f"{fmt_csv_time(end)},"
            f"tempo_morto"
        )

    print()
    print(
        "Troque 'tempo_morto' pelo motivo correto "
        "quando souber: timeout, atendimento_jogador, "
        "discussao_arbitragem, intervalo, etc."
    )


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()

    video_path = resolve_video_path(
        args.video
    )

    if not video_path.exists():

        print(
            f"[ERRO] Vídeo não encontrado:\n"
            f"{video_path}"
        )

        sys.exit(1)

    print()
    print(
        f"[INFO] Abrindo:\n"
        f"{video_path}"
    )

    print()
    print(
        "Fluxo de marcação:"
        "\n  F / botão -> início do tempo morto"
        "\n  F / botão -> fim do tempo morto"
        "\n  repetir para os próximos intervalos"
    )

    print()

    app = CourtSlicerApp(
        video_path
    )

    app.mainloop()

    print_result(
        video_path,
        app.marks,
    )


if __name__ == "__main__":
    main()