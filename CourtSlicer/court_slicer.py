"""CourtSlicer — flag basketball plays and cut video into clips."""

import argparse
import json
import re
import subprocess
import sys
import platform
from pathlib import Path

import tkinter as tk
import vlc

PROJECT_ROOT = Path(__file__).resolve().parent
VIDEOS_ENTRADA = PROJECT_ROOT / "videos_entrada"
VIDEOS_SAIDA = PROJECT_ROOT / "videos_saida"

MIN_RATE = 0.25
MAX_RATE = 4.0
RATE_STEP = 1.25

TRECHO_NAME_RE = re.compile(r"trecho_(\d+)\.mp4$")


def parse_args():
    parser = argparse.ArgumentParser(description="Flag plays and cut video clips.")
    parser.add_argument(
        "video",
        nargs="?",
        help=(
            "Video file: a bare filename is looked up inside videos_entrada/, "
            "or pass a full/relative path directly."
        ),
    )
    parser.add_argument(
        "--editar",
        metavar="TRECHO",
        help=(
            "Reabre um trecho já cortado (ex: videos_saida/partida1/trecho_02.mp4) "
            "para ajustar o início/fim e regravá-lo."
        ),
    )
    args = parser.parse_args()
    if not args.video and not args.editar:
        parser.error("informe um vídeo para cortar, ou --editar <trecho.mp4>")
    return args


def resolve_video_path(video_arg: str) -> Path:
    """Resolve the input video path: bare filename -> videos_entrada/, else as given."""
    given = Path(video_arg)
    if given.exists():
        return given.resolve()

    candidate = VIDEOS_ENTRADA / given.name
    if candidate.exists():
        return candidate.resolve()

    return given.resolve()


# ── metadata (flags.json) ───────────────────────────────────────────────────

def metadata_path(out_dir: Path) -> Path:
    return out_dir / "flags.json"


def load_metadata(out_dir: Path) -> dict:
    path = metadata_path(out_dir)
    if not path.exists():
        raise FileNotFoundError(f"Metadados não encontrados: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_metadata(out_dir: Path, source_name: str, segments: list[dict]):
    data = {"source": source_name, "segments": segments}
    metadata_path(out_dir).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def update_segment_metadata(out_dir: Path, index: int, start: float, end: float):
    data = load_metadata(out_dir)
    for seg in data["segments"]:
        if seg["index"] == index:
            seg["start"] = start
            seg["end"] = end
            break
    else:
        data["segments"].append({"index": index, "start": start, "end": end})
    data["segments"].sort(key=lambda s: s["index"])
    metadata_path(out_dir).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


class CourtSlicerApp(tk.Tk):
    def __init__(self, video_path: str, edit_range: tuple[float, float] | None = None):
        super().__init__()
        self.video_path = Path(video_path)
        self.flags: list[int] = []  # timestamps in ms
        self.total_ms: int = 0
        self.rate: float = 1.0
        self._quitting = False

        self.edit_mode = edit_range is not None
        if self.edit_mode:
            self.new_start, self.new_end = edit_range
        else:
            self.new_start = self.new_end = None

        self.title("CourtSlicer" + (" — editando trecho" if self.edit_mode else ""))
        self.resizable(False, False)
        self.configure(bg="black")

        self._build_ui()
        self._setup_vlc(video_path)
        self._bind_keys()

        # Embed VLC after window is realized
        self.after(150, self._embed_vlc)
        # Start status polling
        self.after(250, self._update_status)

    def _build_ui(self):
        self.canvas = tk.Canvas(self, width=1280, height=720, bg="black",
                                highlightthickness=0)
        self.canvas.pack()

        self.status_var = tk.StringVar(value="Loading...")
        self.status_bar = tk.Label(
            self,
            textvariable=self.status_var,
            bg="#1a1a1a",
            fg="#e0e0e0",
            font=("Monospace", 11),
            anchor="w",
            padx=8,
        )
        self.status_bar.pack(fill=tk.X)

        if self.edit_mode:
            help_text = (
                "  A: -5s   D: +5s   SPACE: pause/play   +/-: speed   "
                "I: novo início   O: novo fim   Q/Esc: salvar e sair"
            )
        else:
            help_text = (
                "  A: -5s   D: +5s   SPACE: pause/play   +/-: speed   "
                "F: flag   Q/Esc: quit & cut"
            )
        tk.Label(
            self,
            text=help_text,
            bg="#111111",
            fg="#888888",
            font=("Monospace", 10),
            anchor="w",
            padx=8,
        ).pack(fill=tk.X)

    def _setup_vlc(self, path: str):
        self.instance = vlc.Instance(
            "--quiet",
            "--no-video-title-show",
        )
        self.player = self.instance.media_player_new()
        media = self.instance.media_new_path(path)
        self.player.set_media(media)

        # Parse for duration at startup
        media.parse_with_options(vlc.MediaParseFlag.local, timeout=5000)
        self.total_ms = media.get_duration()

    def _embed_vlc(self):
        wid = self.canvas.winfo_id()
        system = platform.system()
        if system == "Windows":
            self.player.set_hwnd(wid)
        elif system == "Darwin":
            self.player.set_nsobject(wid)
        else:
            self.player.set_xwindow(wid)
        self.player.play()
        self.after(500, self._check_state)
        self.canvas.focus_set()
        self.after(600, self._grab_focus)

        if self.edit_mode:
            # Seek a couple seconds before the current start so the user has
            # context around the existing in-point.
            preroll_ms = max(0, int((self.new_start - 2.0) * 1000))
            self.after(700, lambda: self.player.set_time(preroll_ms))

    def _grab_focus(self):
        # On Windows the embedded VLC output can steal keyboard focus;
        # force it back to the tk window so key bindings keep working.
        self.focus_force()
        self.canvas.focus_set()

    def _check_state(self):
        state = self.player.get_state()
        if state == vlc.State.Error:
            mrl = self.player.get_media().get_mrl() if self.player.get_media() else "?"
            print(f"Error: VLC could not open media: {mrl}")

    def _bind_keys(self):
        self.bind("<KeyPress-a>", lambda e: self._on_rewind())
        self.bind("<KeyPress-d>", lambda e: self._on_ff())
        self.bind("<KeyPress-space>", lambda e: self._on_pause_toggle())
        self.bind("<KeyPress-plus>", lambda e: self._on_speed_up())
        self.bind("<KeyPress-equal>", lambda e: self._on_speed_up())
        self.bind("<KeyPress-minus>", lambda e: self._on_speed_down())
        self.bind("<KeyPress-underscore>", lambda e: self._on_speed_down())
        self.bind("<KeyPress-q>", lambda e: self._on_quit())
        self.bind("<Escape>", lambda e: self._on_quit())
        if self.edit_mode:
            self.bind("<KeyPress-i>", lambda e: self._on_set_in())
            self.bind("<KeyPress-o>", lambda e: self._on_set_out())
        else:
            self.bind("<KeyPress-f>", lambda e: self._on_flag())
        self.focus_set()

    def _on_rewind(self):
        current = self.player.get_time()
        self.player.set_time(max(0, current - 5000))

    def _on_ff(self):
        current = self.player.get_time()
        self.player.set_time(current + 5000)

    def _on_pause_toggle(self):
        self.player.pause()

    def _on_speed_up(self):
        self._set_rate(self.rate * RATE_STEP)

    def _on_speed_down(self):
        self._set_rate(self.rate / RATE_STEP)

    def _set_rate(self, new_rate: float):
        self.rate = max(MIN_RATE, min(MAX_RATE, new_rate))
        self.player.set_rate(self.rate)

    def _on_flag(self):
        t = self.player.get_time()
        if t >= 0:
            self.flags.append(t)
            self._refresh_status(t)

    def _on_set_in(self):
        self.new_start = self.player.get_time() / 1000.0
        self._refresh_status()

    def _on_set_out(self):
        self.new_end = self.player.get_time() / 1000.0
        self._refresh_status()

    def _on_quit(self):
        if self._quitting:
            return
        self._quitting = True

        length = self.player.get_length()
        if length > 0:
            self.total_ms = length

        self.player.stop()
        self.destroy()

    def _refresh_status(self, flagged_at: int = -1):
        current = self.player.get_time()
        total = self.total_ms
        base = f"  {_fmt_ms(current)} / {_fmt_ms(total)}   |   Speed: {self.rate:.2f}x"
        if self.edit_mode:
            msg = (
                f"{base}   |   Início: {self.new_start:.1f}s"
                f"   Fim: {self.new_end:.1f}s"
            )
        else:
            flags_str = ", ".join(f"{ms/1000:.1f}s" for ms in self.flags) or "none"
            msg = f"{base}   |   Flags: [{flags_str}]"
            if flagged_at >= 0:
                msg += f"   ← flagged {flagged_at/1000:.1f}s"
        self.status_var.set(msg)

    def _update_status(self):
        if self._quitting:
            return
        self._refresh_status()
        self.after(250, self._update_status)


def _fmt_ms(ms: int) -> str:
    if ms < 0:
        return "--:--"
    s = ms // 1000
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"


def _get_duration_ffprobe(path: Path) -> float:
    """Fallback: ask ffprobe for duration in seconds."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def cut_segment(source: Path, start: float, end: float, output_path: Path) -> bool:
    """Cut a single [start, end] (seconds) segment from source into output_path."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(source),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"    WARNING: ffmpeg exited {result.returncode}")
        print(result.stderr[-500:] if result.stderr else "")
        return False
    return True


def cut_video(source: Path, boundaries: list[float]) -> tuple[int, Path]:
    """Cut source into segments defined by boundaries (seconds).

    Clips go to videos_saida/<source_stem>/trecho_NN.mp4. Writes flags.json
    alongside them so individual trechos can later be reopened with --editar.
    """
    out_dir = VIDEOS_SAIDA / source.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    total_cut = 0
    segments: list[dict] = []

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        if end - start < 0.5:
            print(f"  Skipping segment {i+1}: too short ({end - start:.2f}s)")
            continue

        index = i + 1
        output_path = out_dir / f"trecho_{index:02d}.mp4"
        print(f"  Cutting segment {index}: {start:.1f}s → {end:.1f}s → {output_path.name}")
        if cut_segment(source, start, end, output_path):
            total_cut += 1
            segments.append({"index": index, "start": start, "end": end})

    save_metadata(out_dir, source.name, segments)
    return total_cut, out_dir


# ── edit flow ────────────────────────────────────────────────────────────────

def run_edit(trecho_arg: str):
    trecho_path = Path(trecho_arg).resolve()
    if not trecho_path.exists():
        print(f"Error: trecho não encontrado: {trecho_path}")
        sys.exit(1)

    match = TRECHO_NAME_RE.search(trecho_path.name)
    if not match:
        print(f"Error: '{trecho_path.name}' não parece um arquivo trecho_NN.mp4")
        sys.exit(1)
    index = int(match.group(1))

    out_dir = trecho_path.parent
    try:
        data = load_metadata(out_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("(esse trecho foi cortado antes da introdução do flags.json e não pode ser editado automaticamente)")
        sys.exit(1)

    segment = next((s for s in data["segments"] if s["index"] == index), None)
    if segment is None:
        print(f"Error: nenhum registro do trecho {index} em {metadata_path(out_dir)}")
        sys.exit(1)

    source_path = VIDEOS_ENTRADA / data["source"]
    if not source_path.exists():
        print(f"Error: vídeo original '{data['source']}' não encontrado em {VIDEOS_ENTRADA}")
        print("Coloque o vídeo bruto de volta em videos_entrada/ para poder editar este trecho.")
        sys.exit(1)

    print(f"Editando trecho {index} de {trecho_path.name}")
    print(f"Início/fim atuais: {segment['start']:.1f}s → {segment['end']:.1f}s")
    print("Controles: A=-5s  D=+5s  SPACE=pause  +/-=speed  I=novo início  O=novo fim  Q/Esc=salvar\n")

    app = CourtSlicerApp(str(source_path), edit_range=(segment["start"], segment["end"]))
    app.mainloop()

    new_start, new_end = app.new_start, app.new_end
    if new_end - new_start < 0.5:
        print("Início/fim inválidos (trecho ficaria menor que 0.5s) — edição cancelada.")
        sys.exit(1)

    print(f"\nNovo início/fim: {new_start:.1f}s → {new_end:.1f}s")
    print(f"Regravando {trecho_path.name}...")
    if not cut_segment(source_path, new_start, new_end, trecho_path):
        print("Error: falha ao regravar o trecho.")
        sys.exit(1)

    update_segment_metadata(out_dir, index, new_start, new_end)
    print(f"Concluído — {trecho_path} atualizado.")


def main():
    args = parse_args()

    if args.editar:
        run_edit(args.editar)
        return

    video_path = resolve_video_path(args.video)

    if not video_path.exists():
        print(f"Error: file not found: {video_path}")
        print(f"(looked directly, and inside {VIDEOS_ENTRADA})")
        sys.exit(1)

    print(f"Opening: {video_path}")
    print("Controls: A=-5s  D=+5s  SPACE=pause  +/-=speed  F=flag  Q/Esc=quit & cut\n")

    app = CourtSlicerApp(str(video_path))
    app.mainloop()

    flags = app.flags
    total_ms = app.total_ms

    if not flags:
        print("No flags set — nothing to cut.")
        sys.exit(0)

    # Resolve total duration
    if total_ms <= 0:
        print("Resolving duration via ffprobe...")
        total_ms = int(_get_duration_ffprobe(video_path) * 1000)

    if total_ms <= 0:
        print("Error: could not determine video duration.")
        sys.exit(1)

    flags_sorted = sorted(set(flags))
    boundaries = [0.0] + [f / 1000.0 for f in flags_sorted] + [total_ms / 1000.0]

    print(f"\nFlags ({len(flags_sorted)}): {[f/1000 for f in flags_sorted]}")
    print(f"Total duration: {total_ms/1000:.1f}s")
    print(f"Cutting {len(boundaries)-1} segment(s)...\n")

    n, out_dir = cut_video(video_path, boundaries)
    print(f"\nDone — {n} clip(s) written to {out_dir}/")


if __name__ == "__main__":
    main()
