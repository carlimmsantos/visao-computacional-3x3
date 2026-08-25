import sys
from pathlib import Path

from yt_dlp import YoutubeDL

from .models import DownloadItem


def _format_progress_hook(item: DownloadItem):
    def hook(status: dict) -> None:
        if status["status"] == "downloading":
            percent = status.get("_percent_str", "?").strip()
            speed = status.get("_speed_str", "?").strip()
            eta = status.get("_eta_str", "?").strip()

            fragment_index = status.get("fragment_index")
            fragment_count = status.get("fragment_count")
            if fragment_index and fragment_count:
                fragment_info = f" | fragmento {fragment_index}/{fragment_count}"
            else:
                fragment_info = ""

            info = status.get("info_dict") or {}
            stream_kind = "audio" if info.get("acodec") not in (None, "none") and info.get("vcodec") in (None, "none") else "video"

            sys.stdout.write(
                f"\r[{item.filename}] baixando {stream_kind}: "
                f"{percent} a {speed}, ETA {eta}{fragment_info}   "
            )
            sys.stdout.flush()
        elif status["status"] == "finished":
            sys.stdout.write(f"\r[{item.filename}] download concluído, processando...\n")
            sys.stdout.flush()

    return hook


_RESILIENCE_OPTIONS = {
    "retries": 10,
    "fragment_retries": 10,
    "extractor_retries": 5,
    "file_access_retries": 5,
    "continuedl": True,
    "live_from_start": True,
    # "web" às vezes é forçado a SABR streaming sem formatos utilizáveis
    # (https://github.com/yt-dlp/yt-dlp/issues/12482); "android" contorna isso.
    "extractor_args": {"youtube": {"player_client": ["default", "android"]}},
}


class YouTubeDownloader:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self, item: DownloadItem) -> Path:
        if item.format == "mp4":
            return self._download_video(item)
        return self._download_audio(item)

    def _download_video(self, item: DownloadItem) -> Path:
        output_template = str(self.output_dir / f"{item.filename}.%(ext)s")
        options = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": output_template,
            "merge_output_format": "mp4",
            "quiet": True,
            "noprogress": False,
            "progress_hooks": [_format_progress_hook(item)],
            **_RESILIENCE_OPTIONS,
        }
        with YoutubeDL(options) as ydl:
            ydl.download([item.url])
        return self.output_dir / f"{item.filename}.mp4"

    def _download_audio(self, item: DownloadItem) -> Path:
        output_template = str(self.output_dir / f"{item.filename}.%(ext)s")
        options = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
            "noprogress": False,
            "progress_hooks": [_format_progress_hook(item)],
            **_RESILIENCE_OPTIONS,
        }
        with YoutubeDL(options) as ydl:
            ydl.download([item.url])
        return self.output_dir / f"{item.filename}.mp3"
