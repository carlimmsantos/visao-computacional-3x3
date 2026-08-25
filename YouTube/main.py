import sys
from pathlib import Path

from youtube_downloader import YouTubeDownloader, load_download_items

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "downloads.json"
OUTPUT_DIR = BASE_DIR / "output"


def main() -> None:
    items = load_download_items(CONFIG_PATH)
    downloader = YouTubeDownloader(OUTPUT_DIR)

    for item in items:
        try:
            saved_path = downloader.download(item)
            print(f"[OK] {item.filename} ({item.format}) -> {saved_path}")
        except Exception as error:
            print(f"[ERRO] {item.filename} ({item.url}): {error}")


if __name__ == "__main__":
    main()
