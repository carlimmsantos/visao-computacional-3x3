import json
from pathlib import Path

from .models import DownloadItem

REQUIRED_FIELDS = ("url", "filename", "format")


def load_download_items(json_path: Path) -> list[DownloadItem]:
    if not json_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        raw_items = json.load(f)

    if not isinstance(raw_items, list):
        raise ValueError(f"'{json_path}' deve conter uma lista de itens.")

    items: list[DownloadItem] = []
    for index, raw_item in enumerate(raw_items):
        missing = [field for field in REQUIRED_FIELDS if field not in raw_item]
        if missing:
            raise ValueError(
                f"Item {index} em '{json_path}' está incompleto. "
                f"Campos ausentes: {', '.join(missing)}."
            )
        items.append(
            DownloadItem(
                url=raw_item["url"],
                filename=raw_item["filename"],
                format=raw_item["format"],
            )
        )

    return items
