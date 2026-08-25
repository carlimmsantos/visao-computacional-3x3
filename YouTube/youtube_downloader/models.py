from dataclasses import dataclass

SUPPORTED_FORMATS = ("mp4", "mp3")


@dataclass(frozen=True)
class DownloadItem:
    url: str
    filename: str
    format: str

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("O campo 'url' não pode estar vazio.")
        if not self.filename:
            raise ValueError("O campo 'filename' não pode estar vazio.")
        if self.format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Formato '{self.format}' inválido para '{self.filename}'. "
                f"Use um dos seguintes: {', '.join(SUPPORTED_FORMATS)}."
            )
