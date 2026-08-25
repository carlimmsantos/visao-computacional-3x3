# YouTube Downloader

Script para baixar vídeos (mp4) e músicas/áudio (mp3) do YouTube a partir de uma lista configurável.

## Estrutura

```
YouTube/
├── requirements.txt
├── downloads.json           # lista de itens a baixar
├── main.py                  # ponto de entrada
└── youtube_downloader/
    ├── models.py             # DownloadItem
    ├── config.py             # carregamento/validação do downloads.json
    └── downloader.py         # YouTubeDownloader (baixa vídeo ou extrai áudio)
```

## Pré-requisitos

1. Python 3.9+
2. [ffmpeg](https://ffmpeg.org/download.html) instalado e disponível no PATH — necessário para extrair áudio em mp3.
3. Instalar as dependências:

```bash
pip install -r requirements.txt
```

## Como usar

1. Edite o arquivo `downloads.json` adicionando um item para cada vídeo/música que deseja baixar:

```json
[
  { "url": "https://www.youtube.com/watch?v=XXXXXXX", "filename": "meu_video", "format": "mp4" },
  { "url": "https://www.youtube.com/watch?v=YYYYYYY", "filename": "minha_musica", "format": "mp3" }
]
```

   - `url`: link do vídeo no YouTube.
   - `filename`: nome do arquivo de saída (sem extensão).
   - `format`: `"mp4"` para vídeo ou `"mp3"` para áudio.

2. Rode o script:

```bash
python main.py
```

3. Os arquivos baixados aparecem na pasta `output/`, com o nome e formato indicados em cada item. Erros em um item não interrompem o processamento dos demais — o script imprime `[OK]` ou `[ERRO]` para cada um.

## Vídeos do Torneio

Os vídeos do Torneio 3X3 - LPB estão disponíveis no Google Drive: https://drive.google.com/drive/folders/15YOSKSP0Mk9FFCk6YIpDnGTZhyWk_wnR?usp=sharing
