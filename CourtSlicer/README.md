# CourtSlicer

Assista ao vídeo da partida, marque os lances em tempo real e corte o vídeo
automaticamente em trechos. O áudio é preservado (apito do árbitro, torcida).
Os cortes são instantâneos e sem perda de qualidade, via ffmpeg `-c copy`.

Funciona tanto no **Windows** quanto no **Linux** (e macOS).

Baseado no projeto [CourtSlicer](https://github.com/GabrielCFormiga/CourtSlicer)
de Gabriel Campelo Formiga (MIT), com correções para rodar nativamente no
Windows, controle de velocidade de reprodução e uma organização de
pastas de entrada/saída por partida.

## Requisitos

- Python ≥ 3.10
- [ffmpeg](https://ffmpeg.org/) (inclui `ffprobe`)
- [VLC media player](https://www.videolan.org/vlc/) instalado no sistema
  (a biblioteca nativa `libvlc` — não basta o pacote Python `python-vlc`)
- `tkinter` (interface gráfica e captura de teclado)

## Instalação

### Windows (PowerShell)

```powershell
cd CourtSlicer
.\install.ps1
```

O script verifica Python ≥3.10, tkinter, ffmpeg/ffprobe e VLC, instala o
[uv](https://docs.astral.sh/uv/) se necessário, e cria o ambiente virtual com
o comando `court-slicer`.

Se o PowerShell bloquear a execução do script, rode uma vez:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Instalar as dependências de sistema manualmente, se preferir:
```powershell
winget install ffmpeg
winget install VideoLAN.VLC
```

Opções: `-Yes` (não interativo, instala uv automaticamente), `-NoUv` (força pip em vez de uv).

### Linux / macOS (bash)

```sh
cd CourtSlicer
bash install.sh
```

O script verifica Python ≥3.10, tkinter, ffmpeg/ffprobe e VLC, instala o `uv`
se necessário, e cria o ambiente virtual com o comando `court-slicer`.

Instalar as dependências de sistema manualmente, se preferir (Debian/Ubuntu):
```sh
sudo apt install python3-tk ffmpeg vlc
```

Opções: `--yes` (não interativo, instala uv automaticamente), `--no-uv` (força pip em vez de uv).

## Uso

Coloque os vídeos brutos das partidas em `videos_entrada/`. Depois:

**Windows:**
```powershell
.venv\Scripts\court-slicer.exe videos_entrada\partida1.mp4
```
ou, com o ambiente ativado (`.venv\Scripts\Activate.ps1`):
```powershell
court-slicer partida1.mp4
```

**Linux/macOS:**
```sh
.venv/bin/court-slicer videos_entrada/partida1.mp4
```
ou, com o ambiente ativado (`source .venv/bin/activate`):
```sh
court-slicer partida1.mp4
```

Se você passar apenas o nome do arquivo (sem caminho), o CourtSlicer procura
automaticamente dentro de `videos_entrada/`.

### Controles

| Tecla | Ação |
|-------|------|
| `F` | Marca o instante atual como corte |
| `Espaço` | Pausa / retoma |
| `A` | Volta 5 segundos |
| `D` | Avança 5 segundos |
| `+` | Aumenta a velocidade de reprodução |
| `-` | Diminui a velocidade de reprodução |
| `Q` / `Esc` | Sai e gera os cortes |

A velocidade varia entre 0.25x e 4x, em passos multiplicativos de 1.25x, e é
mostrada na barra de status junto com o tempo atual e as marcações feitas.

### Editando um trecho já cortado

Errou uma marcação e um `trecho_NN.mp4` ficou com o início ou o fim errado?
Não precisa refazer a partida inteira — reabra só aquele trecho:

**Windows:**
```powershell
court-slicer --editar videos_saida\partida1\trecho_02.mp4
```

**Linux/macOS:**
```sh
court-slicer --editar videos_saida/partida1/trecho_02.mp4
```

O CourtSlicer reabre o vídeo original (precisa continuar em `videos_entrada/`)
já posicionado perto do início atual desse trecho. Os controles de
navegação/velocidade (`A`, `D`, `Espaço`, `+`, `-`) continuam funcionando, e
mais duas teclas ficam disponíveis:

| Tecla | Ação |
|-------|------|
| `I` | Marca o instante atual como novo início do trecho |
| `O` | Marca o instante atual como novo fim do trecho |
| `Q` / `Esc` | Regrava o `trecho_NN.mp4` com o novo início/fim e sai |

A barra de status mostra o início e o fim atuais enquanto você ajusta. Ao
sair, o arquivo do trecho é sobrescrito e o registro em `flags.json` (dentro
da pasta da partida) é atualizado — os demais trechos da partida não são
afetados.

### Estrutura de pastas

```
CourtSlicer/
├── videos_entrada/
│   ├── partida1.mp4
│   └── partida2.mp4
└── videos_saida/
    ├── partida1/
    │   ├── trecho_01.mp4
    │   ├── trecho_02.mp4
    │   ├── ...
    │   └── flags.json
    └── partida2/
        ├── trecho_01.mp4
        ├── ...
        └── flags.json
```

Cada vídeo processado gera sua própria subpasta em `videos_saida/`, nomeada
igual ao arquivo de entrada (sem extensão), com os trechos numerados
sequencialmente (`trecho_01.mp4`, `trecho_02.mp4`, ...): do início até a
primeira marcação, da primeira até a segunda, e assim por diante até o fim
do vídeo. O `flags.json` guarda o vídeo de origem e o início/fim (em
segundos) de cada trecho — é o que permite reabrir um trecho específico com
`--editar` depois.

## Notas

- Os cortes caem no keyframe mais próximo (até ~2s antes da marcação).
  Um pequeno trecho extra no início do clipe é esperado e normal.
- Marcações duplicadas são ignoradas automaticamente.
- Trechos menores que 0.5s são descartados.
