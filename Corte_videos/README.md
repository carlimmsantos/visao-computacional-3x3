# Ferramentas de corte de vídeo — LPB Vision

Duas etapas, cada uma com seu script e sua planilha CSV. Ninguém da equipe
precisa mexer em código — só preencher a planilha com os timestamps.

## Fluxo geral

```
vídeo bruto (ex: video1.mp4, 3h)
        │
        ▼  (planilha 1: partidas.csv)
partida cortada (ex: Jogo1_TimeA_vs_TimeB.mp4)
        │
        ▼  (planilha 2: cortes_internos.csv)
partida limpa, sem tempo morto (mesmo nome, pasta partidas_limpas/)
```

## Passo 0 — Preparar o ambiente

```bash
pip install --break-system-packages pandas  # opcional, só se for editar CSV via Python
# ffmpeg precisa estar instalado (confira com: ffmpeg -version)
```

Coloque os 3 vídeos brutos dentro da pasta `video_brutos/`.

## Passo 1 — Cortar as partidas

Durante a reunião, preencham a planilha `templates/partidas_template.csv`
(copiem para `templates/partidas.csv` e editem) com uma linha por partida:

```csv
video,inicio,fim,nome_partida
video1.mp4,00:44:02,01:32:15,Jogo1_TimeA_vs_TimeB
```

- **video**: nome do arquivo bruto (dentro de `videos_brutos/`)
- **inicio / fim**: timestamp `HH:MM:SS` no vídeo bruto (o mesmo que aparece
  no player, tipo `44:02` → `00:44:02`)
- **nome_partida**: nome do arquivo de saída (sem espaços, sem extensão)

Rodar (a partir da pasta `Corte_videos/`):

```powershell
cd Corte_videos
python extrair_partidas.py --csv templates/partidas.csv --videos_dir video_brutos --output_dir partidas_cortadas
```

Se preferir quebrar a linha no PowerShell, use o acento grave (`` ` ``) no
lugar da barra invertida (`\`) — o PowerShell não entende `\` como
continuação de linha (isso é sintaxe do bash) e gera o erro
`Expressão ausente após operador unário '--'`:

```powershell
python extrair_partidas.py `
    --csv templates/partidas.csv `
    --videos_dir video_brutos `
    --output_dir partidas_cortadas
```

Isso gera um arquivo `.mp4` por partida em `partidas_cortadas/`, organizados
em uma subpasta por vídeo bruto (uma pasta por nome de vídeo de origem, com
as partidas daquele vídeo dentro):

```
partidas_cortadas/
├── Torneio 3X3 - LPB/
│   ├── Jogo1_ICP_vs_JICS.mp4
│   ├── Jogo2_GUARANI_vs_FÚRIA_B.mp4
│   └── ... (16 partidas)
├── Torneio 3X3 - LPB - 2/
│   └── ... (8 partidas)
└── Torneio 3X3 - LPB - 3/
    └── ... (6 partidas)
```

**Se os cortes saírem meio errados** (começando alguns segundos antes/depois
do pedido), rode de novo com `--reencode` — fica mais lento, mas corta exato
no timestamp.

## Passo 2 — Remover tempo morto de dentro de cada partida

Com as partidas já separadas, a equipe assiste cada uma e preenche
`templates/cortes_internos_template.csv` (copiem para `cortes_internos.csv`)
marcando os trechos a remover:

```csv
video,inicio,fim,motivo
Jogo1_TimeA_vs_TimeB.mp4,00:05:10,00:05:45,timeout
```

**Importante**: aqui o timestamp é relativo ao vídeo **já cortado** da
partida (começa do zero), não ao vídeo bruto original. A coluna `video`
é só o nome do arquivo (ex: `Jogo1_TimeA_vs_TimeB.mp4`), independente de em
qual subpasta de `partidas_cortadas/` ele está — o script procura em todas
as subpastas e replica a mesma estrutura em `partidas_limpas/`.

`motivo` é só documentação — não afeta o corte, mas ajuda a saber depois
por que aquele trecho foi removido (isso vira dado valioso pra treinar um
classificador automático de tempo morto no futuro, então capricha em
preencher com precisão).

Rodar:

```powershell
python remover_trechos.py --csv templates/cortes_internos.csv --videos_dir partidas_cortadas --output_dir partidas_limpas
```

O resultado final — vídeos só com jogo ativo, prontos pra próxima etapa
(anotação) — fica em `partidas_limpas/`.

## Checklist sugerido do que cortar (alinhar com a equipe antes de começar)

**Cortar (remover):**
- Timeouts
- Intervalo entre períodos/partidas
- Jogador caído / atendimento médico
- Qualquer trecho sem bola em jogo

**Manter:**
- Qualquer momento com bola em jogo ativo, mesmo que devagar (aquecimento
  de posse, jogada lenta) — não cortar só porque "parece parado"

## Estrutura de pastas

```
Corte_videos/
├── video_brutos/            # os 3 vídeos originais (11h) — vocês colocam aqui
├── partidas_cortadas/       # saída do Passo 1
├── partidas_limpas/         # saída do Passo 2 (resultado final)
├── templates/
│   ├── partidas_template.csv
│   └── cortes_internos_template.csv
├── extrair_partidas.py
└── remover_trechos.py
```
