"""
Corta as partidas dos vídeos brutos, usando os timestamps definidos em uma
planilha CSV (veja templates/partidas_template.csv).

Cada linha do CSV vira um arquivo de vídeo separado, já com o nome da partida.

Uso:
    python extrair_partidas.py --csv templates/partidas.csv --videos_dir video_brutos --output_dir partidas_cortadas

Formato do CSV (colunas obrigatórias):
    video,inicio,fim,nome_partida

- video: nome do arquivo do vídeo bruto (deve estar em --videos_dir)
- inicio / fim: timestamps no formato HH:MM:SS, relativos ao vídeo BRUTO
- nome_partida: nome que o arquivo de saída vai receber (sem extensão)

Por padrão usa "-c copy" (corte rápido, sem re-codificar). Se os cortes
saírem imprecisos (alguns segundos de diferença), use --reencode para
cortar exatamente no timestamp pedido (mais lento).
"""

import argparse
import csv
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Corta partidas dos vídeos brutos")
    parser.add_argument("--csv", type=str, required=True, help="Caminho do CSV de partidas")
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="video_brutos",
        help="Pasta onde estão os vídeos brutos (default: video_brutos)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="partidas_cortadas",
        help="Pasta de saída das partidas cortadas (default: partidas_cortadas)",
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Re-codifica o vídeo para cortar exatamente no timestamp (mais lento, mais preciso)",
    )
    return parser.parse_args()


def cortar_video(video_in, inicio, fim, video_out, reencode=False):
    if reencode:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_in),
            "-ss", inicio,
            "-to", fim,
            "-c:v", "libx264",
            "-c:a", "aac",
            str(video_out),
        ]
    else:
        # -ss antes do -i é mais rápido (seek), mas com -c copy o corte
        # pode ficar alguns frames impreciso (limitação de keyframe).
        cmd = [
            "ffmpeg", "-y",
            "-ss", inicio,
            "-to", fim,
            "-i", str(video_in),
            "-c", "copy",
            str(video_out),
        ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        print(f"[ERRO] Falha ao cortar {video_out.name}:")
        print(result.stderr[-1000:])
        return False
    return True


def main():
    args = parse_args()
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        linhas = list(reader)

    print(f"[INFO] {len(linhas)} partida(s) para cortar.")

    sucesso, falha = 0, 0
    for linha in linhas:
        nome_video_bruto = linha["video"].strip()
        video_in = videos_dir / nome_video_bruto
        nome_partida = linha["nome_partida"].strip()
        inicio = linha["inicio"].strip()
        fim = linha["fim"].strip()

        pasta_video = output_dir / Path(nome_video_bruto).stem
        pasta_video.mkdir(parents=True, exist_ok=True)
        video_out = pasta_video / f"{nome_partida}.mp4"

        if not video_in.exists():
            print(f"[AVISO] Vídeo não encontrado, pulando: {video_in}")
            falha += 1
            continue

        print(f"[INFO] Cortando '{nome_partida}' de {video_in.name} ({inicio} -> {fim})...")
        ok = cortar_video(video_in, inicio, fim, video_out, reencode=args.reencode)
        if ok:
            sucesso += 1
        else:
            falha += 1

    print(f"\n[INFO] Concluído: {sucesso} partida(s) cortada(s) com sucesso, {falha} falha(s).")
    print(f"[INFO] Arquivos salvos em: {output_dir}")


if __name__ == "__main__":
    main()
