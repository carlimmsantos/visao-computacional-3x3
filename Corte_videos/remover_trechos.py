"""
Remove trechos de tempo morto (timeout, intervalo, jogador caído, etc.) de
dentro de cada partida já cortada, usando uma planilha CSV com os trechos
a excluir (veja templates/cortes_internos_template.csv).

Funciona assim: calcula os trechos que devem ser MANTIDOS (o complemento dos
trechos a excluir) e concatena só esses pedaços num vídeo final "limpo".

Uso:
    python remover_trechos.py --csv templates/cortes_internos.csv --videos_dir partidas_cortadas --output_dir partidas_limpas

Formato do CSV (colunas obrigatórias):
    video,inicio,fim,motivo

- video: nome do arquivo da partida JÁ CORTADA (deve estar em --videos_dir)
- inicio / fim: timestamps no formato HH:MM:SS, relativos a ESSA partida
  (ou seja, começando do 00:00:00 dela, não do vídeo bruto original)
- motivo: só documentação (timeout, intervalo, etc.) — não afeta o corte

Se um vídeo não tiver nenhuma linha no CSV, ele é apenas copiado sem
alterações para a pasta de saída.
"""

import argparse
import csv
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Remove tempo morto de dentro das partidas")
    parser.add_argument("--csv", type=str, required=True, help="Caminho do CSV de cortes internos")
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="partidas_cortadas",
        help="Pasta com as partidas já cortadas (default: partidas_cortadas)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="partidas_limpas",
        help="Pasta de saída das partidas limpas (default: partidas_limpas)",
    )
    return parser.parse_args()


def hhmmss_to_seconds(t):
    partes = [float(p) for p in t.strip().split(":")]
    while len(partes) < 3:
        partes.insert(0, 0.0)
    h, m, s = partes
    return h * 3600 + m * 60 + s


def get_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return float(result.stdout.strip())


def calcular_trechos_mantidos(duracao_total, trechos_excluir):
    # trechos_excluir: lista de (inicio_seg, fim_seg), já ordenada
    trechos_excluir = sorted(trechos_excluir)
    mantidos = []
    cursor = 0.0
    for inicio, fim in trechos_excluir:
        if inicio > cursor:
            mantidos.append((cursor, inicio))
        cursor = max(cursor, fim)
    if cursor < duracao_total:
        mantidos.append((cursor, duracao_total))
    return mantidos


def extrair_trecho(video_in, inicio_seg, fim_seg, saida):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(inicio_seg),
        "-to", str(fim_seg),
        "-i", str(video_in),
        "-c:v", "libx264",
        "-c:a", "aac",
        str(saida),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return result.returncode == 0


def concatenar(trechos_arquivos, saida, tmp_dir):
    lista_path = tmp_dir / "lista_concat.txt"
    with open(lista_path, "w", encoding="utf-8") as f:
        for arq in trechos_arquivos:
            f.write(f"file '{arq.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(lista_path),
        "-c", "copy",
        str(saida),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        print(f"[ERRO] Falha ao concatenar {saida.name}:")
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

    trechos_por_video = defaultdict(list)
    for linha in linhas:
        video = linha["video"].strip()
        inicio = hhmmss_to_seconds(linha["inicio"])
        fim = hhmmss_to_seconds(linha["fim"])
        trechos_por_video[video].append((inicio, fim))

    videos_disponiveis = sorted(videos_dir.rglob("*.mp4"))
    if not videos_disponiveis:
        print(f"[AVISO] Nenhum vídeo .mp4 encontrado em {videos_dir}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        for video_path in videos_disponiveis:
            nome = video_path.name
            saida = output_dir / video_path.relative_to(videos_dir)
            saida.parent.mkdir(parents=True, exist_ok=True)

            if nome not in trechos_por_video:
                print(f"[INFO] '{nome}' sem cortes no CSV — copiando sem alterações.")
                shutil.copy(video_path, saida)
                continue

            duracao = get_duration(video_path)
            mantidos = calcular_trechos_mantidos(duracao, trechos_por_video[nome])

            print(f"[INFO] '{nome}': removendo {len(trechos_por_video[nome])} trecho(s), "
                  f"mantendo {len(mantidos)} pedaço(s).")

            arquivos_pedacos = []
            ok_geral = True
            for i, (ini, fim) in enumerate(mantidos):
                pedaco = tmp_dir / f"{video_path.stem}_pedaco_{i}.mp4"
                ok = extrair_trecho(video_path, ini, fim, pedaco)
                if ok:
                    arquivos_pedacos.append(pedaco)
                else:
                    ok_geral = False
                    print(f"[ERRO] Falha ao extrair pedaço {i} de {nome}")

            if ok_geral and arquivos_pedacos:
                concatenar(arquivos_pedacos, saida, tmp_dir)
                print(f"[INFO] Salvo: {saida}")
            else:
                print(f"[ERRO] '{nome}' não pôde ser processado corretamente.")

    print(f"\n[INFO] Concluído. Vídeos limpos salvos em: {output_dir}")


if __name__ == "__main__":
    main()
