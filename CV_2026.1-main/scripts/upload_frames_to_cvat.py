import os
import subprocess

# --- CONFIGURAÇÕES ---
# Coloque seu usuário e senha do app.cvat.ai

CVAT_USER = ""
CVAT_PASS = ""
PROJECT_ID = "" # Ex: "https://app.cvat.ai/projects/388009?page=1&pageSize=10" é de id 388009

# O diretório raiz onde o trator (os.walk) vai começar a cavar
# Os frames precisam estar numa pasta de Frames, que deve estar organizada por jogo e depois por vídeo
BASE_DIR = "/home/victor/Documentos/Basquete/Jogadas (Frames)"

def main():
    # Pega apenas as pastas principais (Ex: Botafogo_Cabo_Branco, Rop_Maniacos)
    pastas_jogos = [p for p in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, p))]

    # pastas_jogos.remove('Botafogo_Cabo_Branco')
    # pastas_jogos.remove('Rop_Maniacos')
    # pastas_jogos.remove('Fenix_Manaira')


    for jogo in pastas_jogos:
        caminho_jogo = os.path.join(BASE_DIR, jogo)
        nome_da_task = f"Task_{jogo}"
        
        arquivos_imagens = []
        
        # O trator (os.walk) agora varre todas as subpastas DESTE jogo específico
        for root, dirs, files in os.walk(caminho_jogo):
            frames = [
                os.path.join(root, f) 
                for f in files 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            # Adiciona os frames encontrados na lista geral do jogo
            arquivos_imagens.extend(frames)
            
        # Se não achou nenhuma imagem nas subpastas, pula
        if not arquivos_imagens:
            print(f"⚠️ Nenhuma imagem encontrada no jogo {jogo}. Pulando...")
            continue
            
        # Ordena TUDO do jogo. Como o timestamp tem '00s', '01s', 
        # os vídeos vão ficar agrupados em blocos sequenciais automaticamente!
        arquivos_imagens.sort()
        
        print(f"\n🚀 Jogo: {jogo}")
        print(f"Criando a task: {nome_da_task} aglutinando {len(arquivos_imagens)} frames...")
        
        # Monta o comando do CVAT
        comando = [
            "cvat-cli", 
            "--server-host", "app.cvat.ai",
            "--auth", f"{CVAT_USER}:{CVAT_PASS}",
            "task", "create", nome_da_task,
            "--project_id", PROJECT_ID,
            "local"
        ]
        
        # Passa o batalhão de imagens de uma vez só
        comando.extend(arquivos_imagens)
        
        # Executa!
        try:
            subprocess.run(comando, check=True)
            print(f"✅ Task {nome_da_task} finalizada com sucesso!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro ao enviar a task {nome_da_task}. Erro: {e}")

if __name__ == "__main__":
    main()