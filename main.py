import whisper
import time
import json
from pathlib import Path

def listar_audios(pasta="audios"):
    """
    Lista os arquivos de áudio na pasta especificada.
    
    Parâmetros:
        pasta (str): Caminho da pasta para listar os áudios
        
    Retorna:
        list: Lista de arquivos de áudio encontrados
    """
    pasta_path = Path(pasta)
    if not pasta_path.exists():
        print(f"Pasta '{pasta}' não encontrada. Criando pasta...")
        pasta_path.mkdir()
        return []
    
    extensoes = ['.mp3', '.wav', '.ogg', '.opus', '.m4a', '.mp4', '.flac']
    arquivos = [f for f in pasta_path.iterdir() if f.suffix.lower() in extensoes]
    
    return arquivos

def selecionar_audio():
    """
    Lista os arquivos de áudio e permite seleção por número.
    
    Retorna:
        Path: Caminho completo para o arquivo selecionado
    """
    arquivos = listar_audios()
    
    if not arquivos:
        print("Nenhum arquivo de áudio encontrado na pasta 'audios'.")
        print("Por favor, coloque seus arquivos de áudio na pasta 'audios' e tente novamente.")
        return None
    
    print("\nArquivos de áudio disponíveis:")
    for i, arquivo in enumerate(arquivos, 1):
        print(f"{i}. {arquivo.name}")
    
    while True:
        try:
            escolha = input("\nDigite o número do arquivo que deseja transcrever (ou 'q' para sair): ")
            if escolha.lower() == 'q':
                return None
            
            num = int(escolha)
            if 1 <= num <= len(arquivos):
                return arquivos[num-1]
            else:
                print(f"Por favor, digite um número entre 1 e {len(arquivos)}.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número.")

def transcrever_audio(caminho_audio, modelo_nome="medium", saida_txt=False, saida_json=False):
    """
    Transcreve um arquivo de áudio usando o Whisper da OpenAI.
    
    Parâmetros:
        caminho_audio (str): Caminho para o arquivo de áudio
        modelo_nome (str): Nome do modelo a ser usado (tiny, base, small, medium, large)
        saida_txt (bool): Se True, salva a transcrição em arquivo .txt
        saida_json (bool): Se True, salva a transcrição completa em arquivo .json
        
    Retorna:
        dict: Resultado da transcrição
    """
    try:
        print(f"Carregando modelo '{modelo_nome}'...")
        inicio_carregamento = time.time()
        modelo = whisper.load_model(modelo_nome)
        print(f"Modelo carregado em {time.time() - inicio_carregamento:.2f} segundos.")
        
        print(f"Transcrevendo áudio: {caminho_audio.name}...")
        inicio_transcricao = time.time()
        resultado = modelo.transcribe(str(caminho_audio))
        duracao = time.time() - inicio_transcricao
        print(f"Transcrição concluída em {duracao:.2f} segundos.")
        
        # Mostra o texto transcrito
        print("\nTexto transcrito:")
        print("-" * 50)
        print(resultado["text"])
        print("-" * 50)
        
        # Salva saídas se solicitado
        base_nome = caminho_audio.stem
        if saida_txt:
            txt_path = caminho_audio.parent / f"{base_nome}_transcricao.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(resultado["text"])
            print(f"Transcrição salva em {txt_path}")
            
        if saida_json:
            json_path = caminho_audio.parent / f"{base_nome}_transcricao.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(resultado, f, ensure_ascii=False, indent=2)
            print(f"Resultado completo salvo em {json_path}")
        
        return resultado
    
    except Exception as e:
        print(f"Erro durante a transcrição: {str(e)}")
        return None

if __name__ == "__main__":
    # Configurações
    MODELO = "medium"  # Pode ser "tiny", "base", "small", "medium", "large"
    
    print("=== Whisper Transcriber ===")
    print("Selecione um arquivo de áudio para transcrever:\n")
    
    # Seleciona o arquivo
    arquivo_selecionado = selecionar_audio()
    
    if arquivo_selecionado:
        # Executa a transcrição
        transcrever_audio(
            caminho_audio=arquivo_selecionado,
            modelo_nome=MODELO,
            saida_txt=True,
            saida_json=True
        )