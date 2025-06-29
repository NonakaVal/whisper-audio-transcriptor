import whisper
import time
import json
from pathlib import Path

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
        # Verifica se o arquivo de áudio existe
        if not Path(caminho_audio).exists():
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {caminho_audio}")
        
        print(f"Carregando modelo '{modelo_nome}'...")
        inicio_carregamento = time.time()
        modelo = whisper.load_model(modelo_nome)
        print(f"Modelo carregado em {time.time() - inicio_carregamento:.2f} segundos.")
        
        print(f"Transcrevendo áudio: {caminho_audio}...")
        inicio_transcricao = time.time()
        resultado = modelo.transcribe(caminho_audio)
        duracao = time.time() - inicio_transcricao
        print(f"Transcrição concluída em {duracao:.2f} segundos.")
        
        # Mostra o texto transcrito
        print("\nTexto transcrito:")
        print("-" * 50)
        print(resultado["text"])
        print("-" * 50)
        
        # Salva saídas se solicitado
        base_nome = Path(caminho_audio).stem
        if saida_txt:
            txt_path = f"{base_nome}_transcricao.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(resultado["text"])
            print(f"Transcrição salva em {txt_path}")
            
        if saida_json:
            json_path = f"{base_nome}_transcricao.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(resultado, f, ensure_ascii=False, indent=2)
            print(f"Resultado completo salvo em {json_path}")
        
        return resultado
    
    except Exception as e:
        print(f"Erro durante a transcrição: {str(e)}")
        return None

if __name__ == "__main__":
    # Configurações
    ARQUIVO_AUDIO = "Áudio do WhatsApp de 2025-06-12 à(s) 13.14.03_bb4ade67.waptt.opus"  # Altere para o caminho do seu arquivo
    MODELO = "medium"          # Pode ser "tiny", "base", "small", "medium", "large"
    
    # Executa a transcrição
    transcrever_audio(
        caminho_audio=ARQUIVO_AUDIO,
        modelo_nome=MODELO,
        saida_txt=True,
        saida_json=True
    )