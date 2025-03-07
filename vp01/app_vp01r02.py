#########################################################################################
# Detecção com Rastreamento de Objetos no Yolo 11 - Monitora Área de Interesse em Vídeo
#########################################################################################


"""
Faz a carga das libs essenciais para a aplicação - libs em requirements.txt
"""
import cv2
import torch
import threading
import pygame
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

#
# Inicia o myxer do pygames para gerar sons de alerta ...
#

pygame.mixer.init()

def tocar_alarme(arquivo_som="alerta.wav"):
    try:
        sound = pygame.mixer.Sound(arquivo_som)
        sound.play()
    except Exception as e:
        print(f"Erro ao tocar o alarme: {e}")

#
# Função Principal para Processamento do Vídeo do Supermercado ...
#
def processar_video(video_input, video_output, modelo_path, som_alerta="alerta.wav"):
    
    """
    Processa o vídeo na GPU (se disponível) ou CPU
    Carrega o modelo no dispositivo escolhido
    Inicia o rastreador de objetos com parametrização
    Captura o vídeo de entrada em video_input
    Inicia as configurações do vídeo
    Abre vídeo de saída para escrita com parâmetros definidos
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    model = YOLO(modelo_path).to(device)
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_output, fourcc, fps, (frame_width, frame_height))

    """
    Define a 'área de interesse' para monitorar a entrada de pessoas nessa 'zona' do vídeo
    """
    saida_x1, saida_y1 = int(frame_width * 0.4), int(frame_height * 0.4)
    saida_x2, saida_y2 = int(frame_width * 0.95), int(frame_height * 0.95)

    """
    Armazena pessoas que entraram na zona de interesse
    """
    pessoas_na_zona = set()

    """
    Inicia tratamento frame a frame do vídeo ...
    Loop finaliza quando o arquivo de vídeo for fechado
    Faz cópia de cada frame para tratamento, mas mantém original
    Plota "Zona de Saída" na 'caixa' com coordenadas definidas
    Carrega modelo para inferência no 'frame' e desliga 'verbose'
    Detecta apenas objetos pessoas (classe 0 no coco dataset do pré-treinamento)
    Cria lista para guardar objetos detectados e inicia detecção
    """
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 
        frame_draw = frame.copy()
        cv2.rectangle(frame_draw, (saida_x1, saida_y1), (saida_x2, saida_y2), (0, 0, 255), 3)
        cv2.putText(frame_draw, "ZONA DE SAIDA", (saida_x1, saida_y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)


        results = model(frame, verbose=False)

        detections = []
        # Processa cada resultado e extrai as detecções de pessoas (classe 0 no dataset COCO)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = box.conf[0].cpu().item()
                cls = int(box.cls[0].cpu().item())
                if cls == 0 and conf > 0.4:
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls, frame))

        """
        Invoca o método tracker do deepsort para rastrear quadro a quadro dos 
        objetos detectados (na lista).

        Desenha 'bound boxes' e plota IDs dos objetos rastreados no vídeo
        """
        track_objects = tracker.update_tracks(detections, frame=frame)
        for track in track_objects:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            label = f"Pessoa ID {track_id}"
            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_draw, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

            """
            Centro do bound box do objeto detectado e rastreado é calculado
            Se o 'centro' calculado está na área de interesse dispare alerta
            Som de alerta é acionado em thread separado (evita bloqueio de processamento)
            """
            centro_x = x1 + (x2 - x1) // 2
            centro_y = y1 + (y2 - y1) // 2
            if (saida_x1 < centro_x < saida_x2) and (saida_y1 < centro_y < saida_y2):
                if track_id not in pessoas_na_zona:
                    print(f"⚠️ ALERTA! Pessoa ID {track_id} entrou na zona de saída!")
                    pessoas_na_zona.add(track_id)
                    threading.Thread(target=tocar_alarme, args=(som_alerta,), daemon=True).start()

        cv2.imshow("Monitoramento - Rastreamento de Pessoas", frame_draw)
        out.write(frame_draw)

        if cv2.waitKey(1) & 0xFF == ord('q'): #se você pressionar 'Q' ou 'q', interrompe o vídeo
            break
    """
    Libera recursos capturados e desaloca controle de janela
    """
    cap.release()
    out.release()
    cv2.destroyAllWindows()

"""
Carga dos Arquivos e Escolha do Modelo Pré-Treinado - Executar Programa
"""
if __name__ == "__main__":
    video_input = "supermercado.mp4"
    video_output = "output_supermercado.mp4"
    modelo_path = "yolo11n.pt"
    som_alerta = "alerta.wav"  # Novo arquivo de áudio em WAV

    processar_video(video_input, video_output, modelo_path, som_alerta)
