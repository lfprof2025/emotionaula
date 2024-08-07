# -*- coding: utf-8 -*-
from fer import FER
import matplotlib.pyplot as plt
import numpy as np

# Carregar a imagem
test_image_one = plt.imread("imagem01.jpg")

# Inicializar o detector de emoções
emo_detector = FER(mtcnn=True)
captured_emotions = emo_detector.detect_emotions(test_image_one)

if captured_emotions:
    # Para simplificar, vamos considerar apenas o primeiro rosto detectado
    emotions = captured_emotions[0]["emotions"]

    # Filtrar a emoção "neutro"
    emotions = {k: v for k, v in emotions.items() if k != "neutral"}

    # Preparar dados para o gráfico
    emotions_names = list(emotions.keys())
    emotions_values = list(emotions.values())

    # Criar o gráfico de barras
    fig, ax = plt.subplots()
    ax.bar(emotions_names, emotions_values)
    ax.set_ylabel('Scores')
    ax.set_title('Emotions detected')
    plt.xticks(rotation=45)  # Rotação dos nomes das emoções para melhor visualização

    # Mostrar o gráfico
    plt.show()
else:
    print("No emotions detected.")

# Mostrar a imagem com os pontos analisados
plt.imshow(test_image_one)

# Desenhar os pontos no gráfico
if captured_emotions:
    for emotion_data in captured_emotions:
        face_rectangle = emotion_data["box"]
        x, y, w, h = face_rectangle
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))

        # Adicionar pontos nas regiões aproximadas do rosto
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        left_eye = (x + w // 4, y + h // 4)
        right_eye = (x + 3 * w // 4, y + h // 4)
        nose = (face_center_x, y + h // 2)
        mouth = (face_center_x, y + 3 * h // 4)

        points = [left_eye, right_eye, nose, mouth]
        for point in points:
            plt.plot(point[0], point[1], 'bo')  # Ponto azul para regiões específicas

    # Mostrar a imagem com o rosto detectado e pontos das microexpressões
    plt.show()

# Obter a emoção dominante
dominant_emotion, emotion_score = emo_detector.top_emotion(test_image_one)
print(dominant_emotion, emotion_score)
