from flask import Flask, render_template, request, jsonify, url_for
import cv2
import mediapipe as mp
import face_recognition
import sqlite3
import numpy as np
import threading

app = Flask(__name__)

# Conectar ao banco de dados SQLite
conexao = sqlite3.connect("database_faces.db", check_same_thread=False)
cursor = conexao.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY, nome TEXT, codificacao BLOB)''')
conexao.commit()

# Função para adicionar novo rosto ao banco de dados
def adicionar_rosto(nome, codificacao):
    cursor.execute("INSERT INTO faces (nome, codificacao) VALUES (?, ?)", (nome, codificacao.tobytes()))
    conexao.commit()

# Função para carregar todos os rostos conhecidos do banco de dados
def carregar_rostos():
    cursor.execute("SELECT nome, codificacao FROM faces")
    rostos_conhecidos = []
    nomes_conhecidos = []
    for nome, codificacao_blob in cursor.fetchall():
        codificacao = np.frombuffer(codificacao_blob, dtype=np.float64)
        rostos_conhecidos.append(codificacao)
        nomes_conhecidos.append(nome)
    return rostos_conhecidos, nomes_conhecidos

rostos_conhecidos, nomes_conhecidos = carregar_rostos()

# Inicializar MediaPipe e a webcam
webcam = cv2.VideoCapture(0)
solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_rostos = solucao_reconhecimento_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

# Capturar e exibir a imagem da webcam
def capturar_rosto(nome):
    verificador, frame = webcam.read()
    if verificador:
        codificacao_nova = face_recognition.face_encodings(frame)
        if codificacao_nova:
            adicionar_rosto(nome, codificacao_nova[0])
            rostos_conhecidos.append(codificacao_nova[0])
            nomes_conhecidos.append(nome)

        # Exibir a imagem da câmera
        cv2.imshow('Captura de Rosto', frame)
        cv2.waitKey(2000)  
        cv2.destroyWindow('Captura de Rosto') 

    while True:
        verificador, frame = webcam.read()
        if verificador:
            cv2.imshow('Webcam', frame)

            # Adicionando a condição para fechar a câmera ao pressionar ESC
            if cv2.waitKey(5) == 27:  # Tecla ESC para sair
                break

    webcam.release()  # Liberar a câmera
    cv2.destroyAllWindows()  # Fechar todas as janelas abertas

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/registrar_rosto', methods=['POST'])
def registrar_rosto():
    nome = request.form['nome']
    # Thread para capturar o rosto
    threading.Thread(target=capturar_rosto, args=(nome,)).start()
    return jsonify({"status": "processing", "message": "Capturando rosto... Aguarde."})

@app.route('/verificar_acesso', methods=['POST'])
def verificar_acesso():
    verificador, frame = webcam.read()
    if verificador:
        lista_rostos = reconhecedor_rostos.process(frame)
        if lista_rostos.detections:
            for rosto in lista_rostos.detections:
                desenho.draw_detection(frame, rosto)

            codificacao_frame = face_recognition.face_encodings(frame)
            if codificacao_frame:
                resultados = face_recognition.compare_faces(rostos_conhecidos, codificacao_frame[0])
                if any(resultados):
                    indice = resultados.index(True)
                    # Enviar resposta JSON indicando que o acesso foi autorizado
                    return jsonify({"status": "authorized", "message": f"Acesso autorizado para {nomes_conhecidos[indice]}"})
                else:
                    return jsonify({"status": "denied", "message": "Você não está autorizado a entrar nesse site"})
    return jsonify({"status": "denied", "message": "Você não está autorizado a entrar nesse site"})

@app.route('/pagina_autorizada')
def pagina_autorizada():
    return render_template("pagina_autorizada.html")

if __name__ == '__main__':
    app.run(debug=True)
