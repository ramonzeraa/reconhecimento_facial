import cv2
import mediapipe as mp
import face_recognition
import tkinter as tk
from tkinter import Label, Button, simpledialog, messagebox, Frame
import sqlite3
import numpy as np

# Conectar ao banco de dados SQLite
conexao = sqlite3.connect("database_faces.db")
cursor = conexao.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY, nome TEXT, codificacao BLOB)''')
conexao.commit()

# Função para adicionar novo rosto ao banco de dados
def adicionar_rosto(nome, codificacao):
    cursor.execute("INSERT INTO faces (nome, codificacao) VALUES (?, ?)", (nome, codificacao.tobytes()))
    conexao.commit()
    messagebox.showinfo("Sucesso", f"Rosto de {nome} adicionado com sucesso.")

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

# Carregar rostos conhecidos no início do programa
rostos_conhecidos, nomes_conhecidos = carregar_rostos()

# Configurar a janela do tkinter
root = tk.Tk()
root.title("Sistema de Acesso Facial")
root.geometry("500x300")
root.configure(bg="#2b2b2b")

# Função para atualizar a interface gráfica
def update_interface(texto):
    label_resultado.config(text=texto, fg="green" if "autorizado" in texto else "red")
    root.update()

# Função para registrar um novo rosto diretamente no banco de dados
def registrar_rosto():
    update_interface("Capturando novo rosto...")
    verificador, frame = webcam.read()
    if verificador:
        codificacao_nova = face_recognition.face_encodings(frame)
        if codificacao_nova:
            nome = simpledialog.askstring("Registro de Rosto", "Digite o nome da pessoa:")
            if nome:
                adicionar_rosto(nome, codificacao_nova[0])
                rostos_conhecidos.append(codificacao_nova[0])
                nomes_conhecidos.append(nome)
                update_interface(f"Rosto de {nome} registrado com sucesso.")
        else:
            update_interface("Nenhum rosto detectado. Tente novamente.")

# Inicializar MediaPipe e a webcam
webcam = cv2.VideoCapture(0)
solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_rostos = solucao_reconhecimento_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

# Configuração dos elementos da interface gráfica
frame_titulo = Frame(root, bg="#2b2b2b")
frame_titulo.pack(pady=20)

label_titulo = Label(frame_titulo, text="Sistema de Acesso Facial", font=("Arial", 18, "bold"), bg="#2b2b2b", fg="#ffffff")
label_titulo.pack()

frame_corpo = Frame(root, bg="#2b2b2b")
frame_corpo.pack(pady=10)

label_resultado = Label(frame_corpo, text="Aguardando ação do usuário...", font=("Arial", 12), bg="#2b2b2b", fg="#ffffff")
label_resultado.pack(pady=10)

botao_registrar = Button(root, text="Registrar Novo Rosto", command=registrar_rosto, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), width=20)
botao_registrar.pack(pady=15)

# Loop de reconhecimento facial
while True:
    verificador, frame = webcam.read()
    if not verificador:
        break

    lista_rostos = reconhecedor_rostos.process(frame)
    if lista_rostos.detections:
        for rosto in lista_rostos.detections:
            desenho.draw_detection(frame, rosto)

        codificacao_frame = face_recognition.face_encodings(frame)
        if codificacao_frame:
            resultados = face_recognition.compare_faces(rostos_conhecidos, codificacao_frame[0])
            if any(resultados):
                indice = resultados.index(True)
                update_interface(f"Acesso autorizado para {nomes_conhecidos[indice]}")
            else:
                update_interface("Acesso negado")

    cv2.imshow('Leitura Facial', frame)

    if cv2.waitKey(5) == 27:  # Tecla ESC para sair
        break

webcam.release()
cv2.destroyAllWindows()
root.destroy()
conexao.close()