import cv2
import face_recognition
import os
import numpy as np
from tqdm import tqdm
from deepface import DeepFace
import os
import mediapipe as mp

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_images_from_folder(folder):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(image_path)
            # Obter as codificações faciais (assumindo uma face por imagem)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                face_encoding = face_encodings[0]
                # Extrair o nome do arquivo, removendo o sufixo numérico e a extensão
                name = os.path.splitext(filename)[0][:-1]
                # Adicionar a codificação e o nome às listas
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

    return known_face_encodings, known_face_names

def getVideoProperties (cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    return width, height, fps, total_frames

def getKnownFaces (face_encodings):
    # Inicializar uma lista de nomes para as faces detectadas
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconhecido"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
    return face_names
    

def getEmotionsAndFacesFromFrame (frame, emotionsDetected, face_locations, face_names):
# Iterar sobre cada face detectada pelo DeepFace
    for face in emotionsDetected:
        # Obter a caixa delimitadora da face
        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        
        # Obter a emoção dominante
        dominant_emotion = face['dominant_emotion']

        # Desenhar um retângulo ao redor da face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Escrever a emoção dominante acima da face
        cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Associar a face detectada pelo DeepFace com as faces conhecidas
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if x <= left <= x + w and y <= top <= y + h:
                # Escrever o nome abaixo da face
                cv2.putText(frame, name, (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                break
    
    # Escrever o frame processado no vídeo de saída
    return frame

def is_hand_up(landmarks, mp_pose):
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]

    left_hand_up = left_hand.y > left_eye.y
    right_hand_up = right_hand.y > right_eye.y

    return left_hand_up or right_hand_up

def detect_faces_and_emotions(video_path, output_path, known_face_encodings, known_face_names):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    width, height, fps, total_frames = getVideoProperties(cap)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()

        if not ret:
            break

        # Analisar o frame para detectar faces e expressões
        emotionsDetected = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Obter as localizações e codificações das faces conhecidas no frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        results = pose.process(rgb_frame)

        # Desenhar as anotações da pose no frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        face_names = getKnownFaces(face_encodings)
        frame = getEmotionsAndFacesFromFrame(frame,emotionsDetected, face_locations, face_names)
        
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    image_folder = 'images'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_path = os.path.join(script_dir, 'Unlocking Facial Recognition_ Diverse Activities Analysis.mp4')  
    output_video_path = os.path.join(script_dir, 'output_video_recognize.mp4')  

    known_face_encodings, known_face_names = load_images_from_folder(image_folder)
    detect_faces_and_emotions(input_video_path, output_video_path, known_face_encodings, known_face_names)


 