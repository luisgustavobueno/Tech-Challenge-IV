import cv2
import mediapipe as mp
import face_recognition
from deepface import DeepFace
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Força uso da CPU
SHOW_VIDEO = False                          # Modifique para True para exibir o vídeo processado
FACE_INFO = {}                              # Dicionário para armazenar informações dos rostos detectados
FRAME_SKIP = 1                              # Modifique para aumentar a velocidade de processamento, porém diminui a precisão
EMOTION_THRESHOLD = 95                      # Modifique para alterar a sensibilidade da detecção de emoções
ANOMALY_THRESHOLD = 0.2                     # Modifique para alterar a sensibilidade da detecção de anomalias
HEAD_MOVEMENT_THRESHOLD = 0.15              # Modifique para alterar a sensibilidade da detecção de movimentos de cabeça
last_dominant_emotion = None                # Variável global para armazenar a última emoção detectada


class PoseDetector:
    '''Classe para detecção de poses humanas'''
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        '''Função para processar um frame e detectar poses humanas'''
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb_frame)

    def draw_landmarks(self, frame, landmarks):
        '''Função para desenhar os landmarks das poses no frame'''
        self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_pose.POSE_CONNECTIONS)

    def is_hand_up(self, landmarks):
        '''Função para verificar se a mão está levantada'''
        left_hand = landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX.value]
        right_hand = landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX.value]

        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        left_hand_up = left_hand.y < left_shoulder.y
        right_hand_up = right_hand.y < right_shoulder.y


        return left_hand_up or right_hand_up

    def is_head_moving(self, prev_nose, curr_nose):
        '''Função para verificar se a cabeça está se movendo'''

        dx = abs(curr_nose.x - prev_nose.x)
        dy = abs(curr_nose.y - prev_nose.y)

        return dx > HEAD_MOVEMENT_THRESHOLD or dy > HEAD_MOVEMENT_THRESHOLD

    def detect_movements(self, landmarks, prev_nose):
        '''Função para detectar movimentos de poses humanas'''
        movements = {}

        movements["hand_up"] = self.is_hand_up(landmarks)
        movements["head_moving"] = self.is_head_moving(
            prev_nose, landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        )

        return movements


class FaceRecognition:
    '''Classe para reconhecimento de faces e detecção de emoções'''
    def __init__(self, known_faces_dir="images"):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces(known_faces_dir)

    def load_known_faces(self, directory):
        '''Função para carregar imagens de rostos conhecidos'''
        if not os.path.exists(directory):
            print(f"Diretório {directory} não encontrado!")
            return

        for filename in os.listdir(directory):
            img_path = os.path.join(directory, filename)
            img = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(img)
            if encoding:
                self.known_face_encodings.append(encoding[0])
                self.known_face_names.append(
                    os.path.splitext(filename)[0][:-1]
                )  # Nome sem extensão

    def recognize_faces(self, frame):
        '''Função para reconhecer rostos em um frame'''
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_info = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding
            )
            name = "Desconhecido"
            if True in matches:
                best_match_index = matches.index(True)
                name = self.known_face_names[best_match_index]

            face_info.append((name, face_location))
        return face_info

    def detect_emotion(self, frame, face_location):
        '''Função para detectar emoções em um rosto'''
        top, right, bottom, left = face_location
        face_crop = frame[top:bottom, left:right]
        try:
            analysis = DeepFace.analyze(
                face_crop, actions=["emotion"], enforce_detection=False
            )
            dominant_emotion = analysis[0]["dominant_emotion"]
            emotion_score = analysis[0]["emotion"][dominant_emotion]
            global last_dominant_emotion

            if emotion_score >= EMOTION_THRESHOLD:
                last_dominant_emotion = dominant_emotion
                return dominant_emotion
            else:
                return last_dominant_emotion
        except Exception:
            return "Desconhecido"


def display_info(frame, face_info, face_recognizer):
    '''Função para exibir informações dos rostos detectados na tela'''
    for name, face_location in face_info:
        top, right, bottom, left = face_location
        emotion = face_recognizer.detect_emotion(frame, face_location)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{name} ({emotion})",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        format_faces_output(name, emotion)


def format_faces_output(name, emotion):
    '''Função para formatar as informações dos rostos detectados'''
    if name in FACE_INFO:
        if emotion not in FACE_INFO[name]:
            FACE_INFO[name].append(emotion)
    else:
        FACE_INFO[name] = [emotion]


def display_movement_text(frame, movements, counts, movement_flags):
    '''Função para exibir informações de movimentos na tela'''
    y_offset = 30
    for movement, detected in movements.items():
        if detected and not movement_flags[movement]:
            counts[movement] += 1
            movement_flags[movement] = True
        elif not detected:
            movement_flags[movement] = False

        text = f'{movement.replace("_", " ").capitalize()}: {counts[movement]}'
        cv2.putText(
            frame,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y_offset += 30


def detect_anomalies(landmarks, prev_nose):
    '''Função para detectar anomalias em poses humanas'''
    if prev_nose is None:
        return False

    curr_nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
    dx = abs(curr_nose.x - prev_nose.x)
    dy = abs(curr_nose.y - prev_nose.y)

    return dx > ANOMALY_THRESHOLD or dy > ANOMALY_THRESHOLD


def generate_summary(total_frames, anomaly_count, movement_counts):
    '''Função para gerar um resumo do vídeo processado'''
    summary = {
        "Total de frames analisados": total_frames,
        "Número de anomalias detectadas": anomaly_count,
        "Movimentos detectados": movement_counts,
        "Rostos reconhecidos": FACE_INFO,
    }
    return summary


def save_summary_to_file(summary, filename="summary.txt"):
    '''Função para salvar o resumo em um arquivo de texto'''
    with open(filename, "w", encoding="utf-8") as file:
        for key, value in summary.items():
            file.write(f"{key}: {value}\n")
    print(f"Resumo salvo em {filename}")


def process_video(input_path, output_path):
    '''Função para processar um vídeo e detectar anomalias'''
    pose_detector = PoseDetector()
    face_recognizer = FaceRecognition()
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    movement_counts = {"hand_up": 0, "head_moving": 0}
    movement_flags = {"hand_up": False, "head_moving": False}
    prev_nose = None
    anomaly_count = 0
    frame_count = 0

    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            out.write(frame)
            continue

        results = pose_detector.process_frame(frame)

        if results.pose_landmarks:
            pose_detector.draw_landmarks(frame, results.pose_landmarks)
            landmarks = results.pose_landmarks.landmark

            if prev_nose is None:
                prev_nose = landmarks[pose_detector.mp_pose.PoseLandmark.NOSE.value]
                continue

            movements = pose_detector.detect_movements(landmarks, prev_nose)
            if detect_anomalies(landmarks, prev_nose):
                anomaly_count += 1

            prev_nose = landmarks[pose_detector.mp_pose.PoseLandmark.NOSE.value]

            display_movement_text(frame, movements, movement_counts, movement_flags)

        face_info = face_recognizer.recognize_faces(frame)
        display_info(frame, face_info, face_recognizer)

        out.write(frame)
        if SHOW_VIDEO:
            cv2.imshow("Video Processed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Gerar resumo
    summary = generate_summary(total_frames, anomaly_count, movement_counts)
    print("Resumo do Vídeo:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Salvar resumo em arquivo
    save_summary_to_file(summary)


if __name__ == "__main__":
    input_video_path = "Unlocking Facial Recognition_ Diverse Activities Analysis.mp4"
    output_video_path = "output_video.mp4"
    process_video(input_video_path, output_video_path)