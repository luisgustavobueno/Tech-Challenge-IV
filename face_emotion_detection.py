import cv2
import mediapipe as mp
import face_recognition
from deepface import DeepFace
import numpy as np
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Força uso da CPU
SHOW_VIDEO = False
FACE_INFO = {}


class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb_frame)

    def draw_landmarks(self, frame, landmarks):
        self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_pose.POSE_CONNECTIONS)

    def is_hand_up(self, landmarks):
        left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE.value]
        left_hand = landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX.value]
        return left_hand.y < left_eye.y

    def is_head_moving(self, prev_nose, curr_nose, threshold=0.1):
        return (
            abs(curr_nose.x - prev_nose.x) > threshold
            or abs(curr_nose.y - prev_nose.y) > threshold
        )

    def detect_movements(self, landmarks, prev_nose):
        movements = {}

        movements["hand_up"] = self.is_hand_up(landmarks)
        movements["head_moving"] = self.is_head_moving(
            prev_nose, landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        )

        return movements


class FaceRecognition:
    def __init__(self, known_faces_dir="images"):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces(known_faces_dir)

    def load_known_faces(self, directory):
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
        top, right, bottom, left = face_location
        face_crop = frame[top:bottom, left:right]
        try:
            analysis = DeepFace.analyze(
                face_crop, actions=["emotion"], enforce_detection=False
            )
            return analysis[0]["dominant_emotion"]
        except Exception:
            return "Desconhecido"


def display_info(frame, face_info, face_recognizer):
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
    if name in FACE_INFO:
        if emotion not in FACE_INFO[name]:
            FACE_INFO[name].append(emotion)
    else:
        FACE_INFO[name] = [emotion]


def display_movement_text(frame, movements, counts, movement_flags):
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


def detect_anomalies(landmarks, prev_nose, anomaly_threshold=0.2):
    if prev_nose is None:
        return False

    curr_nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
    dx = abs(curr_nose.x - prev_nose.x)
    dy = abs(curr_nose.y - prev_nose.y)

    return dx > anomaly_threshold or dy > anomaly_threshold


def generate_summary(total_frames, anomaly_count, movement_counts, face_info):
    summary = {
        "Total de frames analisados": total_frames,
        "Número de anomalias detectadas": anomaly_count,
        "Movimentos detectados": movement_counts,
        "Rostos reconhecidos": FACE_INFO,
    }
    return summary


def save_summary_to_file(summary, filename="summary.txt"):
    with open(filename, "w", encoding="utf-8") as file:
        for key, value in summary.items():
            file.write(f"{key}: {value}\n")
    print(f"Resumo salvo em {filename}")


def process_video(input_path, output_path):
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

    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

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
    summary = generate_summary(total_frames, anomaly_count, movement_counts, face_info)
    print("Resumo do Vídeo:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Salvar resumo em arquivo
    save_summary_to_file(summary)


if __name__ == "__main__":
    input_video_path = "Unlocking Facial Recognition_ Diverse Activities Analysis.mp4"
    output_video_path = "output_video.mp4"
    process_video(input_video_path, output_video_path)