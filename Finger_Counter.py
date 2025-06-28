import cv2
import mediapipe as mp
import numpy as np
import socket
import struct

# TCP server setup
HOST = '0.0.0.0'
PORT = 10000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"Server listening on {HOST}:{PORT}...")
conn, addr = server_socket.accept()
print(f"Client connected from {addr}")

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam!")
    exit()

print("Webcam opened successfully. Starting video stream...")

def count_fingers(landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 6, 10, 14, 18]
    count = 0

    for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
        if landmarks[tip].y < landmarks[pip].y:
            count += 1

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    wrist = landmarks[0]
    if (thumb_tip.x - wrist.x) > 0.1:
        count += 1

    return count

try:
    frame_count = 0  # Theo dõi số frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to read frame at count {frame_count}")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                num_fingers = count_fingers(hand_landmarks.landmark)
                cv2.putText(frame, f'Fingers: {num_fingers}', (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                with open("gesture1.txt", "w") as f:
                    f.write(str(num_fingers))

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            print(f"Error: Failed to encode frame at count {frame_count}")
            continue
        
        data = jpeg.tobytes()
        try:
            conn.sendall(struct.pack(">L", len(data)) + data)
            print(f"Frame {frame_count} sent successfully")
        except BrokenPipeError:
            print(f"Client disconnected at frame {frame_count}")
            break
        except Exception as e:
            print(f"Error sending data at frame {frame_count}: {e}")
            break

        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1

finally:
    print("Cleaning up resources...")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    conn.close()
    server_socket.close()