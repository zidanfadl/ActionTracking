import cv2
import numpy as np
import time
import torch
import serial
from pathlib import Path
from boxmot import StrongSort
from ultralytics import YOLO

font = cv2.FONT_HERSHEY_SIMPLEX
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO('yolov8s.pt')
tracker = StrongSort(
        reid_weights=Path('osnet_x0_25_msmt17.pt'),
        device=device,
        half=False
    )

def sendData(targetPos):
    ser = serial.Serial(port='COM4', timeout=1,baudrate=9600)
    coordinate = f'{targetPos[0]},{targetPos[1]}\r'
    ser.write(coordinate.encode())
    print("Sent:",coordinate)
    ser.close()

def targetTracking(coor_dets):
    targetPos = None
    for coor_det in coor_dets:
        x, y, track_id = coor_det
        if track_id == 1:
            targetPos = (x, y)
            sendData(targetPos)
            break

def draw_bboxes(annotation, frame, thickness=2):
    coor_dets = []
    for person in annotation:
        x1, y1, x2, y2, track_id, conf, cls,_ = person
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(thickness))
        cv2.putText(frame, f'ID: {int(track_id)}', (x1, y1-40), font, 0.5, (0, 255, 0), thickness)
        cv2.putText(frame, f'Conf: {conf:.2f}', (x1, y1-10), font, 0.5, (0, 255, 0), thickness)
        
        # Draw center coordinate
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(frame, f'({center_x}, {center_y})', (center_x, center_y - 10), font, 0.5, (0, 0, 255), thickness)
        coor_dets.append([center_x, center_y, track_id])

    return frame, coor_dets

def draw_activity_area(frame, thickness=1):
    height, width, _ = frame.shape
    margin = 70
    top_left = (margin, margin)
    bottom_right = (width - margin, height - margin)
    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), thickness)
    return frame

def main():
    vid = "test.mp4"
    cap = cv2.VideoCapture(3)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)       #webcam
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv2.resize(frame, (720, 720))       #video file

        frame = draw_activity_area(frame)
        start = time.time()
        results = model(frame, 
                        classes=[0]
                        )

        conf_thres = 0.5
        dets = []
        for box in results[0].boxes.cpu().numpy():
            if box.conf[0].astype(float).round(2) > conf_thres:
                x1, y1, x2, y2 = box.xyxy[0].astype(float).round(2)
                conf = box.conf[0].astype(float).round(2)
                cls = box.cls[0].astype(int)
                print(x1, y1, x2, y2, "conf:",conf, "cls:", cls)
                dets.append([x1, y1, x2, y2, conf, cls])

        dets = np.array(dets)
        res = tracker.update(dets, frame)

        frame, coor_dets = draw_bboxes(res, frame)
        targetTracking(coor_dets)
        print(frame.shape)

        end = time.time()
        cv2.putText(frame, f'FPS: {1/(end-start):.2f}', (10, 50), font, 1.5, (0, 255, 255), 4)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
