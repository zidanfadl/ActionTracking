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

def draw_boxes(annotation, frame, thickness=4):
    for person in annotation:
        x1, y1, x2, y2, track_id, conf, cls,_ = person
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(thickness))
        cv2.putText(frame, f'ID: {int(track_id)}', (x1, y1-40), font, 1, (0, 255, 0), thickness)
        cv2.putText(frame, f'Conf: {conf:.2f}', (x1, y1-10), font, 1, (0, 255, 0), thickness)
    
    return frame


def main():
    vid = "test.mp4"
    cap = cv2.VideoCapture(vid)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
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
        print('dets:', dets)
        res = tracker.update(dets, frame)
        print('res:', res)
        frame = draw_boxes(res, frame)

        end = time.time()
        cv2.putText(frame, f'FPS: {1/(end-start):.2f}', (10, 50), font, 1.5, (0, 255, 255), 4)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
