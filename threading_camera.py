import threading
import cv2
import time
from collections import deque
from ultralytics import YOLO
from boxmot import StrongSort
from pathlib import Path
import numpy as np
import torch

# Configuration
model= YOLO('yolov8n.pt')
tracker = StrongSort(
        reid_weights=Path('osnet_x0_25_msmt17.pt'),
        device= torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        half=False
    )
batch_size = 4  # Smallest batch size for your model
video = "50d_1s3.mp4"
camera_index = video  # Adjust based on your camera or video
fps_update_interval = 0.1  # Seconds between FPS updates

#annotation
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
FONTCOLOR = (0, 255, 0)
LINETYPE = 2

# Shared resources
buffer = deque(maxlen=batch_size)
buffer_lock = threading.Lock()
frame_available = threading.Condition(buffer_lock)
running = True  # Global flag for controlling threads


def capture_frames():
    """Continuously capture frames from the camera."""
    global running
    cap = cv2.VideoCapture(camera_index)
    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            with buffer_lock:
                buffer.append(frame)
                frame_available.notify_all()
    finally:
        cap.release()

def process_batches():
    """Process batches and handle FPS calculation."""
    global running
    start_time = time.time()
    frames_processed = 0
    
    while running:
        with buffer_lock:
            # Wait for enough frames or exit signal
            while len(buffer) < batch_size and running:
                frame_available.wait(timeout=0.1)
            
            if not running:
                break
            
            current_batch = list(buffer)
            frames_processed += len(current_batch)
        
        # Calculate smoothed FPS
        elapsed_time = time.time() - start_time
        fps = frames_processed / elapsed_time if elapsed_time > 0 else 0
        
        # Process and display with FPS
        if not process_and_display(current_batch, fps):
            break

def process_and_display(batch, fps):
    """Display frames with FPS and handle exit key."""
    global running

    results = model(batch, classes=[0])
    print("how many: " + str(len(results)))
    conf_thres = 0
    
    framesDet, humanDet, targetFrameList = [], [], []
    for frame_idx, (frame, result) in enumerate(zip(batch, results)):
        dets, detXYXY = [], []
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            conf = box.conf[0].astype(float).round(2)
            if conf > conf_thres:
                x1, y1, x2, y2 = box.xyxy[0].astype(float).round(2)
                cls = box.cls[0].astype(int)
                # detXYXY.append([x1, y1, x2, y2])
                dets.append([x1, y1, x2, y2, conf, cls])

        #tracker
        target = tracker.update(np.array(dets), np.array(frame))
        targetList = [int(x[4]) for x in target]
        targetFrameList.append(targetList if targetList else [])

        print(f"Frame {frame_idx} detections: {dets}")
        framesDet.append(dets)

        detXYXY = [det[:4] for det in dets]
        humanDet.append(np.array(detXYXY, dtype=np.float32) if detXYXY else np.empty((0, 4), dtype=np.float32))

    # check
    framesDet = np.array(framesDet, dtype=object)
    frames = np.array(batch)
    print(f"framesDet: {framesDet}")
    print(f"humanDet: {humanDet}")
    print(f"targetList: {targetFrameList}")
    
    # res = tracker.update(framesDet, frames)
    

    for frame, detection, targetList in zip(batch, framesDet, targetFrameList):
        # Add FPS text to frame
        # res = tracker.update(dets, frame)
        if len(detection) > 0:
            for person, id in zip(detection, targetList):
                x1, y1, x2, y2, conf, _ = person
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(f"frame: {frame.shape}")
                personID = id
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                text1 = f"Confidence: {conf}"
                text2 = f"ID: {personID}"
                location1 = (x1+5, y1+15)
                location2 = (x1+5, y1+30)

                # Draw shadow for text1
                shadow_color = (0, 0, 0)
                cv2.putText(frame, text1, (location1[0] + 1, location1[1] + 1), FONT, FONT_SCALE, shadow_color, FONT_THICKNESS, LINETYPE)
                # Draw text1
                cv2.putText(frame, text1, location1, FONT, FONT_SCALE, FONTCOLOR, FONT_THICKNESS, LINETYPE)

                # Draw shadow for text2
                cv2.putText(frame, text2, (location2[0] + 1, location2[1] + 1), FONT, FONT_SCALE, shadow_color, FONT_THICKNESS, LINETYPE)
                # Draw text2
                cv2.putText(frame, text2, location2, FONT, FONT_SCALE, FONTCOLOR, FONT_THICKNESS, LINETYPE)


        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Live Inference', frame)
        
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            return False
    return True

if __name__ == "__main__":
    # Start threads
    capture_thread = threading.Thread(target=capture_frames)
    process_thread = threading.Thread(target=process_batches)

    try:
        capture_thread.start()
        process_thread.start()
        
        # Keep main thread active to catch KeyboardInterrupt
        while capture_thread.is_alive() or process_thread.is_alive():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        running = False
        with buffer_lock:
            frame_available.notify_all()
        
        capture_thread.join()
        process_thread.join()
        cv2.destroyAllWindows()
        print("Program terminated successfully.")