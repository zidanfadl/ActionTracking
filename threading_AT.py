# Import
import threading                    # Threading Essential
import cv2
import time
from collections import deque

from ultralytics import YOLO        # YOLO and Tracker
from boxmot import StrongSort
from pathlib import Path
import numpy as np
import torch

import mmcv                         # OpenMMLab
import mmengine
from mmengine import DictAction
from mmaction.apis import (detection_inference, inference_recognizer,
                           init_recognizer, pose_inference)
from mmaction.registry import VISUALIZERS
from mmaction.utils import frame_extract
import copy as cp
import moviepy.editor as mpy

# Configuration
model= YOLO('yolov8n.pt')           # YOLO Model
tracker = StrongSort(
        reid_weights=Path('osnet_x0_25_msmt17.pt'),
        device= torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        half=False
    )
batch_size = 4  # Smallest batch size for your model
camera_index = 0  # Adjust based on your camera
fps_update_interval = 0.1  # Seconds between FPS updates

# annotation
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

# Arguement Parser
class arg_parser:
    def __init__(self):
        # self.video = '/home/ciis/Desktop/shitass.mp4'
        # self.out_filename = '/home/ciis/Desktop/shitass_out2.mp4'
        # self.det_config = 'mmaction2/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py'
        # self.det_checkpoint = 'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
        # self.det_score_thr = 0.9
        self.pose_config = 'mmaction2/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py'
        self.pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
        self.skeleton_config = 'configs/skeleton/posec3d/ciis_10.py'
        self.skeleton_stdet_checkpoint = 'work_dirs/ciis_10_best-550/best_acc_top1_epoch_550.pth'
        self.action_score_thr = 0.75
        self.label_map_stdet = 'data/skeleton/ciis_label_map.txt'
        self.predict_stepsize = 2
        
        self.output_fps = 4
        self.device = torch.device(0)
        self.output_stepsize = 1
        self.cfg_options={}

#####################################
def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


PLATEBLUE = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
PLATEBLUE = PLATEBLUE.split('-')
PLATEBLUE = [hex2color(h) for h in PLATEBLUE]

#====================================
# MMaction Visualize Function
def visualize_frames_with_annotations(  args,
                                        frames,
                                        annotations,
                                        pose_data_samples,
                                        action_result,
                                        plate=PLATEBLUE,
                                        max_labels_per_box=5):
    """
    Visualizes frames with predicted annotations, including pose data and action detection results.

    Args:
        frames (list[np.ndarray]): List of frames to visualize. The number of frames should be divisible by the number of annotations.
        annotations (list[list[tuple]]): Predicted spatio-temporal detection results.
        pose_data_samples (list[list[PoseDataSample]]): Pose estimation results for each frame.
        action_result (str): Predicted action recognition result.
        plate (str): Color plate used for visualization. Default: PLATEBLUE.
        max_labels_per_box (int): Maximum number of labels to display for each bounding box. Default: 5.

    Returns:
        list[np.ndarray]: List of visualized frames.
    """

    # Ensure the number of labels does not exceed the available colors in the plate
    assert max_labels_per_box + 1 <= len(plate)

    # Create a deep copy of frames and convert them to RGB format
    frames_copy = cp.deepcopy(frames)
    frames_copy = [mmcv.imconvert(frame, 'bgr', 'rgb') for frame in frames_copy]

    num_frames = len(frames_copy)
    num_annotations = len(annotations) if annotations else 1

    # Ensure the number of frames is divisible by the number of annotations
    assert num_frames % num_annotations == 0, "Number of frames must be divisible by the number of annotations."

    frames_per_annotation = num_frames // num_annotations
    height, width, _ = frames[0].shape
    scale_ratio = np.array([width, height, width, height])  # Scale ratios for bounding box coordinates

    # Add pose estimation results to the frames
    if pose_data_samples:
        pose_config = mmengine.Config.fromfile(args.pose_config)
        visualizer = VISUALIZERS.build(
            pose_config.visualizer | {'line_width': 5, 'bbox_color': (101, 193, 255), 'radius': 8}
        )
        visualizer.set_dataset_meta(pose_data_samples[0].dataset_meta)

        for i, (pose_data, frame) in enumerate(zip(pose_data_samples, frames_copy)):
            visualizer.add_datasample(
                'result',
                frame,
                data_sample=pose_data,
                draw_gt=False,
                draw_heatmap=False,
                draw_bbox=True,
                draw_pred=True,
                show=False,
                wait_time=0,
                out_file=None,
                kpt_thr=0.3
            )
            frames_copy[i] = visualizer.get_image()

    # Add spatio-temporal action detection results to the frames
    for annotation_idx in range(num_annotations):
        current_annotation = annotations[annotation_idx]
        if not current_annotation:
            continue

        for frame_idx in range(frames_per_annotation):
            frame_index = annotation_idx * frames_per_annotation + frame_idx
            current_frame = frames_copy[frame_index]

            for annotation in current_annotation:
                bbox, label, score, track_id = annotation

                if not label:
                    continue

                # Scale bounding box coordinates to the frame size
                bbox = (bbox * scale_ratio).astype(np.int64)
                start_point = tuple(bbox[:2])  # Top-left corner of the bounding box
                end_point = tuple(bbox[2:])    # Bottom-right corner of the bounding box

                # Draw bounding box if no pose data is available
                if not pose_data_samples:
                    cv2.rectangle(current_frame, start_point, end_point, plate[0], 2)

                # Add labels and scores to the bounding box
                for label_idx, (label_text, label_score) in enumerate(zip(label, score)):
                    if label_idx >= max_labels_per_box:
                        break

                    # Format label and score text
                    abbreviated_label = abbrev(label_text)
                    label_score_text = f'{abbreviated_label}: {label_score * 100:.1f}%'
                    track_id_text = f'ID: {int(track_id)}'

                    # Calculate text positions
                    label_position = (start_point[0], start_point[1] + 18 + label_idx * 18)
                    track_id_position = (start_point[0], start_point[1] + 18 + label_idx * 18 - 25)

                    # Draw background rectangle for the label
                    text_size = cv2.getTextSize(label_score_text, FONTFACE, FONTSCALE, THICKNESS)[0]
                    text_width = text_size[0]
                    rect_top_right = (label_position[0] + text_width, label_position[1] - 14)
                    rect_bottom_left = (label_position[0], label_position[1] + 2)
                    cv2.rectangle(current_frame, rect_top_right, rect_bottom_left, plate[label_idx + 1], -1)

                    # Determine text color based on label
                    danger_actions = ['melempar', 'membidik senapan', 'membidik pistol', 'memukul', 'menendang', 'menusuk']
                    text_color = (255, 0, 0) if label_text in danger_actions else (255, 255, 255)

                    # Draw text on the frame
                    cv2.putText(current_frame, track_id_text, track_id_position, FONTFACE, FONTSCALE, text_color, THICKNESS, LINETYPE)
                    cv2.putText(current_frame, label_score_text, label_position, FONTFACE, FONTSCALE, text_color, THICKNESS, LINETYPE)

    return frames_copy


#AT
#====================================
def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.

    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}

def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name

def pack_result(human_detection, result, img_h, img_w, track_id_list):
    """Short summary.

    Args:
        human_detection (np.ndarray): Human detection result.
        result (type): The predicted label of each human proposal.
        img_h (int): The image height.
        img_w (int): The image width.
        track_id_list (list[int]): The list of ID of the tracked object.

    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    human_detection[:, 0::2] /= img_w
    human_detection[:, 1::2] /= img_h
    results = []
    if result is None:
        return None
    for prop, res, id in zip(human_detection, result, track_id_list):
        res.sort(key=lambda x: -x[1])
        results.append(
            (prop.data.cpu().numpy(), [x[0] for x in res], [x[1] for x in res], id))
    return results


def expand_bbox(bbox, h, w, ratio=1.25):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    width = x2 - x1
    height = y2 - y1

    square_l = max(width, height)
    new_width = new_height = square_l * ratio

    new_x1 = max(0, int(center_x - new_width / 2))
    new_x2 = min(int(center_x + new_width / 2), w)
    new_y1 = max(0, int(center_y - new_height / 2))
    new_y2 = min(int(center_y + new_height / 2), h)
    return (new_x1, new_y1, new_x2, new_y2)


def cal_iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    intersect = w * h
    union = s1 + s2 - intersect
    iou = intersect / union

    return iou


def skeleton_based_stdet(args, label_map, human_detections, pose_results,
                         num_frame, clip_len, frame_interval, h, w, scele_config):
    window_size = clip_len * frame_interval
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           args.predict_stepsize)

    #skeleton_config = mmengine.Config.fromfile(args.skeleton_config)
    skeleton_config = scele_config
    num_class = max(label_map.keys()) + 1  # for CIIS dataset (9 + 1) == len(label_map)
    skeleton_config.model.cls_head.num_classes = num_class
    print("test here")
    skeleton_stdet_model = init_recognizer(skeleton_config,
                                           args.skeleton_stdet_checkpoint,
                                           args.device)

    skeleton_predictions = []

    print('Performing SpatioTemporal Action Detection for each clip')
    prog_bar = mmengine.ProgressBar(len(timestamps))
    for timestamp in timestamps:
        proposal = human_detections[timestamp - 1]
        if proposal.shape[0] == 0:  # no people detected
            skeleton_predictions.append(None)
            continue

        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)
        num_frame = len(frame_inds)  # 30

        pose_result = [pose_results[ind] for ind in frame_inds]

        skeleton_prediction = []
        for i in range(proposal.shape[0]):  # num_person
            skeleton_prediction.append([])

            fake_anno = dict(
                frame_dict='',
                label=-1,
                img_shape=(h, w),
                origin_shape=(h, w),
                start_index=0,
                modality='Pose',
                num_clips=1,
                clip_len=clip_len,
                total_frames=num_frame)
            num_person = 1

            num_keypoint = 17
            keypoint = np.zeros(
                (num_person, num_frame, num_keypoint, 2))  # M T V 2
            keypoint_score = np.zeros(
                (num_person, num_frame, num_keypoint))  # M T V

            # pose matching
            person_bbox = proposal[i][:4]  #x1, y1, x2, y2
            area = expand_bbox(person_bbox, h, w)

            for j, poses in enumerate(pose_result):  # num_frame
                max_iou = float('-inf')
                index = -1
                if len(poses['keypoints']) == 0:
                    continue
                for k, bbox in enumerate(poses['bboxes']):  # num_person
                    iou = cal_iou(bbox, area)
                    if max_iou < iou:  # if isBelong
                        index = k
                        max_iou = iou
                keypoint[0, j] = poses['keypoints'][index]
                keypoint_score[0, j] = poses['keypoint_scores'][index]

            fake_anno['keypoint'] = keypoint
            fake_anno['keypoint_score'] = keypoint_score

            output = inference_recognizer(skeleton_stdet_model, fake_anno)
            # for multi-label recognition
            score = output.pred_score.tolist()
            for k in range(len(score)):  # 10
                if k not in label_map:
                    continue
                if score[k] > args.action_score_thr:
                    skeleton_prediction[i].append((label_map[k], score[k]))

            # crop the image -> resize -> extract pose -> as input for poseC3D

        skeleton_predictions.append(skeleton_prediction)
        prog_bar.update()

    return timestamps, skeleton_predictions
#====================================

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

def process_and_display(batchFrames, fps):
    """Display frames with FPS and handle exit key."""
    global running
    args = arg_parser()
    skele_config = mmengine.Config.fromfile(args.skeleton_config)
    num_frame = len(batchFrames)
    height, width, _ = batchFrames[0].shape
    new_w, new_h = width, height
    frames = batchFrames
    w_ratio, h_ratio = new_w / width, new_h / height

    # Detection
    results = model(batchFrames, classes=[0])
    print("how many: " + str(len(results)))
    conf_thres = 0
    
    framesDet, human_detections, targetFrameList = [], [], []
    for frame_idx, (frame, result) in enumerate(zip(batchFrames, results)):
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
        human_detections.append(np.array(detXYXY, dtype=np.float32) if detXYXY else np.empty((0, 4), dtype=np.float32))

    framesDet = np.array(framesDet, dtype=object)

    # check
    print(f"framesDet: {framesDet}")
    print(f"humanDet: {human_detections}")
    print(f"targetList: {targetFrameList}")
    
    #####################################
    #====================================
    # Pose Estimation
    pose_datasample = None
    print("test pose detection")
    pose_results, pose_datasample = pose_inference(
        args.pose_config,
        args.pose_checkpoint,
        batchFrames,
        human_detections,
        device=args.device)
    
    #====================================
    # Action Recognition
    # Load spatio-temporal detection label_map
    stdet_label_map = load_label_map(args.label_map_stdet)

    stdet_preds = None

    print('Use skeleton-based SpatioTemporal Action Detection')
    # clip_len, frame_interval = 30, 1
    clip_len, frame_interval = args.predict_stepsize, 1
    timestamps, stdet_preds = skeleton_based_stdet(args, stdet_label_map,
                                                    human_detections,
                                                    pose_results, num_frame,
                                                    clip_len,
                                                    frame_interval, height, width, skele_config)
    
    for i in range(len(human_detections)):
        det = human_detections[i]
        det[:, 0:4:2] *= w_ratio
        det[:, 1:4:2] *= h_ratio
        human_detections[i] = torch.from_numpy(det[:, :4]).to(args.device)

    stdet_results = []
    for timestamp, prediction in zip(timestamps, stdet_preds):
        human_detection = human_detections[timestamp - 1]
        stdet_results.append(
            pack_result(human_detection, prediction, new_h, new_w, 
                        targetFrameList)
                        )

    def dense_timestamps(timestamps, n):
        """Make it nx frames."""
        old_frame_interval = (timestamps[1] - timestamps[0])
        start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
        new_frame_inds = np.arange(
            len(timestamps) * n) * old_frame_interval / n + start
        return new_frame_inds.astype(np.int64)

    dense_n = int(args.predict_stepsize / args.output_stepsize)

    print(timestamps)
    output_timestamps = dense_timestamps(timestamps, dense_n) + 1
    frames = [
        batchFrames[timestamp - 1]
        for timestamp in output_timestamps
    ]   

    pose_datasample = [
        pose_datasample[timestamp - 1] for timestamp in output_timestamps
    ]

    print("what inside stdet:", stdet_results)
    vis_frames = visualize_frames_with_annotations(args, frames, stdet_results, pose_datasample,
                                                                                            None)
    #====================================
    #####################################


    """
    # Visualize
    for frame, detection, targetList in zip(batchFrames, framesDet, targetFrameList):
        # Add FPS text to frame
        # res = tracker.update(dets, frame)
        if len(detection) > 0:
            for person, id in zip(detection, targetList):
                x1, y1, x2, y2, conf, _ = person
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                personID = id
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                text1 = f"Confidence: {int(conf*100)}%"
                text2 = f"ID: {personID}"
                location1 = (x1+5, y1+15)
                location2 = (x1+5, y1+30)

                # text1
                shadow_color = (0, 0, 0)
                cv2.putText(frame, text1, (location1[0] + 1, location1[1] + 1), FONT, FONT_SCALE, shadow_color, FONT_THICKNESS, LINETYPE)
                cv2.putText(frame, text1, location1, FONT, FONT_SCALE, FONTCOLOR, FONT_THICKNESS, LINETYPE)

                # text2
                cv2.putText(frame, text2, (location2[0] + 1, location2[1] + 1), FONT, FONT_SCALE, shadow_color, FONT_THICKNESS, LINETYPE)
                cv2.putText(frame, text2, location2, FONT, FONT_SCALE, FONTCOLOR, FONT_THICKNESS, LINETYPE)
        """
    for frame in vis_frames:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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