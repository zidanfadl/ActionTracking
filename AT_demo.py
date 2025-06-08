# -*- coding: utf-8 -*-
"""
main.py

This script performs real-time, multi-person spatio-temporal action recognition using
a pipeline of deep learning models: YOLOv8 for object detection, BoTSORT for tracking,
HRNet for pose estimation, and PoseC3D for action recognition.

The pipeline is designed to be modular and can process input from either a live camera
feed or a pre-recorded video file. It is multithreaded for camera input to ensure
smooth frame capture and processing.

Key Features:
- Real-time person detection and tracking.
- Skeleton-based pose estimation for each tracked person.
- Spatio-temporal action recognition based on pose sequences.
- Visualization of detection boxes, pose keypoints, and action labels.
- Persistent tracking and highlighting of individuals who perform dangerous actions.
- Option to save the processed video output.

Optimization Enhancements:
- `torch.no_grad()` is used during inference to reduce memory and speed up computation.
- Optional FP16 (half-precision) inference for significant speedup on compatible GPUs.
- Parallelized pose estimation using a ThreadPoolExecutor to better utilize hardware.
"""

# Standard Library Imports
import argparse
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Third-Party Library Imports
import cv2
import mmcv
import mmengine
import moviepy.editor as mpy
import numpy as np
import torch
from boxmot import BoTSORT
from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.registry import VISUALIZERS
from mmaction.utils import frame_extract
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import PoseDataSample, merge_data_samples
from ultralytics import YOLO

# --- Constants and Configuration ---

# Visualization settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
LINETYPE = cv2.LINE_AA

# Color palette for visualizations
PLATEBLUE_HEX = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4-ff0000-ffff00'

def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    """Converts a 6-digit hex string to an (R, G, B) tuple."""
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)

PLATEBLUE = [hex_to_rgb(h) for h in PLATEBLUE_HEX.split('-')]
DANGER_COLOR = PLATEBLUE[6]
SAFE_COLOR = PLATEBLUE[1]

DANGER_ACTIONS = ['melempar', 'membidik senapan', 'membidik pistol', 'memukul', 'menendang']

class ActionRecognizer:
    """
    Encapsulates the entire action recognition pipeline, from frame capture to
    visualization.
    """
    def __init__(self, args: argparse.Namespace):
        """
        Initializes models, configurations, and state variables.

        Args:
            args (argparse.Namespace): Command-line arguments specifying configurations.
        """
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.fp16 = args.fp16 and self.device.type != 'cpu'
        
        print(f"Using device: {self.device}")
        if self.fp16:
            print("Optimization: FP16 (half-precision) inference is ENABLED.")

        # --- Load Models ---
        self.yolo_model = YOLO(args.yolo_weights)
        self.tracker = BoTSORT(
            reid_weights=Path(args.reid_weights),
            device=self.device,
            half=self.fp16
        )
        self.pose_model = init_model(args.pose_config, args.pose_checkpoint, device=self.device)
        self.skeleton_model = init_recognizer(args.skel_config, args.skel_checkpoint, device=self.device)
        
        # Enable FP16 if requested
        if self.fp16:
            self.yolo_model.half()
            self.pose_model.half()
            self.skeleton_model.half()
        
        # --- Visualization ---
        pose_cfg = mmengine.Config.fromfile(args.pose_config)
        self.pose_visualizer = VISUALIZERS.build(
            pose_cfg.visualizer | {'line_width': 2, 'radius': 4}
        )
        if hasattr(self.pose_model, 'dataset_meta'):
             self.pose_visualizer.set_dataset_meta(self.pose_model.dataset_meta)

        # --- Label Maps ---
        self.stdet_label_map = self._load_label_map(args.label_map_stdet)
        
        # --- Real-time Processing State ---
        self.buffer = deque(maxlen=args.batch_size)
        self.buffer_lock = threading.Lock()
        self.frame_available = threading.Condition(self.buffer_lock)
        self.running = True
        self.video_frame_output = []
        self.danger_ids = set() # Set to store IDs of individuals who performed dangerous actions

    @staticmethod
    def _load_label_map(file_path: str) -> Dict[int, str]:
        """Loads a label map from a file."""
        try:
            with open(file_path, 'r') as f:
                lines = [x.strip().split(': ') for x in f.readlines()]
                return {int(x[0]): x[1] for x in lines}
        except FileNotFoundError:
            print(f"Error: Label map file not found at {file_path}")
            return {}

    def _abbrev(self, name: str) -> str:
        """Abbreviates action names for cleaner visualization."""
        return name.split('(')[0].strip()

    def run(self):
        """Main entry point to start the action recognition process."""
        if self.args.input_type == 'video':
            self._process_video()
        elif self.args.input_type == 'camera':
            self._process_camera()
        else:
            print(f"Error: Invalid input type '{self.args.input_type}'")

    def _process_camera(self):
        """Starts the multithreaded pipeline for live camera input."""
        capture_thread = threading.Thread(target=self._capture_frames)
        process_thread = threading.Thread(target=self._process_batches)

        try:
            print("Starting camera capture and processing threads...")
            capture_thread.start()
            process_thread.start()
            while process_thread.is_alive():
                process_thread.join(0.1)
        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt, shutting down...")
        finally:
            self.running = False
            with self.buffer_lock:
                self.frame_available.notify_all()
            
            capture_thread.join()
            process_thread.join()
            cv2.destroyAllWindows()
            self._save_output_video()
            print("Program terminated gracefully.")

    def _capture_frames(self):
        """Continuously captures frames from the camera and adds to a buffer."""
        cap = cv2.VideoCapture(self.args.camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera with index {self.args.camera_index}")
            self.running = False
            return
            
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("Camera feed ended.")
                    break
                with self.buffer_lock:
                    self.buffer.append(frame)
                    self.frame_available.notify()
        finally:
            cap.release()
            self.running = False

    def _process_batches(self):
        """Processes frames in batches for real-time inference and display."""
        start_time = time.time()
        frames_processed = 0
        
        while self.running:
            with self.buffer_lock:
                while len(self.buffer) < self.args.batch_size and self.running:
                    if not self.frame_available.wait(timeout=1):
                        if not self.running:
                            break
                if not self.running and not self.buffer:
                    break
                batch_frames = list(self.buffer)
            
            if not batch_frames:
                continue

            processed_frames = self._process_full_pipeline(batch_frames)
            if not processed_frames:
                continue
            
            frames_processed += 1 # We display one frame at a time
            elapsed_time = time.time() - start_time
            fps = frames_processed / elapsed_time if elapsed_time > 0 else 0
            
            self._display_frames(processed_frames, fps)
            
            with self.buffer_lock:
                if self.buffer:
                    self.buffer.popleft()

    def _process_video(self):
        """Processes a video file frame by frame."""
        try:
            frame_paths, original_frames = frame_extract(self.args.video_path, short_side=720)
        except Exception as e:
            print(f"Error extracting frames from video '{self.args.video_path}': {e}")
            return
            
        if not original_frames:
            print("No frames extracted from the video.")
            return

        print(f"Processing {len(original_frames)} frames from video...")
        
        chunk_size = 32
        for i in range(0, len(original_frames), chunk_size):
            chunk = original_frames[i:i+chunk_size]
            processed_chunk = self._process_full_pipeline(chunk)
            self.video_frame_output.extend(processed_chunk)
            print(f"  Processed frames {i} to {i+len(chunk)}")

        self._save_output_video()
        print("Video processing complete.")

    def _process_full_pipeline(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Runs the complete detection, tracking, pose, and action recognition pipeline.
        
        Args:
            frames (List[np.ndarray]): A list of frames to process.

        Returns:
            List[np.ndarray]: The list of processed frames with visualizations.
        """
        if not frames:
            return []
            
        h, w, _ = frames[0].shape
        
        # --- Stage 1: Detection and Tracking ---
        with torch.no_grad():
            yolo_results = self.yolo_model(frames, classes=[0], verbose=False)
        
        all_tracked_objects_per_frame = []
        for frame, result in zip(frames, yolo_results):
            boxes = result.boxes.cpu().numpy()
            dets = [[*box.xyxy[0], box.conf[0], box.cls[0]] 
                    for box in boxes if box.conf[0] > self.args.det_score_thr]
            tracked_objects = self.tracker.update(np.array(dets), frame)
            all_tracked_objects_per_frame.append(tracked_objects)
        
        all_detections = [[d[:4] for d in tracked] if len(tracked) > 0 else np.empty((0,4)) 
                          for tracked in all_tracked_objects_per_frame]
            
        # --- Stage 2: Parallel Pose Estimation ---
        def run_pose_estimation_task(task_data):
            frame, dets = task_data
            with torch.no_grad():
                pose_datasamples = inference_topdown(self.pose_model, frame, dets, bbox_format='xyxy')
                return merge_data_samples(pose_datasamples)

        pose_tasks = zip(frames, all_detections)
        with ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
            all_pose_datasamples = list(executor.map(run_pose_estimation_task, pose_tasks))
        
        all_pose_results = [pds.pred_instances.to_dict() for pds in all_pose_datasamples]

        # --- Stage 3: Spatio-Temporal Action Detection ---
        action_predictions = self._skeleton_based_stdet(all_tracked_objects_per_frame, all_pose_results, h, w)

        # --- Stage 4: Visualization ---
        vis_frames = self._visualize(frames, all_tracked_objects_per_frame, action_predictions, all_pose_datasamples)
        
        return vis_frames

    def _skeleton_based_stdet(self, all_tracked_objects_per_frame, all_pose_results, h, w):
        """Performs skeleton-based action detection, updating danger_ids."""
        clip_len, frame_interval = self.args.skel_clip_len, self.args.skel_frame_interval
        window_size = clip_len * frame_interval
        num_frames = len(all_tracked_objects_per_frame)
        
        timestamps = np.arange(window_size // 2, num_frames - window_size // 2, self.args.predict_stepsize)
        final_predictions = {}

        for ts in timestamps:
            tracked_this_ts = all_tracked_objects_per_frame[ts]
            if len(tracked_this_ts) == 0: continue

            start_frame = ts - (clip_len // 2 - 1) * frame_interval
            frame_indices = np.clip(start_frame + np.arange(0, window_size, frame_interval), 0, num_frames - 1).astype(int)
            clip_pose_results = [all_pose_results[i] for i in frame_indices]
            
            clip_predictions = []
            for person_data in tracked_this_ts:
                person_bbox = person_data[:4]
                track_id = int(person_data[4])
                
                keypoints, scores = self._match_poses_to_person(person_bbox, clip_pose_results, h, w)
                if keypoints is None: continue
                
                fake_anno = self._prepare_fake_anno(keypoints, scores, h, w, clip_len)
                
                with torch.no_grad():
                    output = inference_recognizer(self.skeleton_model, fake_anno)
                
                action_scores = output.pred_score.tolist()
                person_actions = []
                is_danger = False
                for k, score in enumerate(action_scores):
                    if k in self.stdet_label_map and score > self.args.action_score_thr:
                        action_name = self.stdet_label_map[k]
                        person_actions.append((action_name, score))
                        if action_name in DANGER_ACTIONS:
                            is_danger = True

                if is_danger:
                    self.danger_ids.add(track_id)

                if person_actions:
                    person_actions.sort(key=lambda x: -x[1])
                    clip_predictions.append({"track_id": track_id, "actions": person_actions})

            if clip_predictions:
                final_predictions[ts] = clip_predictions
        
        return final_predictions

    def _match_poses_to_person(self, person_bbox, clip_pose_results, h, w):
        """Finds the corresponding pose for a person's bbox across a clip."""
        num_keypoints = self.pose_model.dataset_meta['num_keypoints']
        keypoints = np.zeros((1, self.args.skel_clip_len, num_keypoints, 2))
        scores = np.zeros((1, self.args.skel_clip_len, num_keypoints))
        expanded_bbox = self._expand_bbox(person_bbox, h, w)

        for i, poses in enumerate(clip_pose_results):
            if poses['bboxes'].shape[0] == 0: continue
            
            ious = [self._calc_iou(expanded_bbox, pose_bbox) for pose_bbox in poses['bboxes']]
            best_match_idx = np.argmax(ious)

            if ious[best_match_idx] > 0.3:
                keypoints[0, i] = poses['keypoints'][best_match_idx]
                scores[0, i] = poses['keypoint_scores'][best_match_idx]
        return keypoints, scores

    @staticmethod
    def _expand_bbox(bbox, h, w, ratio=1.25):
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        width, height = x2 - x1, y2 - y1
        square_l = max(width, height) * ratio
        new_x1 = max(0, int(center_x - square_l / 2))
        new_y1 = max(0, int(center_y - square_l / 2))
        new_x2 = min(w, int(center_x + square_l / 2))
        new_y2 = min(h, int(center_y + square_l / 2))
        return new_x1, new_y1, new_x2, new_y2

    @staticmethod
    def _calc_iou(box1, box2):
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
        s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        s2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        xmin, ymin = max(xmin1, xmin2), max(ymin1, ymin2)
        xmax, ymax = min(xmax1, xmax2), min(ymax1, ymax2)
        w, h = max(0, xmax - xmin), max(0, ymax - ymin)
        intersect = w * h
        union = s1 + s2 - intersect
        return intersect / (union + 1e-6)

    def _prepare_fake_anno(self, keypoints, scores, h, w, clip_len):
        return dict(
            frame_dir='', total_frames=clip_len, label=-1, img_shape=(h, w),
            original_shape=(h, w), start_index=0, modality='Pose', num_clips=1,
            clip_len=clip_len, keypoint=keypoints, keypoint_score=scores
        )

    def _visualize(self, frames, all_tracked_objects_per_frame, action_predictions, pose_datasamples):
        """Draws all annotations onto the frames, with persistent danger highlighting."""
        frames_copy = [mmcv.imconvert(frame, 'bgr', 'rgb') for frame in frames]
        
        # Draw poses first, without their default bounding boxes
        for i, (pds, frame) in enumerate(zip(pose_datasamples, frames_copy)):
            self.pose_visualizer.add_datasample(
                'result', frame, data_sample=pds, draw_pred=True,
                draw_bbox=False, draw_heatmap=False, show=False, kpt_thr=0.3
            )
            frames_copy[i] = self.pose_visualizer.get_image()
            
        # Draw tracked objects, custom bounding boxes, and labels
        for i, frame in enumerate(frames_copy):
            tracked_objects = all_tracked_objects_per_frame[i]
            frame_actions = {pred['track_id']: pred['actions'] 
                             for pred in action_predictions.get(i, [])}

            for obj in tracked_objects:
                x1, y1, x2, y2, track_id = map(int, obj[:5])
                
                # Determine color based on persistent danger tracking
                box_color = DANGER_COLOR if track_id in self.danger_ids else SAFE_COLOR
                
                # Draw the bounding box with the appropriate color
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Prepare text labels
                display_text = [f"ID: {track_id}"]
                if track_id in frame_actions:
                    for label, score in frame_actions[track_id][:2]:
                        display_text.append(f"{self._abbrev(label)}: {score:.1%}")

                # Draw text labels above the bounding box
                y_offset = y1 - 7
                for line in reversed(display_text):
                    (text_w, text_h), _ = cv2.getTextSize(line, FONT, FONT_SCALE, FONT_THICKNESS)
                    bg_y1 = y_offset - text_h - 4
                    bg_y2 = y_offset
                    cv2.rectangle(frame, (x1, bg_y1), (x1 + text_w + 4, bg_y2), box_color, -1)
                    cv2.putText(frame, line, (x1 + 2, y_offset - 5), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, LINETYPE)
                    y_offset -= (text_h + 6)
                    
        return frames_copy

    def _display_frames(self, frames: List[np.ndarray], fps: float):
        """Displays processed frames and handles user input."""
        if self.args.save_output and self.args.input_type == 'camera':
            self.video_frame_output.append(frames[-1])

        frame_to_show = frames[-1]
        cv2.putText(frame_to_show, f"FPS: {fps:.1f}", (10, 30), FONT, 1, (0, 255, 255), 2, LINETYPE)
        cv2.imshow('Action Recognition', frame_to_show)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.running = False

    def _save_output_video(self):
        """Saves the collected frames to a video file."""
        if not self.args.save_output or not self.video_frame_output: return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"video_{timestamp}.mp4"

        print(f"\nSaving {len(self.video_frame_output)} frames to {filename}...")
        try:
            vid_clip = mpy.ImageSequenceClip(self.video_frame_output, fps=self.args.output_fps)
            vid_clip.write_videofile(str(filename), codec='libx264', logger='bar')
            print("Video saved successfully.")
        except Exception as e:
            print(f"Error saving video: {e}")

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Optimized Real-time Spatio-Temporal Action Recognition")
    # ... (arguments remain the same)
    # --- Input ---
    parser.add_argument('input_type', type=str, choices=['camera', 'video'], help="Input source type.")
    parser.add_argument('--video_path', type=str, default='assets/video/default.mp4', help="Path to the input video file.")
    parser.add_argument('--camera_index', type=int, default=0, help="Index of the camera to use.")
    
    # --- Model Weights and Configs ---
    parser.add_argument('--yolo_weights', type=str, default='assets/weights/yolov8l.pt')
    parser.add_argument('--reid_weights', type=str, default='assets/weights/osnet_x0_25_msmt17.pt')
    parser.add_argument('--pose_config', type=str, default='configs/pose/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py')
    parser.add_argument('--pose_checkpoint', type=str, default='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth')
    parser.add_argument('--skel_config', type=str, default='configs/skeleton/posec3d/ciis_10.py')
    parser.add_argument('--skel_checkpoint', type=str, default='assets/weights/ciis_best.pth')
    parser.add_argument('--label_map_stdet', type=str, default='configs/skeleton/label_map.txt')

    # --- Processing and Optimization Parameters ---
    parser.add_argument('--device', type=str, default='cuda:0', help="Device for inference (e.g., 'cuda:0', 'cpu').")
    parser.add_argument('--fp16', action='store_true', help="Enable FP16 (half-precision) inference for speed.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel workers for pose estimation.")
    parser.add_argument('--batch_size', type=int, default=8, help="Processing window size (frames) for camera input.")
    parser.add_argument('--det_score_thr', type=float, default=0.7, help="Confidence threshold for human detection.")
    parser.add_argument('--action_score_thr', type=float, default=0.5, help="Score threshold for action recognition.")
    parser.add_argument('--predict_stepsize', type=int, default=8, help="Frame step size for new action predictions.")
    parser.add_argument('--skel_clip_len', type=int, default=16, help="Number of frames in a clip for the skeleton action model.")
    parser.add_argument('--skel_frame_interval', type=int, default=2, help="Frame interval within a clip.")

    # --- Output ---
    parser.add_argument('--save_output', action='store_true', help="Save the processed output to a video file.")
    parser.add_argument('--output_dir', type=str, default='data/output_videos', help="Directory to save output videos.")
    parser.add_argument('--output_fps', type=int, default=15, help="FPS for the saved output video.")


    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    action_recognizer = ActionRecognizer(args)
    action_recognizer.run()
