#!/bin/bash

python action_recognition.py \
    --video /home/ciis/Desktop/shitass.mp4 \
    --out-filename /home/ciis/Desktop/shitass_out2.mp4 \
    --det-config mmaction2/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --pose-config mmaction2/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
    --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --skeleton-config configs/skeleton/posec3d/ciis_10.py \
    --skeleton-stdet-checkpoint work_dirs/ciis_10_best-550/best_acc_top1_epoch_550.pth \
    --action-score-thr 0.75 \
    --label-map-stdet data/skeleton/ciis_label_map.txt \
    --predict-stepsize 2 \
    --output-fps 4

