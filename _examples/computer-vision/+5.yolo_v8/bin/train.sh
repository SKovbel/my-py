#!/bin/bash

# https://learnopencv.com/custom-object-detection-training-using-yolov5/
# https://blog.paperspace.com/train-yolov5-custom-data/

source _source.sh

#rm -rf $DIR_MODEL
#$DIR_BIN/augmentation.sh

# Train
cd $DIR_YOLO
yolo task=detect mode=train model=yolov8s.pt imgsz=640 data=$DIR_BIN/data.yaml epochs=4000 batch=6 amp=False workers=2 name=$DIR_MODEL
#yolo task=detect mode=train model=yolov8n.pt imgsz=640 data=$DIR_BIN/data.yaml epochs=10 batch=8 amp=False workers=4  name=$DIR_MODEL \
#    save=True exist_ok=True lr0=0.01 lrf=0.002 momentum=0.937 weight_decay=0.0005 warmup_epochs=3.0 \
#    warmup_momentum=0.8 warmup_bias_lr=0.1 box=7.5 cls=0.5 dfl=1.5 pose=12.0 kobj=1.0 label_smoothing=0.1 nbs=32 \
#    hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 degrees=45.0 translate=0.1 scale=0.25 shear=10.0 perspective=0.0005 flipud=0.0 \
#    fliplr=0.5 mosaic=0.5 mixup=0.25 copy_paste=0.3 augment=True

cd $DIR_BIN


