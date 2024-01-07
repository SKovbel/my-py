#!/bin/bash

# https://learnopencv.com/custom-object-detection-training-using-yolov5/
# https://blog.paperspace.com/train-yolov5-custom-data/

source _source.sh

rm -rf $DIR_RESULT
$DIR_BIN/augmentation.sh

# Train
cd $DIR_YOLO
python train.py --data $DIR_BIN/data.yaml --weights yolov5s.pt --img 640 --epochs 4000 --batch-size 16  --name $DIR_RESULT
cd $DIR_BIN