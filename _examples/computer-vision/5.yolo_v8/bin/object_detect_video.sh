#!/bin/bash

source _source.sh

cd $DIR_YOLO
#python export.py --weights $DIR_MODEL/weights/last.pt --include onnx

cd $DIR_ROOT
python $DIR_ROOT/detect_video.py $DIR_MODEL/weights/best.pt $DIR_TMP/v/1.mp4 $DIR_TMP/v/2.mp4
