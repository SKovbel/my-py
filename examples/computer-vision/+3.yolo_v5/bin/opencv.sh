#!/bin/bash

source _source.sh

cd $DIR_YOLO
python export.py --weights $DIR_MODEL/weights/last.pt --include onnx

cd $DIR_ROOT
python $DIR_ROOT/opencv.py
