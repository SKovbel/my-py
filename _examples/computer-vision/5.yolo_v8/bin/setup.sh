#!/bin/bash
source _source.sh

# Clear
#rm -rf $DIR_DATA

# Setup dirs
mkdir -p $DIR_TMP
mkdir -p $DIR_SAMPELS
mkdir -p $DIR_MODEL

# Setup YOLO
cd $DIR_DATA
git clone https://github.com/ultralytics/yolov8.git

cd $DIR_YOLO
pip install -r requirements.txt


# Finalize
cd $DIR_BIN

#wget https://www.dropbox.com/s/qvglw8pqo16769f/pothole_dataset_v8.zip?dl=1 -O pothole_dataset_v8.zip
