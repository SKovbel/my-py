#!/bin/bash
source _source.sh

# Clear
rm -rf $DIR_DATA

# Setup dirs
mkdir -p $DIR_TMP
mkdir -p $DIR_SAMPELS
mkdir -p $DIR_RESULT

# Setup YOLO
cd $DIR_DATA
git clone https://github.com/ultralytics/yolov5.git

cd $DIR_YOLO
pip install -r requirements.txt

# Setup data for train
cd $DIR_SAMPELS
curl -L "https://public.roboflow.com/ds/xKLV14HbTF?key=aJzo7msVta" > roboflow.zip
unzip roboflow.zip
rm roboflow.zip

# Finalize
cd $DIR_BIN
