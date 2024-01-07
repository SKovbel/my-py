#!/bin/bash
source _source.sh

while true
do
    $DIR_BIN/augmentation.sh

    cd $DIR_YOLO
    yolo task=detect mode=train resume model=$DIR_MODEL/weights/last.pt imgsz=640 data=$DIR_BIN/data.yaml \
        batch=6 amp=False workers=2 name=$DIR_MODEL
    timeout 1000 python train.py --data $DIR_BIN/data.yaml --resume $DIR_MODEL/weights/last.pt  --name $DIR_MODEL
    cd $DIR_BIN

	./opencv.sh
	sleep 120
done
