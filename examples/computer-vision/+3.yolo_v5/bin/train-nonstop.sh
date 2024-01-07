#!/bin/bash
source _source.sh

while true
do
    $DIR_BIN/augmentation.sh

    cd $DIR_YOLO
    timeout 1000 python train.py --data $DIR_BIN/data.yaml --resume $DIR_RESULT/weights/last.pt  --name $DIR_RESULT
    cd $DIR_BIN

	./opencv.sh
	sleep 120
done
