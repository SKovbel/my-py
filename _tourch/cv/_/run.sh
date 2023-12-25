#!/bin/bash

PWD=`pwd`
URL=https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
SAMPLES_DIR="../../tmp/pytorch/cv1"

rm -rf $SAMPLES_DIR

echo "${SAMPLES_DIR}"
if [ ! -f $SAMPLES_DIR/PennFudanPed.zip ]
then
    mkdir -p $SAMPLES_DIR
    wget -P "${SAMPLES_DIR}" $URL
    unzip $SAMPLES_DIR/PennFudanPed.zip -d "${SAMPLES_DIR}"
fi
ls $SAMPLES_DIR
