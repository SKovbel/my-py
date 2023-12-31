#!/bin/bash
source _source.sh

rm -rf $DIR_SAMPELS/train-a

python $DIR_ROOT/train_augmentation.py \
    $DIR_SAMPELS/train/objects/images \
    $DIR_SAMPELS/train/objects/labels \
    $DIR_SAMPELS/train-a/objects/images \
    $DIR_SAMPELS/train-a/objects/labels
