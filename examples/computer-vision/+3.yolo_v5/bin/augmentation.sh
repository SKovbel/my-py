#!/bin/bash
source _source.sh

rm -rf $DIR_SAMPELS/train-a
python $DIR_ROOT/augmentation.py