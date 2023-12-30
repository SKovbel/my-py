# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
PWD = $(shell pwd)
PYTHON = python

DIR_SRC = data/src
DIR_DST = data/dst
DIR_MDL = data/models
DIR_EXP = data/exported

MODEL = mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz
MODEL_URL = http://download.tensorflow.org/models/object_detection/tf2/20200711/$(MODEL)


generate-tfrecord:
	$(PYTHON) util/gen_labelme.py $(DIR_SRC)/train --output $(DIR_DST)/train.json
	$(PYTHON) util/gen_labelme.py $(DIR_SRC)/val --output $(DIR_DST)/val.json

	$(PYTHON) util/tf/create_coco_tf_record.py --logtostderr \
		--train_image_dir=$(DIR_SRC)/train \
		--val_image_dir=$(DIR_SRC)/val \
		--train_annotations_file=$(DIR_DST)/train.json \
		--val_annotations_file=$(DIR_DST)/val.json \
		--include_masks=True \
		--output_dir=${DIR_DST}

	$(PYTHON) util/gen_label_map.py $(DIR_DST)/train.json $(DIR_DST)


download-model:
	mkdir -p $(DIR_MDL)
	curl --create-dirs -O --output-dir $(DIR_MDL) $(MODEL_URL)
	tar zxvf $(DIR_MDL)/$(MODEL) -C $(DIR_MDL)
	rm -rf $(DIR_MDL)/$(MODEL)


train-model:
	$(PYTHON) util/tf/model_main_tf2.py \
		--model_dir=models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8 \
		--pipeline_config_path=models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/pipeline.config


export-model:
	$(PYTHON) util/tf/exporter_main_v2.py \
		--input_type image_tensor \
		--pipeline_config_path $(DIR_MDL)/$(MODEL).pipeline.config \
		--trained_checkpoint_dir $(DIR_MDL)/$(MODEL) \
		--output_directory $(DIR_EXP)/$(MODEL)

