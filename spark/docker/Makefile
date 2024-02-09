ROOT_DIR = $(shell pwd)

export WORKING_DIR = $(ROOT_DIR)/../../tmp/spark-working-data
#export WORKING_DIR = $(ROOT_DIR)/working-data

SPARK_TGZ_URL = https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz

main: help

init:
	# rm -rf $(WORKING_DIR)
	mkdir -p $(WORKING_DIR)/spark; \
	mkdir -p $(WORKING_DIR)/downloads; \
	wget -cnv -O $(WORKING_DIR)/downloads/spark.tgz $(SPARK_TGZ_URL); \
	cp $(WORKING_DIR)/downloads/spark.tgz $(ROOT_DIR)/spark;

clean-install:
	rm -rf $(WORKING_DIR)/downloads/spark.tgz;

stop:
	docker stop $(sudo docker ps -a -q); \

start-spark:
	docker-compose -f spark/docker-compose.yml up

probe-spark:
	python3.10 test/test-spark.py

start: start-spark

help:
	# 
	# Hello!
	#
	# make init
	# 
	# make stop - stop docker
	# make start-spark - start docker
	#
	# make probe-spark - test spark
	#