ROOT_DIR = $(shell pwd)

REBUILD = ''
export WORKING_DIR = $(ROOT_DIR)/../../tmp/spark-working-data
#export WORKING_DIR = $(ROOT_DIR)/working-data

SPARK_URL = https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
HADOOP_URL = https://downloads.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz

main: help

init:
	# rm -rf $(WORKING_DIR)
	mkdir -p $(WORKING_DIR)/spark
	mkdir -p $(WORKING_DIR)/hadoop
	mkdir -p $(WORKING_DIR)/downloads
	wget -cnv -O $(WORKING_DIR)/downloads/spark.tgz $(SPARK_URL)
	wget -cnv -O $(WORKING_DIR)/downloads/hadoop.tar.gz $(HADOOP_URL)
	cp $(WORKING_DIR)/downloads/spark.tgz $(ROOT_DIR)/spark
	cp $(WORKING_DIR)/downloads/hadoop.tar.gz $(ROOT_DIR)/hadoop

clean-downloads:
	rm -rf $(WORKING_DIR)/downloads/spark.tgz;
	rm -rf $(WORKING_DIR)/downloads/hadoop.tgz;

clean-dockers:
	yes | docker container stop $(docker container ls -aq) || true
	yes | docker container rm $(docker container ls -aq) || true
	yes | docker image prune -a || true
	yes | docker volume prune -a || true
	yes | docker network prune || true
	yes | docker system prune -a || true
	yes | docker system prune --volumes || true

stop:
	docker stop $(sudo docker ps -a -q)

rebuild:
	echo "Todo docker rebuid"

start-spark:
	docker-compose -f spark/docker-compose.yml up

start-hadoop:
	docker-compose -f hadoop/docker-compose.yml up

probe-hadoop:
	python3.10 test/test-hadoop.py

probe-spark:
	python3.10 test/test-spark.py

start: start-spark

hadoop-leave-safe:
	docker exec hadoop-name hadoop dfsadmin -safemode leave

help:
	# 
	# Hello!
	#
	# make init
	# 
	# make stop 				- stop all dockers
	#
	# make start-spark			- start spark
	# make start-hadoop			- start  hadoop
	#
	# make clean-downloads		- clean downloads folder
	# make clean-dockers		- remove all docker containers
	#
	# make probe-spark			- test spark
	#
	# make hadoop-leave-safe	- leafe save mode
	#