#!/bin/bash
source _source.sh

mkdir -p $ORACLE_DIR/scripts
mkdir -p $ORACLE_DIR/dump
mkdir -p $ORACLE_DIR/oradata

mkdir -p $BACKUP_DIR/log

sudo chown -R ksm:ksm $ORACLE_DIR # @todo - replace with configs
sudo chown -R ksm:ksm $BACKUP_DIR # @todo - replace with configs
sudo chmod 777 -R $ORACLE_DIR # @todo - replace with configs
sudo chmod 777 -R $BACKUP_DIR # @todo - replace with configs
sudo chmod 666 /var/run/docker.sock

docker login container-registry.oracle.com
docker-compose -f oracle.yml up


