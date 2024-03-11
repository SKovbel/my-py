#!/bin/bash
#sudo /usr/local/hadoop/etc/hadoop/hadoop-env.sh
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre/

rm -f /tmp/*.pid
service ssh start

# $HADOOP_PREFIX/sbin/start-dfs.sh

# Launch bash console  
/bin/bash