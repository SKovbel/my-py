# https://github.com/matnar/docker-hadoop/blob/master/config/hdfs-site.xml
FROM ubuntu:18.04
USER root

# install dev tools
RUN apt-get update
RUN apt-get install -y curl tar sudo openssh-server rsync openjdk-8-jre-headless vim net-tools

# passwordless ssh
RUN ssh-keygen -q -N "" -t rsa -f /root/.ssh/id_rsa 
RUN cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys

# # java
ENV JAVA_HOME /usr/lib/jvm/default-java
ENV PATH $PATH:$JAVA_HOME/bin

ENV HADOOP_COMMON_HOME /usr/local/hadoop
ENV HADOOP_HDFS_HOME /usr/local/hadoop
ENV HADOOP_HOME /usr/local/hadoop
ENV HADOOP_MAPRED_HOME /usr/local/hadoop
ENV HADOOP_YARN_HOME /usr/local/hadoop
ENV HADOOP_CONF_DIR /usr/local/hadoop/etc/hadoop
ENV YARN_CONF_DIR $HADOOP_HOME/etc/hadoop
ENV PATH $PATH:$HADOOP_HOME/bin
ENV HDFS_NAMENODE_USER "root"
ENV HDFS_DATANODE_USER "root"
ENV HDFS_SECONDARYNAMENODE_USER "root"
ENV YARN_RESOURCEMANAGER_USER "root"
ENV YARN_NODEMANAGER_USER "root"


# # hadoop
COPY hadoop.tar.gz /tmp/hadoop.tar.gz
RUN tar -zxf /tmp/hadoop.tar.gz -C /usr/local/
RUN rm /tmp/hadoop.tar.gz
RUN ln -s /usr/local/hadoop-3.3.6 $HADOOP_HOME
RUN mkdir -p /usr/local/hadoop/logs

# # pseudo distributed
COPY config/ssh_config /root/.ssh/config
COPY config/bootstrap.sh $HADOOP_HOME/bootstrap.sh
COPY config/hadoop-env.sh $HADOOP_HOME/etc/hadoop/hadoop-env.sh
COPY config/core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml
COPY config/hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml
COPY config/workers $HADOOP_HOME/etc/hadoop/workers
COPY config/core-site.xml.template $HADOOP_HOME/etc/hadoop/core-site.xml.template

# 
RUN chmod 600 /root/.ssh/config
RUN chown root:root /root/.ssh/config
# 
RUN chown root:root $HADOOP_HOME/bootstrap.sh
RUN chmod 777 $HADOOP_HOME/bootstrap.sh
# 
ENV BOOTSTRAP $HADOOP_HOME/bootstrap.sh
# 
CMD $HADOOP_HOME/bootstrap.sh

# # Hdfs ports
EXPOSE 9866 9867 9870 9864 9868 9820 9000 54310
# # Mapred ports
EXPOSE 10020 19888
# #Yarn ports
EXPOSE 8030 8031 8032 8033 8040 8042 8088
# #Other ports
EXPOSE 49707 2122
# SSH
EXPOSE 22
