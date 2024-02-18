from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("HDFS Upload Example") \
    .getOrCreate()

# Get the Hadoop FileSystem object
fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())

# Specify the local file path
local_file_path = "/home/work/python/spark/tests/files/iras.csv"

# Specify the HDFS destination path
hdfs_destination_path = "hdfs://0.0.0.0:9000/"

# Convert paths to Hadoop Path objects
local_path = spark._jvm.org.apache.hadoop.fs.Path(local_file_path)
hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_destination_path)

# Upload file to HDFS
fs.copyFromLocalFile(local_path, hdfs_path)

print("File uploaded successfully.")

# Stop the SparkSession
spark.stop()