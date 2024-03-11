from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark import SparkFiles
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder \
   .appName("First Onex") \
   .master("spark://0.0.0.0:7077") \
   .getOrCreate()

sc = spark.sparkContext
sc.addFile('/home/work/python/spark/docker/test/test.txt')
sc.listFiles
print(sc.listFiles)
file_list = sc.wholeTextFiles('spark://192.168.11.4:35577/files/*')
for file_path, content in file_list.collect():
    print("File path:", file_path)

def func(iterator):
    with open(SparkFiles.get("test.txt")) as testFile:
        for line in testFile:
            return [x * int(line) for x in iterator]
res = sc.parallelize([1, 2, 3, 4]).mapPartitions(func).collect()
print(res)