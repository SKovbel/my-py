# Import the necessary modules
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Create a SparkSession
spark = SparkSession.builder \
   .appName("First Onex") \
   .master("spark://0.0.0.0:7077") \
   .getOrCreate()

rdd = spark.sparkContext.parallelize(range(1, 100))
print("THE SUM IS HERE: ", rdd.sum())


# Stop the SparkSession
spark.stop()
