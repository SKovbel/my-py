# Import the necessary modules
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Create a SparkSession
spark = SparkSession.builder \
   .appName("First Onex") \
   .master("spark://0.0.0.0:7077") \
   .getOrCreate()

context = spark.sparkContext

rdd = context.parallelize(range(1, 100))
print("Sum: ", rdd.sum())
print("Square: ", rdd.map(lambda x: x*x).collect())
print("Sum2: ", rdd.reduce(lambda x, y: x + y))


