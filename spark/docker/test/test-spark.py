# Import the necessary modules
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Create a SparkSession
spark = SparkSession.builder \
   .appName("First Onex") \
   .master("spark://0.0.0.0:7077") \
   .getOrCreate()


dept = [("Finance",10),("Marketing",20),("Sales",30),("IT",40)]
rdd = spark.sparkContext.parallelize(dept)
df = rdd.toDF()
print(df.printSchema(), df.collect(), df.show(truncate=False))

rdd = spark.sparkContext.parallelize(range(1, 100))
print("Sum: ", rdd.sum())
print("Square: ", rdd.map(lambda x: x*x).collect())
print("Sum2: ", rdd.reduce(lambda x, y: x + y))


df = spark.createDataFrame( [(i, 2*i) for i in range(1, 100)], ["A", "B"])
df = df.withColumn("C", df["A"] + 5)
sum = df.agg({"A": "sum", "C": "sum"}).collect()
print(df.collect(), "Sum", sum)


df = spark.createDataFrame([(-2, 2)], ('C1', 'C2'))
query = df.select(sequence('C1', 'C2').alias('r'))
print(query.collect())



spark.stop()
