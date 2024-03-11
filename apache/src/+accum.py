from pyspark.sql import SparkSession
from operator import add
 
spark = SparkSession.builder.appName("HelloWorld").getOrCreate()
context = spark.sparkContext

accum = context.accumulator(0)
context.parallelize([1, 2, 3, 4]).foreach(lambda x: accum.add(x))
print(accum.value)
