from pyspark.sql import SparkSession
from operator import add
 
def init():
  spark = SparkSession.builder.appName("HelloWorld").getOrCreate()
  sc = spark.sparkContext
  return spark,sc

spark, sc = init()

data = sc.parallelize(list("Hello World"))
counts = data.map(lambda x: 
	(x, 1)).reduceByKey(add).sortBy(lambda x: x[1],
	 ascending=False).collect()

for (word, count) in counts:
    print("{}: {}".format(word, count))
