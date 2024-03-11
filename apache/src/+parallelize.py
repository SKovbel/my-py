from pyspark.sql import SparkSession

def init():
  spark = SparkSession.builder.appName("HelloWorld").getOrCreate()
  sc = spark.sparkContext
  return spark,sc

spark, sc = init()

# Create an RDD from a Python list
data = [1,2,3,4,5]
rdd = sc.parallelize(data)


square_rdd = rdd.map(lambda x: x*x)
sum = square_rdd.reduce(lambda x, y: x + y)
print(square_rdd.collect(), sum)


# case 2
strings = ["a", "b", "c"]
result = sc.parallelize(strings, 2).glom().collect()
print(result)

result = sc.parallelize(strings, 12).glom().collect()
print(result)
