# Import the necessary modules
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Create a SparkSession
spark = SparkSession.builder \
   .appName("First Onex") \
   .master("spark://0.0.0.0:7077") \
   .getOrCreate()

context = spark.sparkContext

tables = spark.sql("SHOW TABLES")
tables.show()



#textFile = spark.read.text("docker-compose-full.yml")
#print('A', textFile.value, 'B')
#linesWithSpark = textFile.filter(textFile.value.contains(":"))
#linesWithSpark.cache()
#print(textFile.count())


dept = [("Finance",10),("Marketing",20),("Sales",30),("IT",40)]
rdd = context.parallelize(dept)
df = rdd.toDF()
print(df.printSchema(), df.collect(), df.show(truncate=False))


rdd = context.parallelize(range(1, 100))
print("Sum: ", rdd.sum())
print("Square: ", rdd.map(lambda x: x*x).collect())
print("Sum2: ", rdd.reduce(lambda x, y: x + y))


df = spark.createDataFrame( [(i, 2*i) for i in range(1, 100)], ["A", "B"])
df = df.withColumn("C", df["A"] + 5)
sum = df.agg({"A": "sum", "C": "sum"}).collect()
print(df.collect(), "Sum", sum)

df.write.mode("overwrite").saveAsTable("people")
result = spark.sql("""
    SELECT A, B
    FROM abc
    WHERE C > 30
""")
result.show()
exit(0)

df = spark.createDataFrame([(-2, 2)], ('C1', 'C2'))
df.cache()
query = df.select(sequence('C1', 'C2').alias('r'))
print(query.collect())


def myFunc(s):
   words = s.split(" ")
   return len(words)

context.setJobDescription('X')
rdd = context.parallelize(["word 1", "word 2", "word 3", "word 4 5"])
res = rdd.map(myFunc)
print(res.collect())

accum = context.accumulator(0)
context.parallelize([1, 2, 3, 4]).foreach(lambda x: accum.add(x))
print(accum.value)



spark.stop()
