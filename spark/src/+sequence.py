from pyspark.sql import SparkSession
from operator import add
from pyspark.sql.functions import sequence

spark = SparkSession.builder.appName("HelloWorld").getOrCreate()

df = spark.createDataFrame([(-2, 2)], ('C1', 'C2'))
df.select(sequence('C1', 'C2').alias('r')).collect()
print(df)
