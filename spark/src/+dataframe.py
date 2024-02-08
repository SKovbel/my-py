from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("SimpleSparkExample").getOrCreate()

# Create a Spark DataFrame from a list of tuples
df = spark.createDataFrame([
        ("Alice", 25),
        ("Bob", 30),
        ("Charlie", 22)
    ], 
    ["Name", "Age"])


print("Initial DataFrame:")
df.show()


df_transformed = df.withColumn("AgePlusFive", df["Age"] + 5)
print("Transformed DataFrame:")
df_transformed.show()

# simple aggregation: calculate the average age
average_age = df.agg({"Age": "avg"}).collect()[0]["avg(Age)"]
print(f"Average Age: {average_age}")



spark.stop()

