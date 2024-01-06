from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("SimpleSparkExample").getOrCreate()

# Create a Spark DataFrame from a list of tuples
data = [("Alice", 25), ("Bob", 30), ("Charlie", 22)]
columns = ["Name", "Age"]

df = spark.createDataFrame(data, columns)

# Show the DataFrame
print("Initial DataFrame:")
df.show()

# Perform a simple transformation: add 5 to each age
df_transformed = df.withColumn("AgePlusFive", df["Age"] + 5)

print("Transformed DataFrame:")
df_transformed.show()

# Perform a simple aggregation: calculate the average age
average_age = df.agg({"Age": "avg"}).collect()[0]["avg(Age)"]

print(f"Average Age: {average_age}")

# Stop the Spark session
spark.stop()