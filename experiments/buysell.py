from pyspark.sql import SparkSession
from pyspark.sql.functions import lead, col
from pyspark.sql.window import Window

# Initialize SparkSession
spark = SparkSession.builder.appName("SectorIndexAnalysis").config("spark.executor.memory", "4g").getOrCreate()

# Read the CSV data into a DataFrame
csv_file_path = "sectorIndex.csv"
sector_df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

# Define a window specification ordered by Date
windowSpec = Window.partitionBy("Category").orderBy("Date")

# Use the lead function to get the next day's sectorIndex
lead_col = lead("sectorIndex", 1).over(windowSpec)

# Calculate the difference between the next day's and current day's sectorIndex
sector_df = sector_df.withColumn("Next_Day_SectorIndex", lead_col)
sector_df = sector_df.withColumn("Difference", col("Next_Day_SectorIndex") - col("sectorIndex"))

# Convert the difference into a binary signal (1 for buy, 0 for sell)
sector_df = sector_df.withColumn("BuySell", (col("Difference") > 0).cast("integer"))

# Drop the intermediate column
sector_df = sector_df.drop("Next_Day_SectorIndex", "Difference")

# Save the DataFrame with the new column to CSV
output_path = "combine_newid_binary.csv"
sector_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

# Stop SparkSession
spark.stop()
