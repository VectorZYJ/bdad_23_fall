from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as Fsum, avg
from pyspark.sql.window import Window

# Initialize SparkSession
spark = SparkSession.builder.appName("YahooFinanceData").config("spark.executor.memory", "4g").getOrCreate()

# Read the merged CSV data into a DataFrame
csv_file_path = "merged.csv"
mergedDF = spark.read.csv(csv_file_path, header=True, inferSchema=True)

# Add a column for 'Close' * 'Volume'
mergedDF = mergedDF.withColumn("Close_Volume", col('Close') * col('Volume'))

# Define window specification
windowSpec = Window.partitionBy("Date", "Stock")

# Calculate the VWAP for each stock
vwap_df = mergedDF.withColumn("VWAP", Fsum('Close_Volume').over(windowSpec) / Fsum('Volume').over(windowSpec))

# Calculate the average VWAP for each sector each day
sector_index_df = vwap_df.groupBy("Date", "Category").agg(avg("VWAP").alias("Sector_VWAP"))

# Calculate moving average of Sector_VWAP for stability
moving_window = Window.partitionBy("Category").orderBy("Date").rowsBetween(-5, 0)  # 5-day moving average
moving_avg_df = sector_index_df.withColumn("MovingAvg_VWAP", avg("Sector_VWAP").over(moving_window))

# Select only the desired columns
selected_columns_df = moving_avg_df.select("Date", "Category", "MovingAvg_VWAP")

# Show the DataFrame with selected columns
selected_columns_df.show()

# Save the DataFrame with selected columns to CSV
output_path = "./sectors_with_vwap.csv"
selected_columns_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

# Describe the DataFrame with selected columns
selected_columns_df.describe().show()

# Stop SparkSession
spark.stop()
