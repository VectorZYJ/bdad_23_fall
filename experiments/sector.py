from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, rank
from pyspark.sql.window import Window
from pyspark.sql.functions import min as Fmin, max as Fmax

# Initialize SparkSession
spark = SparkSession.builder.appName("YahooFinanceData").config("spark.executor.memory", "4g").getOrCreate()

# Read the merged CSV data into a DataFrame
csv_file_path = "merged.csv"
mergedDF = spark.read.csv(csv_file_path, header=True, inferSchema=True)

# Calculate the average of Open, High, Low, Close for each stock
average_price_df = mergedDF.withColumn("Average_Price", (col('Open') + col('High') + col('Low') + col('Close')) / 4)

# Calculate the average of Average_Price for each sector each day
sector_index_df = average_price_df.groupBy("Date", "Category").agg(avg("Average_Price").alias("Sector_Average_Price"))

# Create a Window specification for ranking within each sector
windowSpecCategory = Window.partitionBy("Category").orderBy("Category")

# Calculate the rank for each category
ranked_by_category_df = sector_index_df.withColumn("Category_Rank", rank().over(windowSpecCategory))

# Create a Window specification for ranking dates within each sector
windowSpecDate = Window.partitionBy("Category").orderBy("Date")

# Calculate the rank for dates within each sector
ranked_by_date_df = ranked_by_category_df.withColumn("Date_Rank", rank().over(windowSpecDate))

# Window specification for the entire DataFrame to calculate min and max rank values for Date_Rank
windowSpecAll = Window.partitionBy()

# Calculate the min and max values of Date_Rank
min_rank = ranked_by_date_df.select(Fmin("Date_Rank").over(windowSpecAll)).first()[0]
max_rank = ranked_by_date_df.select(Fmax("Date_Rank").over(windowSpecAll)).first()[0]

# Normalize the Date Rank
normalized_rank_df = ranked_by_date_df.withColumn("Normalized_Date_Rank", (col("Date_Rank") - min_rank) / (max_rank - min_rank))

# Select only the desired columns
selected_columns_df = normalized_rank_df.select("Date", "Category", "Sector_Average_Price", "Category_Rank", "Normalized_Date_Rank")

# Show the DataFrame with selected columns
selected_columns_df.show()

# Save the DataFrame with selected columns to CSV
output_path = "./sectors_with_ranked_average_price.csv"
selected_columns_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

# Describe the DataFrame with selected columns
selected_columns_df.describe().show()

# Stop SparkSession
spark.stop()
