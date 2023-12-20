from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, row_number
from pyspark.sql.window import Window

# Initialize SparkSession
spark = SparkSession.builder.appName("RankByCategoryAndDate").getOrCreate()

# Read the input CSV file into a DataFrame
input_path = "firststep.csv"  # Replace with your input file path
input_df = spark.read.csv(input_path, header=True, inferSchema=True)

# Group by Date and Category, calculate the average of avg_sentiment and avg_subjectivity
result_df = input_df.groupBy("Date", "Category").agg(
    avg("avg_sentiment").alias("avg_sentiment"),
    avg("avg_subjectivity").alias("avg_subjectivity"),
    avg("articles_count").alias("avg_articles_count"),  # Average articles_count
    # avg("Sector_VWAP").alias("avg_Sector_VWAP")  # Average Sector_VWAP
)

# Create a window specification to define the ranking order
window_spec = Window.partitionBy("Category").orderBy("Date")

# Add a new column with the rank within each category and date
result_df = result_df.withColumn("Rank", row_number().over(window_spec))

# Show the resulting DataFrame with ranking
result_df.show()

# Save the result to a CSV file if needed
output_path = "output.csv"  # Replace with your desired output path
result_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)


# Stop SparkSession
spark.stop()
