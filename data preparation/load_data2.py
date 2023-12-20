from pyspark.sql import SparkSession
from pyspark.sql.functions import when, lower, regexp_replace, col, concat_ws, udf, count, avg, min, max, to_date, year

spark = SparkSession.builder.appName("Join CSV Files").getOrCreate()
df1 = spark.read.csv("summary_sentiment_by_date_category.csv", header=True, inferSchema=True)
df2 = spark.read.csv("custome.csv", header=True, inferSchema=True)
df1 = df1.withColumn("category", lower(col("category")))
df2 = df2.withColumn("category", lower(col("category")))

df1_alias = df1.alias("df1")
df2_alias = df2.alias("df2")

joined_df = df1_alias.join(df2_alias, (df1_alias.date == df2_alias.date) & (df1_alias.category == df2_alias.category), "inner")
final_merged_df = joined_df.drop(df1_alias.date).drop(df1_alias.category)
final_merged_df = final_merged_df.select("date", "avg_sentiment", "avg_subjectivity", "Open", "Close")
final_merged_df = final_merged_df.orderBy("date")
final_merged_df = final_merged_df.filter(year("date") == 2016)
final_merged_df = final_merged_df.filter((final_merged_df.avg_sentiment != 0) | (final_merged_df.avg_subjectivity != 0))
final_merged_df = final_merged_df.withColumn("open_close_diff", final_merged_df.Open - final_merged_df.Close)

final_merged_df.show()
# final_merged_df.write.mode("overwrite").csv("path_to_output_csv_file.csv")

final_merged_df = final_merged_df.toPandas()
final_merged_df.to_csv("aa.csv", header=True)
