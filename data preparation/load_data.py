from textblob import TextBlob
from datasets import load_dataset
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, lower, regexp_replace, col, concat_ws, udf, count, avg, min, max, to_date
from pyspark.sql.types import StringType, DoubleType
import pandas as pd
import re
from textblob import Word

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)

    contractions = {"can't": "cannot", "won't": "will not"}
    text = ' '.join([contractions[t] if t in contractions else t for t in text.split()])

    text = re.sub(r'[^a-zA-Z\' ]', '', text)

    text = text.lower()

    text = re.sub(r'\s+', ' ', text).strip()

    text = ' '.join([Word(word).lemmatize() for word in text.split()])

    return text

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def analyze_subjectivity(text):
    analysis = TextBlob(text)
    return analysis.sentiment.subjectivity

def normalize(gdelt_df):
    min_max_df = gdelt_df.agg(
    min("avg_tone").alias("min_avg_tone"),
    max("avg_tone").alias("max_avg_tone"),
    min("std_tone").alias("min_std_tone"),
    max("std_tone").alias("max_std_tone"),
    min("median_tone").alias("min_median_tone"),
    max("median_tone").alias("max_median_tone")
)

    # Collect the min and max values
    min_max_values = min_max_df.collect()[0]

    # Normalize each column
    gdelt_df = gdelt_df.withColumn("normalized_avg_tone",
                                (col("avg_tone") - min_max_values["min_avg_tone"]) /
                                (min_max_values["max_avg_tone"] - min_max_values["min_avg_tone"])
                                ).drop("avg_tone")

    gdelt_df = gdelt_df.withColumn("normalized_std_tone",
                                (col("std_tone") - min_max_values["min_std_tone"]) /
                                (min_max_values["max_std_tone"] - min_max_values["min_std_tone"])
                                ).drop("std_tone")

    gdelt_df = gdelt_df.withColumn("normalized_median_tone",
                                (col("median_tone") - min_max_values["min_median_tone"]) /
                                (min_max_values["max_median_tone"] - min_max_values["min_median_tone"])
                                ).drop("median_tone")
    return gdelt_df


spark = SparkSession.builder.appName("HuffPost").getOrCreate()

dataset = load_dataset("khalidalt/HuffPost")
pandas_df = dataset["test"].to_pandas()
spark_df = spark.createDataFrame(pandas_df)

for column in spark_df.columns:
    spark_df.select(column).describe().show()

columns_to_drop = ['link', 'authors']
df_cleaned = spark_df.drop(*columns_to_drop)

date_regex_pattern = "^\\d{4}-\\d{2}-\\d{2}$"
df_cleaned = df_cleaned.filter(col("date").rlike(date_regex_pattern))

clean_text_udf = udf(clean_text, StringType())
analyze_sentiment_udf = udf(analyze_sentiment, DoubleType())
analyze_subjectivity_udf = udf(analyze_subjectivity, DoubleType())

df_concatenated = df_cleaned.withColumn("headline_description", 
                                        concat_ws(" ", 
                                        col("headline"), 
                                        col("short_description")))

df_cleaned_text = df_concatenated.withColumn("cleaned_text", clean_text_udf(col("headline_description")))

df_with_analysis = df_cleaned_text.withColumn("sentiment", analyze_sentiment_udf(col("headline_description")))
df_with_analysis = df_with_analysis.withColumn("subjectivity", analyze_subjectivity_udf(col("headline_description")))


pandas_df_with_analysis = df_with_analysis.toPandas()
pandas_df_with_analysis.to_csv("result.csv", header=True)

tech_summary_df = df_with_analysis.withColumn("category", lower(col("category"))) \
                         .groupBy("date", "category") \
                         .agg(avg("sentiment").alias("avg_sentiment"),
                              avg("subjectivity").alias("avg_subjectivity"))

tech_summary_df = tech_summary_df.filter(col("category") == "tech")

# gdelt_df = spark.read.csv("GDELT_2013_to_2018.csv", header=True, inferSchema=True)
sector_vwap_df = spark.read.csv("sector_vwap.csv", header=True, inferSchema=True)
# gdelt_df = normalize(gdelt_df)

# tech_summary_df = tech_summary_df.join(gdelt_df, on="date", how="inner")
sector_vwap_df = sector_vwap_df.withColumn("category", lower(col("category")))
sector_vwap_df = sector_vwap_df.filter(col("category") == "tech")

final_merged_df = tech_summary_df.join(sector_vwap_df, (tech_summary_df.date == sector_vwap_df.date) & (tech_summary_df.category == sector_vwap_df.category), how="inner")

# final_merged_df = final_merged_df.drop(sector_vwap_df.date).drop(sector_vwap_df.category).drop(sector_vwap_df.Sector_VWAP).drop(gdelt_df.num).drop(tech_summary_df.category)
final_merged_df = final_merged_df.drop(sector_vwap_df.date).drop(sector_vwap_df.category).drop(sector_vwap_df.Sector_VWAP).drop(tech_summary_df.category)

final_merged_df = final_merged_df.withColumn("date", to_date(final_merged_df.date, "yyyy-MM-dd"))

final_merged_df = final_merged_df.orderBy("date")
final_merged_df.show()

final_merged_df.write.mode("overwrite").csv("final_merged_df.csv")
