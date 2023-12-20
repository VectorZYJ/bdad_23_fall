from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import pandas as pd

spark = SparkSession.builder.appName("RegressionExample").getOrCreate()

data = spark.read.csv("aa.csv", header=True, inferSchema=True)

# assembler = VectorAssembler(inputCols=["avg_sentiment", "avg_subjectivity", "normalized_avg_tone", "normalized_std_tone", "normalized_median_tone"], outputCol="features")\
assembler = VectorAssembler(inputCols=["avg_sentiment", "avg_subjectivity"], outputCol="features")

data = assembler.transform(data)

final_data = data.select("features", "open_close_diff")

train_data, test_data = final_data.randomSplit([0.7, 0.3])

lr = LinearRegression(featuresCol='features', labelCol='open_close_diff')

lr_model = lr.fit(train_data)

predictions = lr_model.transform(test_data)

# evaluator = RegressionEvaluator(labelCol="Normalized_Sector_VWAP", predictionCol="prediction", metricName="rmse")
evaluator = RegressionEvaluator(labelCol="open_close_diff", predictionCol="prediction", metricName="r2")

rmse = evaluator.evaluate(predictions)
evaluator = RegressionEvaluator(labelCol="open_close_diff", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
print("R2 on test data = %g" % r2)
pandas_df = predictions.select("prediction", "open_close_diff").toPandas()

plt.scatter(pandas_df['open_close_diff'], pandas_df['prediction'])
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()
