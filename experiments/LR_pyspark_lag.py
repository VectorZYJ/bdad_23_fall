from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder.appName("Stock Analysis with Lags in PySpark").getOrCreate()

# Load the dataset
file_path = 'updated_data_with_close_open.csv'
data = spark.read.csv(file_path, header=True, inferSchema=True)

# Convert 'Date' to timestamp and sort by date
data = data.withColumn('Date', col('Date').cast('timestamp')).orderBy('Date')

# Lag values to test
lag_values = list(range(1, 31)) + [30, 90, 120]

# Base columns for features
base_cols = ['avg_articles_count', 'Momentum', 'avg_subjectivity']

# Dictionary to store metrics for each lag value
lag_metrics = []

for lag_value in lag_values:  # Renamed variable to avoid conflict
    print("Running for lag: ", lag_value)
    total_r2 = 0
    total_mse = 0
    num_stocks = 0

    # Iterate over each stock and perform the analysis
    stocks = data.select('Stock').distinct().rdd.flatMap(lambda x: x).collect()

    for stock in stocks:
        # Filter data for the current stock
        stock_data = data.filter(data['Stock'] == stock)

        # Define the window specification
        windowSpec = Window.partitionBy('Stock').orderBy('Date')

        # Create lagged features
        for i in range(1, lag_value + 1):  # Adjusted to use lag_value
            stock_data = stock_data.withColumn(f'Close-Open_Lag_{i}', lag(col('Close-Open'), i).over(windowSpec))

        # Drop rows with NaN values (due to lagging)
        stock_data = stock_data.na.drop()

        # Prepare features for Linear Regression
        lag_cols = [f'Close-Open_Lag_{i}' for i in range(1, lag_value + 1)]
        assembler = VectorAssembler(inputCols=base_cols + lag_cols, outputCol="features")
        scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

        # Linear Regression
        lr = LinearRegression(featuresCol='scaledFeatures', labelCol='Close-Open')
        pipeline = Pipeline(stages=[assembler, scaler, lr])

        # Split the data
        (trainingData, testData) = stock_data.randomSplit([0.8, 0.2])

        # Train the model
        model = pipeline.fit(trainingData)

        # Make predictions
        predictions = model.transform(testData)

        # Evaluation Metrics
        evaluator = RegressionEvaluator(labelCol='Close-Open', metricName='r2')
        r2 = evaluator.evaluate(predictions)
        evaluator.setMetricName('mse')
        mse = evaluator.evaluate(predictions)

        # Accumulate the R2 and MSE
        total_r2 += r2
        total_mse += mse
        num_stocks += 1

    # Calculate the average R2 and average MSE for the current lag value
    avg_r2 = total_r2 / num_stocks
    avg_mse = total_mse / num_stocks

    # Store metrics in the list
    lag_metrics.append((lag_value, avg_mse, avg_r2))  # Adjusted to use lag_value
    print('Lag:', lag_value, 'MSE:', avg_mse, 'R2:', avg_r2)

# Convert lag_metrics to a DataFrame
lag_metrics_df = pd.DataFrame(lag_metrics, columns=['Lag', 'MSE', 'R2'])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(lag_metrics_df['Lag'], lag_metrics_df['R2'], label='R2')
plt.plot(lag_metrics_df['Lag'], lag_metrics_df['MSE'], label='MSE')
plt.xlabel('Lag Value')
plt.ylabel('Metric Value')
plt.title('Impact of Lag Value on R2 and MSE')
plt.legend()
plt.grid(True)
plt.show()

# Stop the Spark session
spark.stop()
