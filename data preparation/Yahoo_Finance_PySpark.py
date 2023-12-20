import yfinance as yf
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, lit, format_number, date_format, round, avg
from pyspark.sql.window import Window
from pyspark.sql.functions import min as Fmin, max as Fmax, col


# Ticker symbols and their corresponding categories
stock_categories = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'FB': 'Tech', 'IBM': 'Tech',
    'NVDA': 'Tech', 'INTC': 'Tech', 'AMD': 'Tech', 'ORCL': 'Tech', 'SAP': 'Tech',

    'PFE': 'Pharmaceutical', 'JNJ': 'Pharmaceutical', 'MRK': 'Pharmaceutical', 'GSK': 'Pharmaceutical', 'NVS': 'Pharmaceutical',
    'ABBV': 'Pharmaceutical', 'BMY': 'Pharmaceutical', 'AMGN': 'Pharmaceutical', 'SNY': 'Pharmaceutical', 'LLY': 'Pharmaceutical',

    'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'C': 'Financial', 'GS': 'Financial',
    'MS': 'Financial', 'AXP': 'Financial', 'BRK.B': 'Financial', 'TFC': 'Financial', 'SPGI': 'Financial',

    'XOM': 'Oil', 'CVX': 'Oil', 'RDS.A': 'Oil', 'BP': 'Oil', 'TOT': 'Oil',
    'PBR': 'Oil', 'ENB': 'Oil', 'E': 'Oil', 'SLB': 'Oil', 'COP': 'Oil',

    'AMZN': 'Retail', 'WMT': 'Retail', 'HD': 'Retail', 'COST': 'Retail', 'TGT': 'Retail',
    'LOW': 'Retail', 'JD': 'Retail', 'EBAY': 'Retail', 'ORLY': 'Retail', 'BBY': 'Retail',

    'TSLA': 'Automotive', 'GM': 'Automotive', 'F': 'Automotive', 'HMC': 'Automotive', 'TM': 'Automotive',
    'VWAGY': 'Automotive', 'DAI.DE': 'Automotive', 'RACE': 'Automotive', 'TTM': 'Automotive', 'STLA': 'Automotive',

    'T': 'Telecommunications', 'VZ': 'Telecommunications', 'NOK': 'Telecommunications', 'ERIC': 'Telecommunications', 'CHT': 'Telecommunications',
    'CHA': 'Telecommunications', 'TLK': 'Telecommunications', 'TELIA.ST': 'Telecommunications', 'BCE': 'Telecommunications', 'S': 'Telecommunications',

    'BA': 'Aerospace', 'LMT': 'Aerospace', 'RTX': 'Aerospace', 'NOC': 'Aerospace', 'GD': 'Aerospace',
    'AIR.PA': 'Aerospace', 'BAE.L': 'Aerospace', 'HEI': 'Aerospace', 'COL': 'Aerospace', 'SAFRY': 'Aerospace',

    'SPG': 'Real Estate', 'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate', 'EQIX': 'Real Estate',
    'DLR': 'Real Estate', 'WELL': 'Real Estate', 'AVB': 'Real Estate', 'O': 'Real Estate', 'GPT': 'Real Estate',

    'DIS': 'Entertainment', 'NFLX': 'Entertainment', 'CMCSA': 'Entertainment', 'CHTR': 'Entertainment', 'LBRDK': 'Entertainment',
    'VIAC': 'Entertainment', 'LYV': 'Entertainment', 'FOXA': 'Entertainment', 'DISH': 'Entertainment', 'ROKU': 'Entertainment',

    'NKE': 'Apparel', 'UAA': 'Apparel', 'LULU': 'Apparel', 'PVH': 'Apparel', 'VFC': 'Apparel',
    'RL': 'Apparel', 'DECK': 'Apparel', 'CROX': 'Apparel', 'GIL': 'Apparel', 'SKX': 'Apparel'
}


spark = SparkSession.builder.appName("YahooFinanceData").config("spark.executor.memory", "4g").getOrCreate()

dataFrames = []

# Define the weights
weights = {
    "Open": 1,
    "High": 1,
    "Low": 1,
    "Close": 1,
    "Volume": 0.0000001  
}

factors = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Stock', 'Category']

for stockname, category in stock_categories.items():
    try:
        stockData = yf.Ticker(stockname)
        stockDf = stockData.history(period='1d', start='2012-1-28', end='2018-5-26')
        
        stockDf.reset_index(inplace=True)
        stockDf = stockDf.drop(columns=['Stock Splits', 'Dividends'])

        sparkDF = spark.createDataFrame(stockDf)
        sparkDF = sparkDF.withColumn("Stock", lit(stockname)).withColumn("Category", lit(category))
        
        for col_name in factors:
            if col_name not in sparkDF.columns:
                sparkDF = sparkDF.withColumn(col_name, lit(None))

        sparkDF = sparkDF.select(factors)

        dataFrames.append(sparkDF)
    except Exception as e:
        print(f"Error processing {stockname}: {e}")

mergedDF = dataFrames[0]
for df in dataFrames[1:]:
    mergedDF = mergedDF.unionByName(df)

for attribute, weight in weights.items():
    mergedDF = mergedDF.withColumn(attribute, (col(attribute) * weight).cast(DoubleType()))

for col_name in weights.keys():
    mergedDF = mergedDF.withColumn(col_name, format_number(col(col_name), 2).cast(DoubleType()))


mergedDF = mergedDF.withColumn('Date', date_format(col('Date'), 'yyyy-MM-dd'))
mergedDF = mergedDF.withColumn("Custom_Metric", col('Open') + col('High') + col('Low') + col('Close') + col('Volume'))
mergedDF = mergedDF.withColumn("Custom_Metric", round(sum([col(attr) for attr in weights.keys()]), 2))


# Calculate the VWAP for each stock
vwap_df = mergedDF.withColumn("VWAP", (col('Close') * col('Volume')) / col('Volume'))

# Calculate the average VWAP for each sector each day
sector_index_df = vwap_df.groupBy("Date", "Category").agg(avg("VWAP").alias("Sector_VWAP"))

# Show the sector index DataFrame
# sector_index_df.show()

# Window specification for the entire DataFrame
windowSpec = Window.partitionBy()

# Calculate the min and max values of Sector_VWAP
min_vwap = sector_index_df.select(Fmin("Sector_VWAP").over(windowSpec)).first()[0]
max_vwap = sector_index_df.select(Fmax("Sector_VWAP").over(windowSpec)).first()[0]

# Normalize the Sector_VWAP
normalized_sector_index_df = sector_index_df.withColumn("Normalized_Sector_VWAP", (col("Sector_VWAP") - min_vwap) / (max_vwap - min_vwap))

# Show the normalized sector index DataFrame
normalized_sector_index_df.show()

output_path = "./stock_sector.csv"
normalized_sector_index_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path) 
normalized_sector_index_df.describe()
normalized_sector_index_df.show()

output_path = "./stock_data_all.csv"
mergedDF.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path) 
mergedDF.describe()
mergedDF.show()

spark.stop()