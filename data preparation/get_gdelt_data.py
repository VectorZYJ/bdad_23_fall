import gdelt
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, variance, count, median


def main():
    global spark
    spark = SparkSession.builder.appName("DataIngestion").getOrCreate()
    get_data_after_2013()
    get_data_2012_and_2013()


def get_data_2012_and_2013():
    dates = ['201201', '201202', '201203', '201204', '201205', '201206', '201207',
             '201208', '201209', '201210', '201211', '201212', '201301', '201302', '201303']
    for date in dates:
        df = get_data(date)
        df = df[df['ActionGeo_CountryCode'] == 'US']
        df = df[['Day', 'AvgTone']]
        df['AvgTone'].astype('double')
        df.to_csv('temp.csv', index=False)
        spark_process('temp.csv')


def spark_process(filename):
    daily_df = (spark.read.csv(filename, header=True)
                .withColumnRenamed('Day', 'Date')
                .withColumnRenamed('AvgTone', 'Tone')
                .withColumn('Tone', col('Tone').cast('double'))
                )
    stat_df = (daily_df.groupby('Day')
               .agg(count('Tone').alias('num'),
                    avg('Tone').alias('avg_tone'),
                    variance('Tone').alias('std_tone'),
                    median('Tone').alias('median_tone'))
               .withColumnRenamed('Day', 'Date')
               )
    stat_df.write.option('header', 'true').csv('filename.csv')


def get_data_after_2013():
    data = []
    for month in range(4, 13):
        days = days_of_month(2013, month)
        for day in range(1, days+1):
            daily = get_daily_data(2013, month, day)
            data.append(daily)

    start, end = 2014, 2018
    for year in range(start, end + 1):
        for month in range(1, 13):
            days = days_of_month(year, month)
            for day in range(1, days + 1):
                daily = get_daily_data(year, month, day)
                data.append(daily)

    df = pd.DataFrame(data, columns=['Date', 'num', 'avg_tone', 'std_tone', 'median_tone'])
    df.to_csv('2013.41-2018.csv', index=False)


def get_daily_data(year, month, day):
    date = gen_date(year, month, day)
    num, average, stderr, mid = 0, 0, 0, 0
    try:
        df = get_data(date)
    except ValueError:
        num, average, stderr, mid = 0, 0, 0, 0
    else:
        df = df[df['ActionGeo_CountryCode'] == 'US']
        df = df[['DATEADDED', 'AvgTone', 'SOURCEURL']]
        df = df.drop_duplicates(['SOURCEURL'])
        df.drop('SOURCEURL', axis=1, inplace=True)
        df['AvgTone'].astype('double')
        num, average, stderr, mid = len(df['AvgTone']), df['AvgTone'].mean(), df['AvgTone'].std(), df[
            'AvgTone'].median()
    finally:
        daily = ['-'.join(date.split(' ')), num, average, stderr, mid]

    return daily


def gen_date(year, month, day):
    year = str(year)
    month = str(month) if month >= 10 else '0' + str(month)
    day = str(day) if day >= 10 else '0' + str(day)
    date = f"{year} {month} {day}"
    return date


def days_of_month(year, month):
    if month == 4 or month == 6 or month == 9 or month == 11:
        return 30
    elif month == 2:
        return 28 + (year % 100 != 0 and year % 4 == 0 or year % 100 == 0 and year % 400 == 0)
    else:
        return 31


def get_data(date):
    print(date)
    gd = gdelt.gdelt(version=1)
    results = gd.Search(date, table='events')
    return results


if __name__ == '__main__':
    main()
