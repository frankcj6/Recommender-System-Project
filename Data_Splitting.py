'''Usage:

    $ spark-submit Data_Splitting.py data_file_path

'''

# Get Command Line
import sys

# add pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def main(spark, data_file):
    # save csv file to parquet in advance
    # goodreads_interaction = spark.read.csv('hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv',
    # schema='user_id INT,book_id INT, is_read INT,rating INT,is_reviewed INT')
    # user_id_map = spark.read.csv('hdfs:/user/bm106/pub/goodreads/user_id_map.csv',
    # schema='user_id_csv INT, user_id STRING')
    # book_id_map = spark.read.csv('hdfs:/user/bm106/pub/goodreads/book_id_map.csv',
    # schema='book_id_csv INT, book_id STRING')
    # goodreads_interaction.createOrReplaceTempView
    # user_id_map.createOrReplaceTempView
    # book_id_map.createOrReplaceTempView
    # goodreads_interaction.write.parquet('final-project-final-project/goodreads_interaction.parquet')
    # user_id_map.write.parquet('final-project-final-project/user_id_map.parquet')
    # book_id_map.write.parquet('final-project-final-project/book_id_map.parquet')

    # split dataframe to training, validation, testing (0.6,0.2,0.2)
    goodreads_interaction = spark.read.parquet(data_file)
    split_list = goodreads_interaction.randomSplit([0.6, 0.2, 0.2], seed=20)
    train_df = split_list[0]
    validation_df = split_list[1]
    test_df = split_list[2]
    # add row number into validation dataframe
    window = Window.partitionBy('user_id').orderBy('book_id')
    validation_df = (validation_df.select("user_id", "book_id", "is_read", "rating", "is_reviewed",
                                          F.row_number().over(window).alias("row_number")))
    # split validation dataframe according to even and odd row number
    validation_even = validation_df.filter(validation_df.row_number % 2 == 0)
    validation_odd = validation_df.filter(validation_df.row_number % 2 != 0)

    # even to train df, odd to validation df
    train_df = train_df.union(validation_even.select('user_id', 'book_id', 'is_read', 'rating', 'is_reviewed'))
    validation_df = validation_odd.select('user_id', 'book_id', 'is_read', 'rating', 'is_reviewed')

    # add row number into test dataframe
    window = Window.partitionBy('user_id').orderBy('book_id')
    test_df = (test_df.select('user_id', 'book_id', 'is_read', 'rating', 'is_reviewed',
                              F.row_number().over(window).alias('row_number')))

    # split test dataframe according to even and odd row number
    test_even = test_df.filter(test_df.row_number % 2 == 0)
    test_odd = test_df.filter(test_df.row_number % 2 != 0)

    # even to train df, odd to test df
    train_df = train_df.union(test_even.select('user_id', 'book_id', 'is_read', 'rating', 'is_reviewed'))
    test_df = test_odd.select('user_id', 'book_id', 'is_read', 'rating', 'is_reviewed')

    # save file to hdfs
    train_df.createOrReplaceTempView('train_df')
    validation_df.createOrReplaceTempView('validation_df')
    test_df.createOrReplaceTempView('test_df')

    train_df.write.parquet('hdfs:/user/hj1325/final-project-final-project/train_df.parquet')
    validation_df.write.parquet('hdfs:/user/hj1325/final-project-final-project/validation_df.parquet')
    test_df.write.parquet('hdfs:/user/hj1325/final-project-final-project/test_df.parquet')


if __name__ == '__main__':
    # Create spark session object
    spark = SparkSession.builder.appName('Split Data').getOrCreate()
    # Get filename from command line
    data_file = sys.argv[1]

    main(spark, data_file)
