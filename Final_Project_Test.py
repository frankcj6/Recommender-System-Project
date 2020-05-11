'''Usage:

    $ spark-submit Final_Project_Test.py test_data_path index_path model_path

'''
# Get Command Line
import sys

# add pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator


def main(spark, test_file, index_file, model_file):
    # load test data and create dataframe
    test_df = spark.read.parquet(test_file)
    model_indexer = PipelineModel.load(index_file)
    # transform user and track ids for test data
    test_df = model_indexer.transform(test_df)
    # store ground truth for user
    user_truth = test_df.groupby('user_label').agg(F.collect_list('book_label').alias('truth'))
    print('created ground truth df')
    als_model = ALSModel.load(model_file)

    # predict based on the top 500 item of each user
    recommend = als_model.recommendForAllUsers(500)
    print('recommendation has been created.')
    # RMSE
    predict = als_model.transform(test_df)
    evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
    rmse = evaluator.evaluate(predict)
    print('Root mean square error is '+str(rmse))

    # prediction = spark.sql('SELECT * FROM recommend INNER JOIN user_truth WHERE recommend.user_label=user_truth.user_label')
    # after running panda udf is faster than using sparksql
    prediction = recommend.join(user_truth, recommend.user_label == user_truth.user_label, 'inner')

    score = prediction.select('recommendations.book_label', 'truth').rdd.map(tuple)
    rank_metric = RankingMetrics(score)
    precision = rank_metric.precisionAt(500)
    mean_precision = rank_metric.meanAveragePrecision
    print(' precision at 500 ' + str(precision) + 'mean average precision of ' + str(mean_precision))


if __name__ == '__main__':
    # Create spark session object
    memory = '15g'
    spark = (SparkSession.builder
                         .appName('Recommend Test')
                         .config('spark.executor.memory', memory)
                         .config('spark.driver.memory', memory)
                         .config('spark.executor.memoryOverhead', '4096')
                         .config("spark.sql.broadcastTimeout", "36000")
                         .config("spark.storage.memoryFraction", "0")
                         .config("spark.memory.offHeap.enabled", "true")
                         .config("spark.memory.offHeap.size", "16g")
                         .getOrCreate())
    # Get filename from command line
    test_file = sys.argv[1]
    index_file = sys.argv[2]
    model_file = sys.argv[3]
    print('evaluating recommender using test dataset.')
    main(spark, test_file, index_file, model_file)
