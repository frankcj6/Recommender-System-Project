"""Usage:
    $ spark-submit path_test_file path_index_file path_model_file, limit

"""

# Get commend line
import sys

# Use pyspark.sql to get the spark session
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import functions as F
from pyspark.sql import Row
from annoy import AnnoyIndex
from time import time


def main(spark, sc, test_file, index_file, model_file, limit=0.01):
    # Load the dataframe
    test_df = spark.read.parquet(test_file)
    model_indexer = PipelineModel.load(index_file)
    # transform test_df using index_model
    test_df = model_indexer.transform(test_file)
    # select distinct user for recommendation, limit to save run time
    test_user = test_df.select('user_label').distinct().alias('userCol').sample(limit)
    # establish user_truth
    user_truth = test_df.groupby('user_label').agg(F.collect_list('book_label').alias('truth'))
    print('test data and user_truth has been preprocessed')
    # load als model
    als_model = ALSModel.load(model_file)

    # default settings
    baseline(als_model, user_truth, test_user)
    annoy(als_model, user_truth, test_user, sc)

    # hyper-parameter tunning:
    trees = [10, 15, 20]
    k_list = [-1, 5, 10]

    for i in trees:
        for j in k_list:
            annoy(als_model, user_truth, test_user, sc, n_trees=i, search_k=j)
    print('fast search feature has been established')


def baseline(als_model, user_truth, test_user):
    print('creating baseline model')
    time_start = time()
    recommend = als_model.recommendForUserSubset(test_user, 500)
    print('recommendation has been created.')
    predictions = recommend.join(user_truth, recommend.user_label == user_truth.user_label, 'inner')

    score = predictions.select('recommendations.book_label', 'truth').rdd.map(tuple)
    metrics = RankingMetrics(score)
    precision = metrics.precisionAt(500)
    mean_average_precision = metrics.meanAveragePrecision
    print('time taken: ' + str(time() - time_start))
    print('precision at 500: ' + str(precision))
    print('mean average precision: ' + str(mean_average_precision))


def annoy(als_model, user_truth, test_user, sc, n_trees=10, search_k=-1):
    print('creating annoy baseline with n_trees: ' + str(n_trees), 'search_k: ' + str(search_k))
    sc = SparkContext.getOrCreate()
    factors = als_model.userFactors
    size = factors.limit(1).select(F.size('features').alias('calculation')).collect()[0].calculation
    time_start = time()
    annoy_list = AnnoyIndex(size)
    for row in factors.collect():
        annoy_list.add_item(row.id, row.features)
    annoy_list.build(n_trees)
    annoy_list.save('./home/hj1325/final-project-final-project/annoy_list' + str(n_trees) + '_k_' + str(search_k) +
                    '.ann')
    recommend_list = [(user.user_label, annoy_list.get_nns_by_item(int(user.user_label), 500)) for user in
                      test_user.collect()]
    temp = sc.parallelize(recommend_list)
    print('recommendations has been created')
    recommend = spark.createDataFrame(temp, ['user_label', 'recommendation'])
    predictions = recommend.join(user_truth, recommend.user_label == user_truth.user_label, 'inner')

    score = predictions.select('recommendation', 'truth').rdd.map(tuple)
    metrics = RankingMetrics(score)
    precision = metrics.precisionAt(500)
    mean_average_precision = metrics.meanAveragePrecision
    print('time taken: ' + str(time() - time_start))
    print('precision at 500: ' + str(precision))
    print('mean average precision: ' + str(mean_average_precision))
    annoy_list.unload()


# only enter this block if we are in main
if __name__ == '__main__':

    # create spark session
    memory = '15g'
    spark = (SparkSession.builder
                         .appName('fast_search')
                         .config('spark.executor.memory', memory)
                         .config('spark.driver.memory', memory)
                         .config('spark.executor.memoryOverhead', '4096')
                         .config("spark.sql.broadcastTimeout", "36000")
                         .config("spark.storage.memoryFraction", "0")
                         .config("spark.memory.offHeap.enabled", "true")
                         .config("spark.memory.offHeap.size", "16g")
                         .getOrCreate())

    sc = SparkContext.getOrCreate()

    # Get file from commend line
    test_file = sys.argv[1]
    index_file = sys.argv[2]
    model_file = sys.argv[3]

    try:
        limit = sys.argv[4]

    except:
        limit = 0.01

    # call main routine
    main(spark, sc, test_file, index_file, model_file, limit)
