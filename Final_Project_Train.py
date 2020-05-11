"""Part: Train model
Usage:

    $ spark-submit Final_Project_Train.py train_data_path validation_data_path test_data_path path_to_save_model
        tuning_option

"""
# Get command line
import sys

# Import pyspark.sql to get the Spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator


def main(spark, data_file, validation_file, test_file, model_file, tuning=False):
    # load data and create dataframe
    # train data
    train_df = spark.read.parquet(data_file)
    train_df.createOrReplaceTempView('train_df')
    # validation data
    validation_df = spark.read.parquet(validation_file)
    validation_df.createOrReplaceTempView('validation_df')
    # test data
    test_df = spark.read.parquet(test_file)
    test_df.createOrReplaceTempView('test_df')

    # omit data that not contains users in the validation and test data
    train_df = spark.sql(
        "SELECT DISTINCT(user_id), book_id, rating FROM train_df "
        "WHERE user_id IN ((SELECT user_id FROM validation_df) UNION (SELECT user_id FROM test_df)) AND rating!=0")

    # sub sample 60% of data
    (train_df, train_rest) = train_df.randomSplit([0.6, 0.4], seed=20)

    print('data has been preprocessed. ')

    try:
        # load saved Model Indexer. If haven't created, then create indexer
        print('load Model Indexer')
        model_indexer = PipelineModel.load('./home/hj1325/final-project-final-project/model_indexer')

    except:
        # create indexer
        print('create Model Indexer')
        user_indexer = StringIndexer(inputCol='user_id', outputCol='user_label').setHandleInvalid('skip')
        book_indexer = StringIndexer(inputCol='book_id', outputCol='book_label').setHandleInvalid('skip')
        training_pipeline = Pipeline(stages=[user_indexer, book_indexer])

        model_indexer = training_pipeline.fit(train_df)
        model_indexer.write().overwrite().save('./home/hj1325/final-project-final-project/model_indexer')
        print('Model indexer has been created.')


    # use indexer to transform dataframe for training and validation
    train_df = model_indexer.transform(train_df)
    validation_df = model_indexer.transform(validation_df)
    validation_user = validation_df.select('user_label').distinct().alias('userCol')

    validation_t_df = validation_df.select(
        ['user_label', 'book_label']).repartition(800, 'user_label')
    # use panda udf to save run time
    user_truth = validation_t_df.groupby('user_label').agg(F.collect_list('book_label').alias('truth')).cache()
    print('Training and Validation dataframe have been transformed.')

    # set tuning to true to tune using hyper-parameter, by default use the the following hyper-parameter to save running
    # time
    # regularization parameter = 0.1, alpha = 1, rank = 100(handling implicit feedback)
    if tuning:
        RegParam = [0.1, 1, 10, 100]
        Alpha = [0.1, 1]
        Rank = [10, 100]
    else:
        RegParam = [0.1]
        Alpha = [1]
        Rank = [100]

    # precision_at_k store precision and average corresponding to each regparam, alpha and rank
    PRECISION_AT_K = {}
    RMSE_list = {}
    count = 0
    total = len(RegParam) * len(Alpha) * len(Rank)

    for a in RegParam:
        for b in Alpha:
            for c in Rank:
                print('currently using model with regParam =' + str(a) + ', Alpha =' + str(b) + ', Rank =' + str(c))

                # use train_df to fit ALS model
                als_train = ALS(maxIter=10, regParam=a, alpha=b, rank=c, userCol='user_label',
                                itemCol='book_label', ratingCol='rating',
                                coldStartStrategy='drop', implicitPrefs=True)

                als_model = als_train.fit(train_df)

                # evaluate model
                predict = als_model.transform(validation_df)
                evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
                rmse = evaluator.evaluate(predict)
                RMSE_list[rmse] = [rmse, als_model, als_train]

                count += 1
                print(str(count) + 'of the total ' + str(total) + ' finished.')

                print(' RMSE value= '+str(rmse)+' RegParam= '+str(a)+' Alpha= '+str(b)+' rank= '+str(c))

                # predict based on the top 500 item of each user
                # recommend = als_model.recommendForUserSubset(validation_df, 500)

                # prediction = spark.sql('SELECT * FROM recommend INNER JOIN user_truth WHERE recommend.user_label=user_truth.user_label')
                # after running panda udf is faster than using sparksql
                # prediction = recommend.join(user_truth, recommend.user_label == user_truth.user_label, 'inner')

                #score = prediction.select('recommendations.book_label', 'truth')
                #score = score.rdd.map(tuple).repartition(800)
                #rank_metric = RankingMetrics(score)

                #mean_precision = rank_metric.meanAveragePrecision
                #precision = rank_metric.precisionAt(500)

                #PRECISION_AT_K[mean_precision] = [precision, als_model, als_train]
                #count += 1

                #print(str(count) + 'of the total' + str(total) + 'finished.')
                #print(str(precision) + str(mean_precision))

    # store model with the best root square mean error statistic
    best_rmse = min(list(RMSE_list.keys()))
    lowest_rmse, best_model, best_als_model = RMSE_list[best_rmse]
    best_model.write().overwrite().save(model_file)


    # store model with the best precision statistic
    #best_mean_precision = max(list(PRECISION_AT_K.keys()))
    #highest_precision, best_model, best_als_model = PRECISION_AT_K[best_mean_precision]
    #best_model.write().overwrite().save(model_file)

    # save best ALS model
    # best_als_model.save('./recommender/alsFile')

    #print('Best model with the mean average precision of' + str(best_mean_precision) +
          #'and the best precision of ' + str(highest_precision) +
          #'regParam=' + str(best_als_model.getregParam) +
          #'Alpha=' + str(best_als_model.getAlpha) +
          #'Rank=' + str(best_als_model.getRank))

    print('Best model with the root mean square error of ' + str(lowest_rmse) +
          ' and the regParam of ' + {best_als_model.getRegParam} +
          ' Alpha of ' + {best_als_model.getAlpha} +
          ' Rank of ' + {best_als_model.getRank})


# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object, reset the memory space to avoid spark run out of memory
    memory = '15g'
    spark = (SparkSession.builder
             .appName('Final_Project_Train')
             .config('spark.executor.memory', memory)
             .config('spark.driver.memory', memory)
             .config('spark.executor.memoryOverhead', '4096')
             .config("spark.sql.broadcastTimeout", "36000")
             .config("spark.storage.memoryFraction", "0")
             .config("spark.memory.offHeap.enabled", "true")
             .config("spark.memory.offHeap.size", "16g")
             .getOrCreate())

    # Get the filename from the command line
    data_file = sys.argv[1]

    validation_file = sys.argv[2]

    test_file = sys.argv[3]

    # And the location to store the trained model
    model_file = sys.argv[4]

    print('Training Recommender System')

    # Call our main routine
    main(spark, data_file, validation_file, test_file, model_file, tuning=False)
