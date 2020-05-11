# Recommender-System-Project

We built a recommender system in this project, we will be using the Goodreads dataset collect in late 2017 from goodreads.com by Mengting Wan and Julian McAuley. This metadata contains information about ‘876,145 users; 228,648,342 user-book interactions in users' shelves (include 112,131,203 reads and 104,551,549 ratings)’. We use the Alternative Least Square method in Spark to use collaborative filtering to learn latent factor representations for users and items. We also incorporate hyperparameter tuning in model training as well. For the fast search extension feature, we apply the annoy library (which are currently used for spotify recommendation system). The recommender system is built on the Dumbo cluster on NYU HPC. 

Further details and evaluation are stored in the pdf file. 

# Data Storage

All data has been preprocessed and saved in the following repository on the Dumbo cluster as well as the Model indexer and ALS model which might be used during installation. 

Training data:   	'hdfs:/user/hj1325/final-project-final-project/train_df.parquet'
Validation data:   	'hdfs:/user/hj1325/final-project-final-project/validation_df.parquet'
Testing data 	'hdfs:/user/hj1325/final-project-final-project/test_df.parquet'
Model indexer	'./home/hj1325/final-project-final-project/model_indexer'
ALS model		'hdfs:/user/hj1325/final-project-final-project/final_model

# Installation instructions

Basic Recommender System: The basic recommender system includes three components, Final_Project_Train.py, Final_Project_Test.py and Data_Splitting.py. 

      --Data Splitting:   - spark-submit    Data_Splitting.py     path_to_data_file

This program clean, preprocess and split the original data into training, validation, and testing data with the percentage of 60%,20%,20% accordingly. 

      --Training and Tuning Model:   - spark-submit    Final_Project_Train.py	path_to_train_data     
         path_to_validation_data	     path_to_test_data      path_to_save_model      tuning option
     
This program takes the input of the training data, validation data, and testing data, preprocess the data to determine the necessary user row for creating the model. The default parameter ‘tuning’ is set to off to test through the entire pipeline that creates an index, transform, then fit the ALS model with training data. Set tuning=true to activate hyper-parameter tuning.     
     
     --Testing Model:   - spark-submit    Final_Project_Test.py     path_to_test_data      
       path_to_model_indexer      path_to_als_model
     
This program takes the input of the test data, loads a model indexer(string), a fitted ALS model, and evaluate the performance of the model against test data. 

Extension: Fast Search: This extension requires the use of packages annoy, which is available through pip install annoy 

    --Evaluate Elapsed Time and Precision:   - spark-submit    Extension_fast_search.py      
     Path_to_test_data     path_to_model_indexer      path_to_als_model     limit option

This program takes the input of the testing data, a model indexer(string), a fitted ALS model to 
generate trained brute-force search model. The limit option determines the proportion of testing data we are using to train our fast search extension. (limit defaults to 0.01)
