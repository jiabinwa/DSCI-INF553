"""
    USC Spring 2020
    INF 553 Foundations of Data Mining
    Assignment 3
    
    Student Name: Jiabin Wang
    Student ID: 4778-4151-95
"""
from pyspark import SparkConf, SparkContext, StorageLevel
from trainAuxiliary import *
'''
import os
import re
import json
import time
import sys
import math
import random
import itertools
'''

if __name__ == "__main__":
    time_start = time.time()

    # Get the input parameters
    input_file_path = sys.argv[1] #"./Dataset/train_review.json"
    model_file_path = sys.argv[2]
    cf_type = sys.argv[3]

    # Configure the Spark
    conf = (
        SparkConf()
        .setAppName("task3")
        .set("spark.driver.memory", "4g")
        .set("spark.executor.memory", "4g")
    )
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    if cf_type == "item_based":
        pairCollection = itemBasedTrain(sc, input_file_path, model_file_path)
        itemBasedOutputModel(model_file_path, pairCollection)
    
    if cf_type == "user_based":
        result = userBasedTrain(sc, input_file_path, model_file_path)
        userBasedOutputModel(model_file_path, result)
    
    time_end = time.time()
    print("Duration: ", time_end - time_start, "s")
