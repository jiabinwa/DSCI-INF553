"""
    USC Spring 2020
    INF 553 Foundations of Data Mining
    Assignment 3
    
    Student Name: Jiabin Wang
    Student ID: 4778-4151-95
"""
from pyspark import SparkConf, SparkContext, StorageLevel
from testAuxiliary import *
import os
import re
import json
import time
import sys
import math
import random
import itertools

if __name__ == "__main__":
    time_start = time.time()

    conf = (
        SparkConf()
        .setAppName("task3")
        .set("spark.driver.memory", "4g")
        .set("spark.executor.memory", "4g")
    )
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    input_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    model_file_path = sys.argv[3]
    output_file_path = sys.argv[4]
    cf_type = sys.argv[5]

    if cf_type == "item_based":
        test = itemBased(sc, input_file_path, test_file_path, model_file_path, output_file_path)
        itemBasedPredictionOutPut(output_file_path, test)
    if cf_type == "user_based":
        userBased(sc, input_file_path, test_file_path, model_file_path, output_file_path)

    time_end = time.time()
    print("Duration: ", time_end - time_start, "s")
