"""
    USC Spring 2020
    INF 553 Foundations of Data Mining
    Assignment 4
    
    Student Name: Jiabin Wang
    Student ID: 4778-4151-95
"""
from pyspark import SparkConf, SparkContext, StorageLevel
from pyspark.sql import SparkSession, SQLContext
from graphframes import *
import os
import re
import json
import time
import sys
import math
import random
import itertools


def makeCombinations(bid, uidList):
    answerSet = set()
    candidates = list(itertools.combinations(uidList, 2))
    for candidate in candidates:
        uid1 = candidate[0]
        uid2 = candidate[1]
        if uid1 > uid2:
            uid1, uid2 = uid2, uid1
        if (uid1, uid2) not in answerSet:
            answerSet.add((uid1, uid2))
    answer = []
    for uidPair in answerSet:
        answer.append((uidPair, [bid]))
    return answer


def makeListSorted(l):
    l.sort()
    return l

time_start = time.time()

spark = (
    SparkSession.builder.master("local[3]")
    .appName("inf553")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .getOrCreate()
)

os.environ[
    "PYSPARK_SUBMIT_ARGS"
] = "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11"

spark.sparkContext.setLogLevel("ERROR")


THRESHOLD = int(sys.argv[1])
input_file_path = sys.argv[2]
output_file_path = sys.argv[3]


csv_file = spark.sparkContext.textFile(input_file_path)
header = csv_file.first()
csv_file = csv_file.filter(lambda _: _ != header)

rdd_file = (
    csv_file.map(lambda line: (line.split(",")[1], [line.split(",")[0]]))
    .reduceByKey(lambda a, b: a + b)
    .mapValues(lambda values: list(set(values)))
    .flatMap(lambda x: makeCombinations(x[0], x[1]))
    .reduceByKey(lambda a, b: a + b)
    .filter(lambda x: len(x[1]) >= THRESHOLD)
)

rdd_file.persist(storageLevel=StorageLevel.MEMORY_AND_DISK)

user_vertices = rdd_file.flatMap(lambda x: [(x[0][0],), (x[0][1],)]).distinct()
user_vertices.persist(storageLevel=StorageLevel.MEMORY_AND_DISK)

user_edges = rdd_file.flatMap(lambda x: [(x[0][0], x[0][1]), (x[0][1], x[0][0])])
user_edges.persist(storageLevel=StorageLevel.MEMORY_AND_DISK)

user_vertices_df = user_vertices.toDF(["id"])
user_edges_df = user_edges.toDF(["src", "dst"])

graph = GraphFrame(user_vertices_df, user_edges_df)
result = (
    graph.labelPropagation(maxIter=5)
    .rdd.map(lambda x: (x["label"], [x["id"]]))
    .reduceByKey(lambda a, b: a + b)
    .map(lambda x : makeListSorted(x[1]))
    .sortBy(lambda x : (len(x), x[0]))
    .collect()
)



output = open(output_file_path, "w")
for community in result:
    community.sort()
    community = ', '.join(["'{}'".format(i) for i in community])
    output.write(community)
    output.write("\n")
output.close()

time_end = time.time()
print("Duration: ", time_end - time_start, "s")
