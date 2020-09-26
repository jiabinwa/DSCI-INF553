"""
    USC Spring 2020
    INF 553 Foundations of Data Mining
    Assignment 4
    
    Student Name: Jiabin Wang
    Student ID: 4778-4151-95
"""
from pyspark import SparkConf, SparkContext, StorageLevel
from pyspark.sql import SparkSession, SQLContext
from collections import deque
from auxiliary import makeCombinations, makeListSorted, buildGraph, Graph
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

    THRESHOLD = int(sys.argv[1])
    input_file_path = sys.argv[2]
    betweenness_output_file_path = sys.argv[3]
    community_output_file_path = sys.argv[4]

    spark = (
        SparkSession.builder.master("local[3]")
        .appName("inf553")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )

    user_vertices, user_edges_si, user_edges_bi = buildGraph(
        spark, THRESHOLD, input_file_path
    )
    vertices = set(user_vertices.map(lambda x: x[0]).collect())
    edges = user_edges_si.map(lambda x: (x, 0)).collectAsMap()
    neighbour = (
        user_edges_bi.map(lambda x: (x[0], [x[1]]))
        .reduceByKey(lambda a, b: a + b)
        .collectAsMap()
    )

    graph = Graph(neighbour, vertices, edges)
    
    betweeness_list = graph.calculate_current_betweeness()
    output = open(betweenness_output_file_path, "w")
    for b_triple in betweeness_list:
        output.write(
            "('" + b_triple[0] + "', '" + b_triple[1] + "'), " + str(b_triple[2])
        )
        output.write("\n")
    output.close()

    best_partition = list(graph.detect_community())
    # best_partition = sorted(best_partition, key=lambda x: (len(x),list(x)[0]))

    result = []
    for item in best_partition:
        x = list(item)
        result.append(x)
    best_partition = result
    best_partition = sorted(best_partition, key=lambda x: (len(x), x[0]))

    output = open(community_output_file_path, "w")
    for item in best_partition:
        item = list(item)
        item.sort()
        o = ", ".join(["'{}'".format(i) for i in item])
        output.write(o)
        output.write("\n")
    output.close()

    time_end = time.time()
    print("Duration: ", time_end - time_start, "s")
