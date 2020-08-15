'''
    USC Spring 2020
    INF 553 Foundations of Data Mining
    Assignment 1
    
    Student Name: Jiabin Wang
'''
from pyspark import SparkContext
import sys
import os
import re
import json
import time
import heapq


def parseReviewInfo(line):
    '''
    Function: 
        Extract the variable business_id from JSON file
        Form the tuple (business_id, 1)

    Parameter:
        One JSON object: one line from the JSON file
    
    Return:
        A tuple: (business_id, 1)
    '''
    return (json.loads(line)['business_id'],1)


def saveToFile(result,result_file_path):
    with open(result_file_path,'w') as File:
        json.dump(result,File)

def countPartition(iterator):
    yield len(list(iterator))


def default(sc, review_file_path, n_partitions, N):
    '''
    Function: 
        1. Read the review text from JSON file and transform it
            into RDD format ==> (business_id, 1)
        2. Log the number of partitions
        3. Log the number of items each partition contains
        4. Log the final result: those business_ids who have 
            at least n reviews

    Parameter:
        sc: SparkContext
        review_file_path: the path of review JSON file
        n_partitions: No use at here, can be removed
        N: the threshold of n reviews
    
    Return:
        A List of tuples: [(business_id, count)]
    '''
    raw_reviewRDD = sc.textFile(review_file_path).map(lambda line: parseReviewInfo(line))
    result = {}
    result['n_partitions'] = raw_reviewRDD.getNumPartitions()
    result['n_items'] = raw_reviewRDD.mapPartitions(countPartition).collect()
    result['result']  = (raw_reviewRDD\
                            .reduceByKey(lambda a,b: a + b)\
                            .filter(lambda pairwise: pairwise[1] > N)\
                            .map(lambda x: list(x)).collect())
    return result

def customized(sc, review_file_path, n_partitions, N):
    '''
        Same logic with the default() function,
        except using the python in-built hash() function as
        the partition function.

        hash() function can be replaced as other hash functions,
        such as md5, but will cost more time. 
    '''
    partitionRDD = sc.textFile(review_file_path).map(lambda line: parseReviewInfo(line))
    result = {}
    partitionRDD = partitionRDD.partitionBy(n_partitions,lambda key : hash(key))
    result['n_items'] = partitionRDD.mapPartitions(countPartition).collect()
    result['n_partitions'] = n_partitions
    result['result'] = (partitionRDD.reduceByKey(lambda a,b: a + b)\
                                   .filter(lambda pairwise: pairwise[1] > N)\
                                   .map(lambda x: list(x)).collect())
    return result

def main():
    time_start=time.time()
    review_file_path = sys.argv[1]
    result_file_path = sys.argv[2]
    mode = sys.argv[3]
    n_partitions = int(sys.argv[4])
    N = int(sys.argv[5])
    sc = SparkContext('local[*]', 'task3')
    sc.setLogLevel("ERROR")
    
    if mode == "default":
        print("Default")
        result = default(sc, review_file_path, n_partitions, N)
    else:
        print("Customized")
        result = customized(sc, review_file_path, n_partitions, N)
    
    saveToFile(result,result_file_path)
    time_end=time.time()
    print('time cost',time_end-time_start,'s')

if __name__ == "__main__":
    main()