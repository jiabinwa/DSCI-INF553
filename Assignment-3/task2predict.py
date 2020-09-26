"""
    USC Spring 2020
    INF 553 Foundations of Data Mining
    Assignment 3
    
    Student Name: Jiabin Wang
    Student ID: 4778-4151-95
"""
from pyspark import SparkConf, SparkContext, StorageLevel
import os
import re
import json
import time
import sys
import math
import random
import itertools

def makePair(triple):
    return (triple[1],triple[2])

def cosine_similarity(pair, businessProfile, userProfile):
    if pair[0] not in businessProfile or pair[1] not in userProfile:
        return 0
    a = set(businessProfile[pair[0]])
    v = set(userProfile[pair[1]])
    return len(a.intersection(v))/(math.sqrt(len(a)) * math.sqrt(len(v)))

time_start = time.time()

conf = (
    SparkConf()
    .setAppName("task2")
    .set("spark.driver.memory", "4g")
    .set("spark.executor.memory", "4g")
)
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")


test_file_path = sys.argv[1]
model_file_path = sys.argv[2]
output_file_path = sys.argv[3]


model = sc.textFile(model_file_path).map(
    lambda line: (
        json.loads(line)["profile_type"],
        json.loads(line)["profile_id"],
        json.loads(line)["profile_content"],
    )
).persist(storageLevel=StorageLevel.MEMORY_AND_DISK)


userProfile = model.filter(lambda triple: triple[0] == 'user').map(lambda triple : makePair(triple)).collectAsMap()
businessProfile = model.filter(lambda triple: triple[0] == 'business').map(lambda triple : makePair(triple)).collectAsMap()


test = sc.textFile(test_file_path)\
         .map(lambda line : (json.loads(line)["business_id"],json.loads(line)["user_id"], cosine_similarity((json.loads(line)["business_id"],json.loads(line)["user_id"]), businessProfile, userProfile)))\
         .filter(lambda triple : triple[2] >= 0.01 ).collect()

output = open(output_file_path, "a")
for triple in test:
    user_id = triple[1]
    business_id = triple[0]
    sim = triple[2]
    content = json.dumps(
        {"user_id": user_id, "business_id": business_id, "sim": sim}
    )
    output.write(content)
    output.write("\n")
output.close()

print("Accuracy: " + str(len(test)/58480))
time_end = time.time()
print("Duration: ", time_end - time_start, "s")
