'''
    USC Spring 2020
    INF 553 Foundations of Data Mining
    Assignment 3
    
    Student Name: Jiabin Wang
    Student ID: 4778-4151-95
'''
from pyspark import SparkContext
import os
import re
import json
import time
import sys
import math
import random
import itertools

'''
def fullVector(old, uid_max):
    result = [0] * uid_max
    for i in old:
        result[0] = 1
    return result
utility_matrix = utility_matrix_nopadding.mapValues(lambda val : fullVector(val, uid_max))
'''


# input_file_path = './Dataset/train_review.json'
input_file_path = sys.argv[1]
output_file_path = sys.argv[2]
train_review = sc.textFile(input_file_path)

def sortValue(pair):
    k = pair[0]
    pair[1].sort()
    v = pair[1]
    return (k,v)

def generateParameters(size):
    parameters = []
    i = 0
    while(i < size):
        i = i + 1
        a = random.randint(1,pow(2,20))
        b = random.randint(1,pow(2,20))
        parameters.append((a,b))
    return parameters

def permutation_function(a,b,x,m):
    return (a * x + b)%m

def generateSignatures(values, parameters, uid_max):
    ans = []
    for (a,b) in parameters:
        l = []
        for one in values:
            l.append(permutation_function(a,b,one,uid_max))
        ans.append(min(l))
    return ans

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def generateHash(values,row,bid):
    ans = []
    l = []
    bandid = 0
    i = 0
    while(i < len(values)):
        l.append(values[i])
        i = i + 1
        if i % row == 0:
            ans.append(((hash(tuple(l)),bandid),[bid]))
            l = []
            bandid = bandid + 1
    return ans

def sortTuple(t):
    l = list(t)
    l.sort()
    return tuple(l)

def buildDetails(pair, u_matrix):
    bid1 = pair[0]
    bid2 = pair[1]
    v1 = u_matrix[bid1]
    v2 = u_matrix[bid2]
    sim = jaccard_similarity(v1,v2)
    return ((bid1,bid2),sim)

time_start=time.time()

# 1. Numbering
uid_numbering  = train_review.map(lambda line : json.loads(line)['user_id'])\
                             .distinct().zipWithIndex().collectAsMap()

uid_max = len(uid_numbering) - 1


# 2. Create Utility Matrix
utility_matrix = train_review.map(lambda line: (json.loads(line)['business_id'] , uid_numbering[json.loads(line)['user_id']]))\
                                       .distinct().map(lambda pair : (pair[0], list([pair[1]]))).reduceByKey(lambda a,b : a + b)\
                                       .map(lambda pair: sortValue(pair))


# 3. Create Signatures
# generate function clusters:
signatureLength = 50
parameters = generateParameters(signatureLength)

signatures = utility_matrix.mapValues(lambda values: generateSignatures(values, parameters, uid_max))

band = 50
row = (int)(signatureLength / band)

candidates = signatures.flatMap(lambda pair : generateHash(pair[1],row,pair[0]))\
                        .reduceByKey(lambda a,b: a + b).filter(lambda pair : len(pair[1]) > 1)\
                        .flatMap(lambda pair : list(itertools.combinations(pair[1],2)))\
                        .map(lambda pair: sortTuple(pair)).distinct()
u_matrix = utility_matrix.collectAsMap()
result = candidates.map(lambda pair : buildDetails(pair, u_matrix)).filter(lambda pair : pair[1] >= 0.05).collect()

output = open(output_file_path, "a")
for pair in result:
    b1 = pair[0][0]
    b2 = pair[0][1]
    sim = pair[1]
    content = json.dumps(
        {"b1": b1, "b2": b2, "sim": sim}
    )
    output.write(content)
    output.write("\n")
output.close()

time_end=time.time()
print('Duration: ',time_end-time_start,'s')





