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
from trainAuxiliary import *


'''
def sortValue(pair):
    k = pair[0]
    pair[1].sort()
    v = pair[1]
    return (k, v)


def generateParameters(size):
    parameters = []
    i = 0
    random.seed(0)
    while i < size:
        i = i + 1
        a = random.randint(1, pow(2, 20))
        b = random.randint(1, pow(2, 20))
        parameters.append((a, b))
    return parameters


def permutation_function(a, b, x, m):
    return (a * x + b) % m


def generateSignatures(values, parameters, uid_max):
    ans = []
    for (a, b) in parameters:
        l = [permutation_function(a, b, one, uid_max) for one in values]
        ans.append(min(l))
    return ans


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def generateHash(values, row, bid):
    # values: signature vector
    # row: r
    # bid: key
    ans = []
    l = []
    bandid = 0
    i = 0
    while i < len(values):
        l.append(values[i])
        i = i + 1
        if i % row == 0:
            ans.append(((hash("".join([str(m) for m in l])), bandid), [bid]))
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
    sim = jaccard_similarity(v1, v2)
    return ((bid1, bid2), sim)


def isAtleast3CoItems(uid1, uid2, u_matrix):
    if uid1 not in u_matrix or uid2 not in u_matrix:
        return False
    uid1Rates = set(u_matrix[uid1])
    uid2Rates = set(u_matrix[uid2])
    if len(uid1Rates.intersection(uid2Rates)) >= 3:
        return True
    return False


def getCoRatedUser(comment1, comment2):
    author1 = set(map(lambda x : x[0],comment1))
    author2 = set(map(lambda x : x[0],comment2))
    coratedUser = author1.intersection(author2)
    return coratedUser

# def filterNoCoratedComment(before, coratedUser):
#     after = {}
#     for pair in before:
#         if pair[0] in coratedUser:
#             if pair[0] not in after:
#                 after[pair[0]] = pair[1]
#             else:
#                 after[pair[0]] = after[pair[0]] * 0.5 + pair[1] * 0.5
#     return after

def getAvg(commentMap):
    tmp = []
    for uid in commentMap:
        tmp.append(commentMap[uid])
    return sum(tmp) / len(tmp)

def calculateW(bid1, bid2, comment, coratedUser):
    rate_i = filterNoCoratedComment(comment[bid1], coratedUser)
    rate_j = filterNoCoratedComment(comment[bid2], coratedUser)
    avg_i = getAvg(rate_i)
    avg_j = getAvg(rate_j)
    A = 0
    B = 0
    C = 0
    for uid in coratedUser:
        A = A + ((rate_i[uid] - avg_i) * (rate_j[uid] - avg_j))
        B = B + pow((rate_i[uid] - avg_i), 2)
        C = C + pow((rate_j[uid] - avg_j), 2)

    D = math.sqrt(B) * math.sqrt(C)
    if D == 0:
        return 0
    else:
        return A / D


def makeUserBasedPearson(uid1, uid2, comment):
    coratedItem = getCoRatedUser(comment[uid1], comment[uid2])
    return (uid1, uid2, calculateW(uid1, uid2, comment, coratedItem))
'''



conf = (
    SparkConf()
    .setAppName("task")
    .set("spark.driver.memory", "4g")
    .set("spark.executor.memory", "4g")
)
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

# input_file_path = './Dataset/train_review.json'
input_file_path = sys.argv[1] #'./Dataset/train_review.json'
model_file_path = sys.argv[2]


time_start = time.time()


result = userBasedTrain(sc, input_file_path, model_file_path)
userBasedOutputModel(model_file_path, result)

time_end = time.time()
print("Duration: ", time_end - time_start, "s")
