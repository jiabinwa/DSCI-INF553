'''
    USC Spring 2020
    INF 553 Foundations of Data Mining
    Assignment 3
    
    Student Name: Jiabin Wang
    Student ID: 4778-4151-95
'''
from pyspark import SparkConf, SparkContext, StorageLevel
import os
import re
import json
import time
import sys
import math
import random
import itertools


def generatePair(index):
    ans = []
    for i in range(0, index):
        ans.append((i, index))
    return ans

def getCoRatedUser(comment1, comment2):
    author1 = set(map(lambda x : x[0],comment1))
    author2 = set(map(lambda x : x[0],comment2))
    coratedUser = author1.intersection(author2)
    return coratedUser

def filtCoRatedAtLeast3(pair, comment, indexToBid):
    bid1 = indexToBid[pair[0]]
    bid2 = indexToBid[pair[1]]
    if len(getCoRatedUser(comment[bid1], comment[bid2])) >= 3:
        return True
    return False

def filterNoCoratedComment(before, coratedUser):
    after = {}
    for pair in before:
        if pair[0] in coratedUser:
            if pair[0] not in after:
                after[pair[0]] = (0,0)
            after[pair[0]] = (after[pair[0]][0] + pair[1],after[pair[0]][1] + 1)
    for bid in after:
        after[bid] = after[bid][0] / after[bid][1]
    return after


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

def MakePearson(pair, comment, indexToBid):
    bid1 = indexToBid[pair[0]]
    bid2 = indexToBid[pair[1]]
    coratedUser = getCoRatedUser(comment[bid1], comment[bid2])
    return (bid1, bid2, calculateW(bid1, bid2, comment, coratedUser))

def itemBasedOutputModel(model_file_path, pairCollection):
    model = open(model_file_path, "a")
    for triple in pairCollection:
        if triple[0] < triple[1]:
            b1 = triple[0]
            b2 = triple[1]
        else:
            b1 = triple[1]
            b2 = triple[0]
        content = json.dumps(
            {"b1": b1, "b2": b2, "sim": triple[2]}
        )
        model.write(content)
        model.write("\n")
    model.close()


def itemBasedTrain(sc, input_file_path, model_file_path):
    train_review = sc.textFile(input_file_path)

    bidCollection = train_review.map(
        lambda line: json.loads(line)["business_id"]
    ).distinct()

    bidCollection.persist(storageLevel=StorageLevel.MEMORY_AND_DISK)
    
    indexToBid = (
        bidCollection.zipWithIndex().map(lambda pair: (pair[1], pair[0])).collectAsMap()
    )
    bidAmount = len(indexToBid)

    comment = (
        train_review.map(
            lambda line: (
                json.loads(line)["business_id"],
                [(json.loads(line)["user_id"], json.loads(line)["stars"])],
            )
        )
        .reduceByKey(lambda a, b: a + b)
        .collectAsMap()
    )

    pairCollection = (
        sc.parallelize(list(range(0, bidAmount)))
        .flatMap(lambda index: generatePair(index))
        .filter(lambda pair: filtCoRatedAtLeast3(pair, comment, indexToBid)) #Huge
        .map(lambda pair : MakePearson(pair, comment, indexToBid))
        .filter(lambda triple: triple[2] > 0.0001)
        .collect()
    )

    return pairCollection






