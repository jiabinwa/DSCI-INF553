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
    model = open(model_file_path, "w")
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

# User Based Adds:

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

def makeUserBasedPearson(uid1, uid2, comment):
    coratedItem = getCoRatedUser(comment[uid1], comment[uid2])
    return (uid1, uid2, calculateW(uid1, uid2, comment, coratedItem))


def userBasedTrain(sc, input_file_path, model_file_path):
    train_review = sc.textFile(input_file_path)
    # 1. Numbering
    uid_numbering = (
        train_review.map(lambda line: json.loads(line)["business_id"])
        .distinct()
        .zipWithIndex()
        .collectAsMap()
    )

    uid_max = len(uid_numbering) - 1

    # 2. Create Utility Matrix
    utility_matrix = (
        train_review.map(
            lambda line: (
                json.loads(line)["user_id"],
                uid_numbering[json.loads(line)["business_id"]],
            )
        )
        .map(lambda pair: (pair[0], list([pair[1]])))
        .reduceByKey(lambda a, b: a + b)
        .map(lambda pair: sortValue(pair))
    )

    # 3. Create Signatures
    signatureLength = 50
    parameters = generateParameters(signatureLength)
    signatures = utility_matrix.mapValues(
        lambda values: generateSignatures(values, parameters, uid_max)
    )

    band = 50
    row = (int)(signatureLength / band)


    candidates = (
        signatures.flatMap(lambda pair: generateHash(pair[1], row, pair[0]))
        .reduceByKey(lambda a, b: a + b)
        .filter(lambda pair: len(pair[1]) > 1)
        .flatMap(lambda pair: list(itertools.combinations(pair[1], 2)))
        .map(lambda pair: sortTuple(pair))
        .groupByKey()
        .flatMap(lambda x: [(x[0], i) for i in set(x[1])])
    )


    u_matrix = utility_matrix.collectAsMap()


    comment = (
        train_review.map(
            lambda line: (
                json.loads(line)["user_id"],
                [(json.loads(line)["business_id"], json.loads(line)["stars"])],
            )
        )
        .reduceByKey(lambda a, b: a + b)
        .collectAsMap()
    )


    result = (
        candidates.map(lambda pair: buildDetails(pair, u_matrix))
        .filter(lambda pair: pair[1] >= 0.01 and isAtleast3CoItems(pair[0][0], pair[0][1], u_matrix))
        .map(lambda pair : makeUserBasedPearson(pair[0][0], pair[0][1], comment))
        .persist(storageLevel = StorageLevel.MEMORY_AND_DISK)
        .filter(lambda triple : triple[2] >= 0.000001)
        .collect()
    )   

    return result

def userBasedOutputModel(model_file_path, result):
    with open(model_file_path, "w") as output:
        for triple in result:
            u1 = triple[0]
            u2 = triple[1]
            sim = triple[2]
            if u1 > u2:
                u1, u2 = u2, u1
            content = json.dumps({"u1": u1, "u2": u2, "sim": sim})
            output.write(content)
            output.write("\n")






