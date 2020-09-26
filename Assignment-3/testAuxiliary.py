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


def sortBid(bid1, bid2):
    if bid1 > bid2:
        bid1, bid2 = bid2, bid1
    return bid1, bid2


def removeDuplicatesRate(pastRate):
    after = {}
    for pair in pastRate:
        if pair[0] not in after:
            after[pair[0]] = (0, 0)
        after[pair[0]] = (after[pair[0]][0] + pair[1], after[pair[0]][1] + 1)
    for bid in after:
        after[bid] = after[bid][0] / after[bid][1]
    return after


def getSortedCombinations(bidCandidates, bid, bidSimilarity, TopN):
    ans = []
    for candidate in bidCandidates:
        _bid, _candidate = sortBid(bid, candidate)
        if (_bid, _candidate) in bidSimilarity:
            ans.append((bid, candidate, bidSimilarity[(_bid, _candidate)]))
    ans = sorted(ans, key=lambda triple: -triple[2])[:TopN]
    return ans


def itemBasedPredict(uid, bid, pastRate, bidSimilarity, TopN):
    bidCandidates = set(map(lambda x: x[0], pastRate))
    pastRate = removeDuplicatesRate(pastRate)
    sortedCombinations = getSortedCombinations(bidCandidates, bid, bidSimilarity, TopN)
    A = 0
    W = 0
    for triple in sortedCombinations:
        A = A + triple[2] * pastRate[triple[1]]
        W = W + triple[2]
    if W == 0:
        return (1, 1)  # Must be changed
    predict_stars = A / W
    return (uid, bid, predict_stars)


def itemBased(sc, input_file_path, test_file_path, model_file_path, output_file_path):
    validBidCollection = (
        sc.textFile(model_file_path)
        .map(lambda line: [json.loads(line)["b1"], json.loads(line)["b2"]])
        .flatMap(lambda pair: pair)
        .zipWithIndex()
        .collectAsMap()
    )

    bidSimilarity = (
        sc.textFile(model_file_path)
        .map(
            lambda line: (
                (json.loads(line)["b1"], json.loads(line)["b2"]),
                json.loads(line)["sim"],
            )
        )
        .collectAsMap()
    )

    userRate = (
        sc.textFile(input_file_path)
        .map(
            lambda line: (
                json.loads(line)["user_id"],
                [(json.loads(line)["business_id"], json.loads(line)["stars"])],
            )
        )
        .reduceByKey(lambda a, b: a + b)
        .collectAsMap()
    )

    test = (
        sc.textFile(test_file_path)
        .map(
            lambda line: (
                json.loads(line)["user_id"],
                json.loads(line)["business_id"],
            )
        )
        .filter(
            lambda triple: triple[1] in validBidCollection and triple[0] in userRate
        )
        .map(
            lambda triple: itemBasedPredict(
                triple[0], triple[1], userRate[triple[0]], bidSimilarity, 7
            )
        )
        .filter(lambda quintuple: len(quintuple) >= 3)
        .collect()
    )
    return test


def itemBasedPredictionOutPut(output_file_path, test):
    output = open(output_file_path, "a")
    for quintuple in test:
        uid = quintuple[0]
        bid = quintuple[1]
        predict_stars = quintuple[2]
        content = json.dumps(
            {"user_id": uid, "business_id": bid, "stars": predict_stars}
        )
        output.write(content)
        output.write("\n")
    output.close()

def userBasedPredict(uid, bid, pastRate, bidSimilarity, TopN):
    average = sum(list(map(lambda x : x[1], pastRate))) / len(list(map(lambda x : x[1], pastRate)))
    uidCandidates = set(map(lambda x : x[0], pastRate))
    pastRate = removeDuplicatesRate(pastRate)
    sortedCombinations = getSortedCombinations(uidCandidates, bid, bidSimilarity, TopN)
    A = 0
    W = 0
    for triple in sortedCombinations:
        A = A + (triple[2] - average) * pastRate[triple[1]]
        W = W + triple[2]
    if W == 0:
        predict_stars = average # Must be changed
    else:
        predict_stars = A / W
    return (uid, bid, predict_stars)


def userBased(sc, input_file_path, test_file_path, model_file_path, output_file_path):
    validUidCollection = (
        sc.textFile(model_file_path)
        .map(lambda line: [json.loads(line)["u1"], json.loads(line)["u2"]])
        .flatMap(lambda pair: pair)
        .zipWithIndex()
        .collectAsMap()
    )

    uidSimilarity = (
        sc.textFile(model_file_path)
        .map(
            lambda line: (
                (json.loads(line)["u1"], json.loads(line)["u2"]),
                json.loads(line)["sim"],
            )
        )
        .collectAsMap()
    )

    businessRate = (
        sc.textFile(input_file_path)
        .map(
            lambda line: (
                json.loads(line)["business_id"],
                [(json.loads(line)["user_id"], json.loads(line)["stars"])],
            )
        )
        .reduceByKey(lambda a, b: a + b)
        .collectAsMap()
    )

    test = (
        sc.textFile(test_file_path)
        .map(
            lambda line: (
                json.loads(line)["user_id"],
                json.loads(line)["business_id"],
            )
        )
        .filter(
            lambda triple: triple[0] in validUidCollection and triple[1] in businessRate
        )
        .map(
            lambda triple: userBasedPredict(
                triple[0], triple[1], businessRate[triple[1]], uidSimilarity, 5
            )
        )
        .collect()
    )

    output = open(output_file_path, "a")
    for quintuple in test:
        uid = quintuple[0]
        bid = quintuple[1]
        predict_stars = quintuple[2]
        content = json.dumps({"user_id": uid, "business_id": bid, "stars": predict_stars})
        output.write(content)
        output.write("\n")
    output.close()
