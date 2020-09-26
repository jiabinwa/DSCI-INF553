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


review_file_path = sys.argv[1]
business_file_path = sys.argv[2]
result_file_path = sys.argv[3]

conf = (
    SparkConf()
    .setAppName("task2")
    .set("spark.driver.memory", "4g")
    .set("spark.executor.memory", "4g")
)
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

# input_file_path = "./Dataset/train_review.json"
# stopwords_file_path = "./Dataset/stopwords"

input_file_path = sys.argv[1]
model_file_path = sys.argv[2]
stopwords_file_path = sys.argv[3]


train_review = sc.textFile(input_file_path)
stopwords = sc.textFile(stopwords_file_path).collect()


def stopwordFilter(value, stopwords):
    ans = []
    for word in value:
        if (
            word not in stopwords
            and word != ""
            and re.match(r"^(\d+|[a-z])$", word) is None
        ):
            ans.append(word)
    return ans


def rarewordFilter(value, rarewords):
    ans = []
    for word in value:
        if word not in rarewords:
            ans.append(word)
    return ans


def getWordsCount(x):
    words = x[1]
    hashmap = {}
    ans = []
    for word in words:
        if word not in hashmap:
            hashmap[word] = 0
        hashmap[word] = hashmap[word] + 1
    for word in hashmap:
        ans.append((word, [hashmap[word]]))  # NAN
    return ans


def makeTf(x, max_frequency):
    k = x[0]
    v = {}
    words = x[1]
    hashmap = {}
    for word in words:
        if word not in hashmap:
            hashmap[word] = 0
        hashmap[word] = hashmap[word] + 1
    for word in hashmap:
        v[word] = (hashmap[word]) / max_frequency[word]
    return (k, v)


def makeIdf(x):
    ans = []
    words = x[1]
    s = {}
    for word in words:
        if word not in s:
            s[word] = 0
    for word in s:
        ans.append((word, 1))
    return ans


def makeTfIdf(value, idf):
    for word in value:
        value[word] = value[word] * idf[word]
    value = dict(sorted(value.items(), key=lambda item: -item[1])[:200])
    return value


def wordsToIndex(value, vector):
    ans = []
    for word in value:
        ans.append(vector[word])
    return set(ans)


def unionIndex(value, documentProfile):
    ans = set()
    for bid in value:
        ans = ans.union(documentProfile[bid])
    return ans


time_start = time.time()

# 1. Concatenating all the review texts
comment = (
    train_review.map(
        lambda line: (
            json.loads(line)["business_id"],
            json.loads(line)["text"].lower(),
        )
    )
    .reduceByKey(lambda a, b: a + " " + b)
    .map(
        lambda pair: (
            pair[0],
            re.split(r"\\[a-z]|\s|[!'\"#$%&()*+,\-./:;<=>?@\[\]^_`{|}~\\]", pair[1]),
        )
    )
    .mapValues(lambda value: stopwordFilter(value, stopwords))
)

comment.persist(storageLevel=StorageLevel.MEMORY_AND_DISK)

word_count = comment.flatMap(
    lambda x: list(map(lambda word: (word, 1), x[1]))
).reduceByKey(lambda a, b: a + b)

word_count.persist(storageLevel=StorageLevel.MEMORY_AND_DISK)

total_words_number = int(
    word_count.map(lambda x: ("count", x[1]))
    .reduceByKey(lambda a, b: a + b)
    .collect()[0][1]
)

rareword_threshold = int(total_words_number * 1e-6)
rarewords = word_count.filter(lambda x: x[1] <= rareword_threshold).collectAsMap()

comment = comment.mapValues(lambda value: rarewordFilter(value, rarewords))
comment.persist(storageLevel=StorageLevel.MEMORY_AND_DISK)

# 2. TF-IDF

max_frequency = (
    comment.flatMap(lambda x: getWordsCount(x))
    .reduceByKey(lambda a, b: a + b)
    .mapValues(lambda value: max(value))
    .collectAsMap()
)

tf = comment.map(lambda x: makeTf(x, max_frequency))

# IDF

N = comment.count()
idf = (
    comment.flatMap(lambda x: makeIdf(x))
    .reduceByKey(lambda a, b: a + b)
    .map(lambda pair: (pair[0], N / pair[1]))
    .collectAsMap()
)

tfIdf = tf.mapValues(lambda value: makeTfIdf(value, idf))

vector = (
    tfIdf.flatMap(lambda pair: pair[1].keys()).distinct().zipWithIndex().collectAsMap()
)

documentProfile = tfIdf.mapValues(
    lambda value: wordsToIndex(value, vector)
).collectAsMap()

userProfile = (
    train_review.map(
        lambda line: (json.loads(line)["user_id"], [json.loads(line)["business_id"]])
    )
    .reduceByKey(lambda a, b: list(set(a + b)))
    .mapValues(lambda value: unionIndex(value, documentProfile))
    .collectAsMap()
)

model = open(model_file_path, "a")
for bid in documentProfile:
    v = documentProfile[bid]
    v = list(v)
    v.sort()
    content = json.dumps(
        {"profile_type": "business", "profile_id": bid, "profile_content": tuple(v)}
    )
    model.write(content)
    model.write("\n")

for uid in userProfile:
    v = userProfile[uid]
    v = list(v)
    v.sort()
    content = json.dumps(
        {"profile_type": "user", "profile_id": uid, "profile_content": tuple(v)}
    )
    model.write(content)
    model.write("\n")
model.close()


time_end = time.time()
print("Duration: ", time_end - time_start, "s")
"""
# print("Total distinct words number :  " + str(word_count.count()))
# print("Total words number :  " + str(total_words_number))
# print("Total distinct rarewords number :  " + str(rare_word.count()))
"""
