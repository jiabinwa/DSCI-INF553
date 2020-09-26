'''
    USC Spring 2020
    INF 553 Foundations of Data Mining
    Assignment 2
    
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
import csv
import argparse

def parseReviewInfo(line):
    return (json.loads(line)['business_id'], json.loads(line)['user_id'])

def parseBusinessInfo(line):
    return (json.loads(line)['business_id'], json.loads(line)['state'])

def validState(line):
    if json.loads(line)['state'] == "NV":
        return True
    else:
        return False

sc = SparkContext('local[*]', 'csv')
sc.setLogLevel("ERROR")

review_file_path = './Dataset/review.json'
business_file_path = './Dataset/business.json'

raw_reviewRDD = sc.textFile(review_file_path)
raw_businessRDD = sc.textFile(business_file_path)

reviewRDD = raw_reviewRDD.map(lambda line : parseReviewInfo(line))

businessRDD = raw_businessRDD.filter(lambda line: validState(line)).map(lambda line: parseBusinessInfo(line))

joinRDD = businessRDD.join(reviewRDD).map(lambda x: (x[1][1], x[0])).collect()

header = ["user_id" , "business_id"]
with open('user_business.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(header)
    for pair in joinRDD:
        writer.writerow(pair)