'''
    USC Spring 2020
    INF 553 Foundations of Data Mining
    Assignment 1
    
    Student Name: Jiabin Wang
'''
from pyspark import SparkContext
import os
import re
import sys
import json
import time
import heapq


def parseReviewInfo(line):
    '''
    Function: 
        Extract the variables business_id and stars from JSON file
        Form the tuple (business_id, stars)

    Parameter:
        One JSON object: one line from the JSON file
    
    Return:
        A tuple: (business_id, stars)
    '''
    return (json.loads(line)['business_id'],json.loads(line)['stars'])

def parseBusinessInfo(line):
    '''
    Function: 
        Extract the variables business_id and categories from JSON file
        Form the tuple (business_id, categories)

    Parameter:
        One JSON object: one line from the JSON file
    
    Return:
        A tuple: (business_id, categories)
    '''
    return (json.loads(line)['business_id'],json.loads(line)['categories'])

def validCategories(line):
    '''
    Function: 
        Verify whether the categories variable is valid
        If this variable is None in the JSON, it is invalid

    Parameter:
        One JSON object: one line from the JSON file
    
    Return:
        True: Valid
        False: Invalid
    '''
    if json.loads(line)['categories'] is None:
        return False
    else:
        return True

def segmentCateg(line):
    '''
    Function: 
        Extract all categories from the 'categories' text
        Eg. "Chinese, Indian, English" => ["Chinese", "Indian", "English"] 

    Parameter:
        One JSON object: one line from the JSON file
    
    Return:
        A list of tuples: (category, (1, stars))
    '''
    categories = re.split(r'(\s*[,，]\s*)',line[1][1].strip())
    result = []
    for category in categories:
        result.append((category,(1,line[1][0])))
    return result

def saveToFile(result,result_file_path):
    '''
    Function: 
        The function properly saving the data to the output file
        in valid JSON
        
    Parameter:
        result: the data
        result_file_path: the output file
    
    Return:
        Null
    '''
    with open(result_file_path,'w') as File:
        json.dump(result,File)

# Spark Version
def sparkVersion(review_file_path, business_file_path,topN):
    '''
    Function: 
        1. Read the review file and business file and transform
            them into the RDD format
        2. Extract the useful information
            variable: reviewRDD : (businessId, stars)
            variable: businessRDD : (businessId, categories)
        3. Perform join operation through RDD.join() function
            to build the new RDD (businessId, stars, categories)
        4. split the monolithic categories text into many categories texts
        5. Compute the average stars
        6. Ranking
        7. Output the topN categories
        
    Parameter:
        review_file_path: the path of review file 
        result_file_path: the path of output file
        topN: number
    
    Return:
        List: [(category, average stars)]
    '''
    sc = SparkContext('local[*]', 'task2')
    sc.setLogLevel("ERROR")
    raw_reviewRDD = sc.textFile(review_file_path)
    raw_businessRDD = sc.textFile(business_file_path)
    reviewRDD = raw_reviewRDD.map(lambda line : parseReviewInfo(line))

    businessRDD = raw_businessRDD.filter(lambda line: validCategories(line))\
                                .map(lambda line: parseBusinessInfo(line))

    joinRDD = reviewRDD.join(businessRDD)\
                    .flatMap(lambda line: segmentCateg(line))\
                    .reduceByKey(lambda a,b : (a[0] + b[0], a[1] + b[1]))\
                    .map(lambda line: (line[0],line[1][1]/line[1][0]))\
                    .sortBy(lambda a: (-a[1],a[0])).take(topN)
    result = []
    for item in joinRDD:
        result.append([item[0],item[1]])
    return result

def nonSparkVersion(review_file_path, business_file_path,topN):
    '''
        Same logic with function sparkVersion()
        Except performing the join procedure through
        the naive approach: build it one by one

        This function does not use Spark 
    '''
    # Non-Spark Version
    id2categories = {} # business_id => categories
    with open(business_file_path) as File:
        for line in File:
            categories = json.loads(line)['categories']
            if categories is None:
                continue
            business_id = json.loads(line)['business_id']
            id2categories[business_id] = categories


    categories2stars = {} # categories => stars

    with open(review_file_path) as File:
        for line in File:
            business_id = json.loads(line)['business_id']
            stars = json.loads(line)['stars']
            if business_id not in id2categories:
                continue
            categories = id2categories[business_id]
            categories = re.split(r'(\s*[,，]\s*)',categories.strip())
            for category in categories:
                numSum = categories2stars.get(category,[0,0])
                numSum[0] = numSum[0] + 1
                numSum[1] = numSum[1] + stars
                categories2stars[category] = numSum

    result = []        
    for category, numSum in categories2stars.items():
        result.append([category, numSum[1]/numSum[0]])

    result = heapq.nsmallest(topN, result, key = lambda tuple : (-tuple[1],tuple[0]))
    return result

def main():
    time_start=time.time()
    review_file_path = sys.argv[1]
    business_file_path = sys.argv[2]
    result_file_path = sys.argv[3]
    mode = sys.argv[4]
    topN = int(sys.argv[5])

    result = {}
    if mode == "spark":
        result['result'] = sparkVersion(review_file_path, business_file_path,topN)
    else:
        result['result'] = nonSparkVersion(review_file_path, business_file_path,topN)

    saveToFile(result,result_file_path)

    time_end=time.time()
    print('time cost',time_end-time_start,'s')

if __name__ == "__main__":
    main()

