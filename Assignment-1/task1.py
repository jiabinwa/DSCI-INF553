'''
    USC Spring 2020
    INF 553 Foundations of Data Mining
    Assignment 1
    
    Student Name: Jiabin Wang
'''
from pyspark import SparkContext
import os
import re
import json
import time
import sys

# ======================== Auxiliary Functions ========================

def parseDateToYear(line):
    '''
    Function: 
        Extract the variable year From json file

    Parameter:
        One JSON object: one line from the JSON file
    
    Return:
        Number: the year
    '''
    return json.loads(line)['date'][:4]

def parseUserID(line):
    '''
    Function: 
        Extract the variable user ID From json file

    Parameter:
        One JSON object: one line from the JSON file
    
    Return:
        String: the user ID
    '''
    return json.loads(line)['user_id']

def BFilter(line,year):
    '''
    Function: 
        Judge if the line contains the critical year
    
    Parameter:
        line: one JSON object; one line from the JSON file
        year: the critical year that plays as criterion
    
    Return:
        True: Yes, it contains
        False: No, it does not
    '''
    if parseDateToYear(line) == year:
        return True
    else:
        return False

def parseText(line):
    '''
    Function: 
        Extract the review text from the JSON line, 
        and remove all the characters that are ,
        neither numerical nor alphabetical. 
    
    Parameter:
        One JSON object: one line from the JSON file

    Return:
        String: Review text content
    '''
    text = json.loads(line)['text']
    text = text.lower()
    text = re.sub('[^A-Za-z0-9 ]', ' ', text)
    return text.split(" ")

def EFilter(word,punctuations,stopwords):
    '''
    Function: 
        Extract the review text from the JSON line, 
        and remove all the characters that are ,
        neither numerical nor alphabetical. 
    
    Parameter:
        One JSON object: one line from the JSON file
    
    Return:
        String: Review text content
    '''
    if word not in punctuations and word not in stopwords and word != ' ' and word != '':
        return True
    else:
        return False

def A(textRDD):
    '''
    Function: 
        The function for task A:
            compute the total number of reviews
    
    Parameter:
        The RDD that contains the JSON file
    
    Return:
        Number: the total number of reviews
    '''
    line = textRDD.map(lambda l: ("count",1))
    line.persist()
    total = line.reduceByKey(lambda a,b: a + b).collect()
    return total[0][1]

def B(textRDD,year):
    '''
    Function: 
        The function for task B:
            compute the number of reviews in a given year
    
    Parameter:
        textRDD: The RDD that contains the review file
        year: the year wanted to be queried
    
    Return:
        Number: the total number of reviews in a given year
    '''
    line = textRDD.filter(lambda l: BFilter(l,year)).map(lambda l: (parseDateToYear(l),1))
    line.persist()
    total = line.reduceByKey(lambda a,b: a + b).sortByKey().collect()
    return total[0][1]

def C(textRDD):
    '''
    Function: 
        The function for task C:
            compute the number of distinct users who have written the reviews
    
    Parameter:
        textRDD: The RDD that contains the review file
    
    Return:
        Number: the number of distinct users who have written the reviews
    '''
    line = textRDD.map(lambda l: (parseUserID(l),1)) \
            .reduceByKey(lambda a,b: a + b).map(lambda a: ("User",1))\
            .reduceByKey(lambda a,b: a + b).collect()
    return line[0][1]

def D(textRDD,topM):
    '''
    Function: 
        The function for task D:
            Collect Top M users who have the largest number of reviews and its count 
    
    Parameter:
        textRDD: The RDD that contains the review file
        topM: a number
    
    Return:
        List: user list
    '''
    line = textRDD.map(lambda l: (parseUserID(l),1))\
                .reduceByKey(lambda a,b: a + b)\
                .sortBy(lambda a: a[1],False).take(topM)
    result = []
    for item in line:
        result.append([item[0],item[1]])
    return result


def E(textRDD,topN,punctuations,stopwords):
    '''
    Function: 
        The function for task E:
            Collect the Top n frequent words in the review text. 
                    The words should be in lower cases. The following punctuations 
                    i.e., “(”, “[”, “,”, “.”, “!”, “?”, “:”, “;”, “]”, “)”
                    , and the given stopwords are excluded (1pts)
        
    Parameter:
        textRDD: The RDD that contains the review file
        topN: a number
        punctuations: a list contains all punctuations
        stopwords: a list contains all stopwords
    
    Return:
        List: a word list
    '''
    line = textRDD.flatMap(lambda line : parseText(line))\
                .filter(lambda word: EFilter(word,punctuations,stopwords))\
                .map(lambda word : (word,1))\
                .reduceByKey(lambda a,b : a + b)\
                .sortBy(lambda a: (-a[1],a[0])).take(topN)
    result = []
    for item in line:
        result.append(item[0])
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


def main():
    time_start=time.time()
    input_file_path = sys.argv[1]
    result_file_path = sys.argv[2]
    stopwords_file_path = sys.argv[3]
    year = sys.argv[4]
    topM = int(sys.argv[5])
    topN = int(sys.argv[6])

    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel("ERROR")
    punctuations = ["(", "[", ",", ".", "!", "?", ":", ";", "]", ")"]
    stopwords = sc.textFile(stopwords_file_path).collect()
    textRDD = sc.textFile(input_file_path)

    result = {}
    result['A'] = A(textRDD)
    result['B'] = B(textRDD,year)
    result['C'] = C(textRDD)
    result['D'] = D(textRDD,topM)
    result['E'] = E(textRDD,topN,punctuations,stopwords)

    saveToFile(result,result_file_path)

    time_end=time.time()
    print('time cost',time_end-time_start,'s')

if __name__ == "__main__":
    main()