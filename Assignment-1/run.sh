echo "*** This is the Task 1: "
spark-submit task1.py ./Dataset/review.json task1-result.json ./Dataset/stopwords 2017 10 10

echo "*** This is the Task 2: "
spark-submit task2.py ./Dataset/review.json ./Dataset/business.json task2-result.json no_spark 50

echo "*** This is the Task 3: " # default customized
python task3.py ./Dataset/review.json task3-result.json customized 2 1000
