clear
echo "*** This is the Task 1: "
spark-submit task1.py 1 4 ./Dataset/small2.csv ./task1-result.txt
echo "*** This is the Task 2: "
spark-submit task2.py 1 50 ./user_business.csv ./task2-result.txt