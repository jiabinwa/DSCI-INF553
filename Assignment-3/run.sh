# python task1.py ./Dataset/train_review.json ./task1.res

# python task2train.py ./Dataset/train_review.json ./task2.model ./Dataset/stopwords
# python task2predict.py ./Dataset/test_review.json ./task2.model ./output

rm task3item.model
rm task3item.predict
python task3train.py ./Dataset/train_review.json ./task3item.model item_based
python task3predict.py ./Dataset/train_review.json ./Dataset/test_review_ratings.json ./task3item.model ./task3item.predict item_based


spark-submit task3rmse.py ./Dataset/test_review_ratings.json ./task3item.predict ./Dataset/business_avg.json item_based


spark-submit task3train.py ./Dataset/train_review.json ./task3user.model user_based