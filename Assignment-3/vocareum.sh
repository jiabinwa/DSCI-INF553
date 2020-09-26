
spark-submit task1.py $ASNLIB/publicdata/train_review.json task1.re
spark-submit task2train.py $ASNLIB/publicdata/train_review.json task2.model $ASNLIB/publicdata/stopwords
spark-submit task2predict.py $ASNLIB/publicdata/test_review.json task2.model task2.predict
spark-submit task3train.py $ASNLIB/publicdata/train_review.json task3item.model item_based
spark-submit task3train.py $ASNLIB/publicdata/train_review.json task3user.model user_based
spark-submit task3predict.py $ASNLIB/publicdata/train_review.json $ASNLIB/publicdata/test_review.json task3item.model task3item.predict item_based
spark-submit task3predict.py $ASNLIB/publicdata/train_review.json $ASNLIB/publicdata/test_review.json task3user.model task3user.predict user_based