# 1 10
# 2 10
# 3 5
# 4 8
# 5 15
CASE_NO=$1
K=$2
python bfr.py "./Dataset/test"${CASE_NO} ${K} "cluster.json" "intermediate.csv"
python evaluate_clusters.py "cluster"${CASE_NO}".json" "cluster.json"