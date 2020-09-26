"""
    USC Spring 2020
    INF 553 Foundations of Data Mining
    Assignment 5
    
    Student Name: Jiabin Wang
    Student ID: 4778-4151-95
"""

from Package import math, load_data, time, SparkContext, random

class _KMeans():
    # train_data = []
    # K = 1
    # tol = 1e-4
    # max_iter = 300
    def __init__(self, K = 10, tol=1e-4, max_iter=300):
        self.K = K
        self.tol = tol 
        self.max_iter = max_iter

    def euclidean_distance(self, a, b):
        a = a[1:]
        b = b[1:]
        c = [pow(i - j,2) for i, j in zip(a, b)]
        return math.sqrt(sum(c))
    
    def initialise_centroid(self, train_data, K):
        # random.seed(0)
        centroid = []
        centroid_set = set()


        while K > 0:
            index = random.randrange(len(train_data))
            candidate = train_data[index]
            if index in centroid_set:
                continue 
            K -= 1
            centroid.append(candidate)
            centroid_set.add(index)

        '''
        centroid.append(train_data[0])
        centroid_set.add(train_data[0][0])
        K = K - 1
        while K > 0:
            K = K - 1
            globalMax = 0
            candidate = []
            for point in train_data:
                if point[0] in centroid_set:
                    continue
                localMax = 0
                for center in centroid:
                    localMax = localMax + self.euclidean_distance(point, center)
                if localMax > globalMax:
                    globalMax = localMax
                    candidate = point
            centroid.append(candidate)
            centroid_set.add(candidate[0])
        '''
        print("Finish Centroid")
        return centroid

    def find_nearest_center(self, point, centroid):
        min_distance = self.euclidean_distance(point, centroid[0])
        min_center = centroid[0]
        for i in range(1, len(centroid)):
            cur_distance = self.euclidean_distance(point, centroid[i])
            if cur_distance < min_distance:
                min_distance = cur_distance
                min_center = centroid[i]
        return min_center

    def update_centroid(self, old_centroid, old_centroid_label, train_data, predict_data):
        old_centroid_label_reverse = {}
        # label -> centroid
        for old_centroid in old_centroid_label:
            old_centroid_label_reverse[old_centroid_label[old_centroid]] = old_centroid
        label_centroid_computing = {}
        # label -> centroid
        
        for point in train_data:
            if predict_data[point[0]] not in label_centroid_computing:
                label_centroid_computing[predict_data[point[0]]] = {"num" : point, "count" : 1}
            else:
                a = label_centroid_computing[predict_data[point[0]]]["num"]
                b = point
                c = [i + j for i, j in zip(a, b)]
                c[0] = 0
                label_centroid_computing[predict_data[point[0]]]["num"] = c
                label_centroid_computing[predict_data[point[0]]]["count"] = label_centroid_computing[predict_data[point[0]]]["count"] + 1
        new_centroid = []
        new_centroid_label = {}
        for label in label_centroid_computing:
            center = [i/label_centroid_computing[label]["count"] for i in label_centroid_computing[label]["num"]]
            center[0] = 0
            new_centroid_label[tuple(center)] = label
            new_centroid.append(center)
        
        centroid_moving = 0

        for center in new_centroid_label:
            label = new_centroid_label[center]
            old_center = old_centroid_label_reverse[label]
            centroid_moving = centroid_moving + self.euclidean_distance(old_center, list(center))
        
        return new_centroid, new_centroid_label, centroid_moving
        
    def fit(self, X):
        self.train_data = X
        train_data = X
        K = self.K
        centroid = self.initialise_centroid(train_data, K)
        iter_time = 0
        predict_data = [0] * len(train_data)
        label = 1
        centroid_label = {}
        # initialise the label
        for center in centroid:
            centroid_label[tuple(center)] = label
            label = label + 1
        while iter_time < self.max_iter:
            centroid_moving = 0
            iter_time = iter_time + 1
            # Find the nearest center and label it
            for i in range(0, len(train_data)):
                center = self.find_nearest_center(train_data[i], centroid)
                predict_data[train_data[i][0]] = centroid_label[tuple(center)]

            # Update centroids
            centroid, centroid_label, centroid_moving = self.update_centroid(centroid, centroid_label, train_data, predict_data)
            print("==> Iteration Time: " + str(iter_time))
            print("    Centroid Moving Distance: " + str(centroid_moving))
            if centroid_moving < tol:
                for cen in centroid:
                    print(cen)
                break
        return predict_data


if __name__ == "__main__":
    time_start = time.time()
    # THRESHOLD = int(sys.argv[1])
    # input_file_path = sys.argv[2]
    # output_file_path = sys.argv[3]
    K = 10
    tol = 1e-4
    train_data = []
    paths = ["./Dataset/test1/data0.txt", "./Dataset/test1/data1.txt", "./Dataset/test1/data2.txt", "./Dataset/test1/data3.txt", "./Dataset/test1/data4.txt"]

    for input_file_path in paths:
        train_data = train_data + load_data(input_file_path)
    # input_file_path = "./Dataset/test1/data0.txt"
    # train_data = load_data(input_file_path)
    
    
    # random.seed(0)
    # X = []
    # scikit_X = []
    # for i in range(14000):
    #     a = random.uniform(0, 10)
    #     b = random.uniform(0, 10)
    #     x = [i, a, b]
    #     X.append(x)
    #     x = [a,b]
    #     scikit_X.append(x)

    scikit_X = train_data
    for i in range(len(scikit_X)):
        scikit_X[i][0] = 0

    time_end = time.time()
    print("Duration: ", time_end - time_start, "s")