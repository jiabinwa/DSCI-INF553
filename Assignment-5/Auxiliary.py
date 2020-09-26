"""
    USC Spring 2020
    INF 553 Foundations of Data Mining
    Assignment 5
    
    Student Name: Jiabin Wang
    Student ID: 4778-4151-95
"""

from Package import math, random, csv, json, os, multiprocessing

def load_data(input_file_path):
    train_data = []
    for line in open(input_file_path):
        datum = list(map(lambda x : float(x), line.strip('\n').split(",")))
        datum[0] = int(datum[0])
        train_data.append(datum)
    return train_data

def make_predict_dict(K, X, predict_data):
    result = {}
    for i in range(K):
        result[i] = []
    for i in range(len(X)):
        result[predict_data[i]].append(X[i])
    return result

def output_intermediate_result(intermediate, output_file_path):
    titles = ["round_id", "nof_cluster_discard", "nof_point_discard", "nof_cluster_compression", "nof_point_compression", "nof_point_retained"]
    content = []
    for index in range(1, len(intermediate) + 1):
        tmp = []
        for t in titles:
            tmp.append(intermediate[index][t])
        content.append(tmp)
    
    with open(output_file_path,"w") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(titles)
        writer.writerows(content)

def output_cluster_result(DS_set, RS_set, output_file_path):
    middle = []
    data = {}
    for ds_label in DS_set:
        for point_index in DS_set[ds_label].points_index_buffer:
            middle.append((point_index, int(ds_label)))
            # data[str(point_index)] = int(ds_label)
    for point in RS_set:
        middle.append((point[0], -1))
        # data[str(point[0])] = int(-1)
    middle = sorted(middle, key=lambda x: x[0])
    for pair in middle:
        data[str(pair[0])] = pair[1]
    with open(output_file_path, 'w') as fw:
        json.dump(data,fw)

def inverse_index(X):
    inverseMap = {}
    for i in range(len(X)):
        inverseMap[tuple(X[i])] = i
    return inverseMap

class DiscardSet():
    '''
    Members:
    LABEL: the label for this DS. When it is CS => -1
    D: the dimension of every point: dimension + 1: [0] is an index
    N: the total counts of points in this Set
    stale: True when statistics is not fresh => update statistics
    centroid: [statistics]
    variance: [statistics] standard variance
    SUM = []
    SUMSQ = []
    points_index_buffer = [] 
    '''
    def __init__(self, label, points):
        self.LABEL = label
        self.D = len(points[0])
        self.N = 0
        self.stale = False
        self.SUM = [0.0] * self.D
        self.SUMSQ = [0.0] * self.D
        self.points_index_buffer = []
        self.merge_points(points)
        self.SUMSQ[0] = 0 # [0] is always 0
        self.SUM[0] = 0   # [0] is always 0
        self.update_statistics()
    
    def merge_points(self, points):
        self.N += len(points)
        for p in points:
            self.points_index_buffer.append(p[0])
            for i in range(1,self.D):
                self.SUM[i] += p[i]
                self.SUMSQ[i] += p[i] ** 2
            # self.SUM = [i + j for i, j in zip(self.SUM, p)]
            # self.SUMSQ = [i + j * j for i, j in zip(self.SUMSQ, p)]
        self.stale = True
    
    def merge_Cluster(self, cluster):
        self.N += cluster.N
        self.SUM += cluster.SUM
        self.SUMSQ += cluster.SUMSQ
        self.points_index_buffer += cluster.points_index_buffer
        self.stale = True
        
    def update_statistics(self):
        centroid = [0] * self.D
        variance = [0] * self.D
        for i in range(1, self.D):
            centroid[i] = self.SUM[i] / self.N
            # variance[i] = (self.SUMSQ[i] / self.N) - math.pow((self.SUM[i] / self.N),2)
            # variance[i] = math.sqrt(variance[i])
            variance[i] = math.sqrt((self.SUMSQ[i] / self.N) - (self.SUM[i] / self.N) ** 2)
        self.stale = False
        self.centroid, self.variance = centroid, variance
    
    def mahalanobis_distance(self, point):
        if(self.stale):
            self.update_statistics()
        centroid, variance = self.centroid, self.variance
        distance = 0
        for i in range(1,self.D):
            distance = distance + ((point[i] - centroid[i]) / variance[i]) ** 2 
        distance = math.sqrt(distance)
        return distance


class _KMeans():
    def __init__(self, K = 10, tol=1e-4, max_iter=100, init='random', n_init=5, verbose=False, compulsory=False):
        self.K = K
        self.tol = tol 
        self.max_iter = max_iter
        self.init = init
        self.n_init = n_init
        self.verbose = verbose
        self.compulsory = compulsory
    
    def run(self, pid, X):
        kmeans = _KMeans_impl(K=self.K, tol = self.tol, max_iter=self.max_iter, init=self.init, verbose=self.verbose, compulsory=self.compulsory)
        Y = kmeans.fit(X)
        score = kmeans.evaluate_coherence()
        del(kmeans)
        # print(score)
        return (Y, score)


    def fit(self, X):
        multiprocessing.freeze_support()
        pool = multiprocessing.Pool()
        cpus = self.n_init
        results = []
        for i in range(0, cpus):
            result = pool.apply_async(self.run, args=(i, X,))
            results.append(result)

        pool.close()
        pool.join()
        resultList = [result.get() for result in results]
        resultList = sorted(resultList, key=lambda x:x[1])
        return resultList[0][0]
        


class _KMeans_impl():
    def __init__(self, K = 10, tol=1e-4, max_iter=100, init='random', verbose=False, compulsory=False):
        self.K = K
        self.tol = tol 
        self.max_iter = max_iter
        self.init = init
        self.verbose = verbose
        self.compulsory = compulsory
        self.centroids = []
        self.centroids_hashmap = {}

    def euclidean_distance(self, a, b):
        a = a[1:]
        b = b[1:]
        c = [pow(i - j,2) for i, j in zip(a, b)]
        return math.sqrt(sum(c))
    
    def initialise_centroid(self, train_data, K):
        centroid = []
        centroid_set = set()
        if self.init == 'random':
            while K > 0:
                index = random.randrange(len(train_data))
                candidate = train_data[index]
                if index in centroid_set:
                    continue 
                K -= 1
                centroid.append(candidate)
                centroid_set.add(index)
        
        else:
            index = random.randrange(len(train_data))
            centroid.append(train_data[index])
            centroid_set.add(train_data[index][0])
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
        self.centroids = centroid
    
    def make_centroid_hashmap(self):
        centroids_hashmap = {}
        for center in self.centroids:
            centroids_hashmap[tuple(center)] = []
        self.centroids_hashmap = centroids_hashmap

    def find_nearest_centroid(self, point):
        min_distance = self.euclidean_distance(point, self.centroids[0])
        min_center = self.centroids[0]
        for i in range(1, len(self.centroids)):
            cur_distance = self.euclidean_distance(point, self.centroids[i])
            if cur_distance < min_distance:
                min_distance = cur_distance
                min_center = self.centroids[i]
        return min_center

    def update_centroid(self):
        new_centroids = []
        for centroid in self.centroids:
            count = len(self.centroids_hashmap[tuple(centroid)])
            if count == 0:
                new_centroids.append(centroid)
                continue
            center = [0] * len(centroid)
            for point in self.centroids_hashmap[tuple(centroid)]:
                center = [ i + j for i,j in zip(center, point)]
            center = [i/count for i in center]
            center[0] = 0
            new_centroids.append(center)
        
        centroid_moving = 0
        for i in range(len(new_centroids)):
            centroid_moving += self.euclidean_distance(new_centroids[i], self.centroids[i])
        self.centroids = new_centroids
        return centroid_moving
    
    def make_predict(self, X):
        index_map = inverse_index(X)
        Y = [0] * len(X)
        label = 0
        for centroid in self.centroids_hashmap:
            for point in self.centroids_hashmap[centroid]:
                Y[index_map[tuple(point)]] = label
            label += 1
        return Y

    def evaluate_coherence(self):
        # coherence = 0
        # for centroid in self.centroids_hashmap:
        #     temp_sum = 0
        #     if self.compulsory and len(self.centroids_hashmap[centroid]) == 0:
        #         return float('inf')
        #     for point in self.centroids_hashmap[centroid]:
        #         # coherence += self.euclidean_distance(centroid, point)
        #         temp_sum += self.euclidean_distance(centroid, point)
        #     coherence += temp_sum #/len(self.centroids_hashmap[centroid])
        # return coherence
        for centroid in self.centroids_hashmap:
            if self.compulsory and len(self.centroids_hashmap[centroid]) == 0:
                return float('inf')

        non_empty_centroids_hashmap = {}
        for centroid in self.centroids_hashmap:
            if len(self.centroids_hashmap[centroid]) > 0:
                non_empty_centroids_hashmap[centroid] = self.centroids_hashmap[centroid]
        
        S = {}
        for centroid in non_empty_centroids_hashmap:
            temp_sum = 0
            for point in non_empty_centroids_hashmap[centroid]:
                temp_sum += self.euclidean_distance(centroid, point)
            S[centroid] = temp_sum / len(non_empty_centroids_hashmap[centroid])
        
        D = {}
        for centroid in non_empty_centroids_hashmap:
            if centroid not in D:
                D[centroid] = []
            for c in S:
                if c != centroid:
                    D[centroid].append((S[centroid] + S[c])/self.euclidean_distance(centroid, c))
        
        totalSum = 0
        for centroid in D:
            totalSum += max(D[centroid])
        
        return totalSum/len(D)
        


    def fit(self, X):
        K = self.K
        train_data = X
        self.initialise_centroid(train_data, K)
        iter_time = 0
        while iter_time < self.max_iter:
            iter_time = iter_time + 1
            self.make_centroid_hashmap() # reset the hashmap

            # Find the nearest center and add it to the hashmap
            for point in train_data:
                nearest_centroid = self.find_nearest_centroid(point)
                self.centroids_hashmap[tuple(nearest_centroid)].append(point)

            # Update centroids
            centroid_moving = self.update_centroid()
            if self.verbose:
                print("==> Iteration Time: " + str(iter_time))
                print("    Centroid Moving Distance: " + str(centroid_moving))
            if centroid_moving < self.tol:
                break
        Y = self.make_predict(X)
        return Y