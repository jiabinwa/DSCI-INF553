"""
    USC Spring 2020
    INF 553 Foundations of Data Mining
    Assignment 5
    
    Student Name: Jiabin Wang
    Student ID: 4778-4151-95
"""
from Package import time, math, os, itertools, copy, json, sys
from Auxiliary import (
    _KMeans,
    DiscardSet,
    make_predict_dict,
    output_intermediate_result,
    output_cluster_result,
    load_data,
)

# TODO: Delete these
# from sklearn.cluster import KMeans
# import numpy as np
# import warnings
# warnings.filterwarnings("ignore")


def useScikitKmeans(X, K):
    # SciX = [i[:10] for i in X]
    # SciX = np.array(SciX)
    # kmeans = KMeans(n_clusters=K, init="random", tol=1e-4, n_init=10).fit(SciX)
    # SciY = kmeans.labels_
    # Y = SciY.tolist()
    keamns = _KMeans(K=K, tol = 1)
    Y = keamns.fit(X)
    return Y


def displaySet(DS_set):
    return
    print("==> Information of DS Sets")
    for label in DS_set:
        print(DS_set[label].N)
    print("---------------------------")


def load_all_data(paths):
    train_data = []
    for input_file_path in paths:
        train_data = train_data + load_data(input_file_path)
    return train_data


def initialise_DS(X, Y, K):
    middle = {}
    DS_set = {}
    for i in range(K):
        middle[i] = []
    for i in range(len(X)):
        middle[Y[i]].append(X[i])
    for label in middle:
        DS_set[label] = DiscardSet(label, middle[label])
    return DS_set


def find_nearest_DS(DS_set, alpha, point):
    dimension = DS_set[0].D - 1
    THRESHOLD = alpha * math.sqrt(dimension)
    distance = float("inf")
    _label = -1
    for label in DS_set:
        dist = DS_set[label].mahalanobis_distance(point)
        if dist < distance:
            distance = dist
            if dist < THRESHOLD:
                _label = label
    return _label


def find_nearest_CS(CS_set, alpha, point):
    # CS_set is a list
    if len(CS_set) == 0:
        return -1
    dimension = CS_set[0].D - 1
    THRESHOLD = alpha * math.sqrt(dimension)
    distance = float("inf")
    _index = -1
    for index in range(len(CS_set)):
        dist = CS_set[index].mahalanobis_distance(point)
        if dist < distance and dist < THRESHOLD:
            distance = dist
            _index = index
    return _index


def sample_data(train_data, sample_ratio, beta):
    _bar = int(len(train_data) * sample_ratio)
    dim = len(train_data[0])
    remains = train_data[_bar:]
    sample_candidate = train_data[:_bar]
    dummy_set = DiscardSet("DUMMY", sample_candidate)
    variance = dummy_set.variance
    centroid = dummy_set.centroid
    sample = []
    possible_outlier = []
    not_select = 0
    for point in sample_candidate:
        is_select = True
        for i in range(1, dim):
            if abs(point[i] - centroid[i]) > beta * variance[i]:
                not_select += 1
                possible_outlier.append(point)
                is_select = False
                break
        if is_select:
            sample.append(point)
    del(dummy_set)
    # print("We drop " + str(not_select) + " data")
    remains = remains + possible_outlier
    return remains, sample


class BFR:
    def process_one_round(self, train_data, next_line, ROUND_BATCH):
        X = train_data[next_line : next_line + ROUND_BATCH]
        ds_batch_process_add = {}
        cs_batch_process_add = {}

        for point in X:
            nearest_DS_label = find_nearest_DS(self.DS_set, self.alpha, point)
            if nearest_DS_label >= 0:  # valid: If they can enter DS ==> enter DS
                if nearest_DS_label not in ds_batch_process_add:
                    ds_batch_process_add[nearest_DS_label] = []
                ds_batch_process_add[nearest_DS_label].append(point)
            else:
                # If they can enter CS ==> enter CS
                nearest_CS_index = find_nearest_CS(self.CS_set, self.alpha, point)
                if nearest_CS_index >= 0:
                    if nearest_CS_index not in cs_batch_process_add:
                        cs_batch_process_add[nearest_CS_index] = []
                    cs_batch_process_add[nearest_CS_index].append(point)
                else:
                    # Else: enter RS
                    self.RS_set.append(point)
        for _label in ds_batch_process_add:
            self.DS_set[_label].merge_points(ds_batch_process_add[_label])
        for _index in cs_batch_process_add:
            self.CS_set[_index].merge_points(cs_batch_process_add[_index])

    def make_CS(self):
        if len(self.RS_set) <= 3 * self.K:
            return
        # print("=> We need to make more CS at here")
        X = self.RS_set
        keamns = _KMeans(K= 3 * self.K, tol = 1, n_init=1, init="kmeans++")
        Y = keamns.fit(X)
        dict_res = make_predict_dict(K * 3, self.RS_set, Y)
        self.RS_set = []
        for l in dict_res:
            if len(dict_res[l]) == 1:
                self.RS_set.append(dict_res[l][0])
            elif len(dict_res[l]) > 1:
                self.CS_set.append(DiscardSet("CS", dict_res[l]))
        '''
        print(
            "=> After CS Making: we still have  "
            + str(len(self.RS_set))
            + " points in RS"
        )
        '''

    def merge_CS(self):
        alpha = 1 # self.alpha
        if len(self.CS_set) == 0:
            return
        dimension = self.CS_set[0].D
        exist = [True] * len(self.CS_set)
        THRESHOLD = alpha * math.sqrt(dimension) # a small alpha here
        index_combination = list(
            itertools.combinations(list(range(len(self.CS_set))), 2)
        )
        index_combination_distance = []
        cs_batch_process_merge = {}
        for i in range(len(self.CS_set)):
            self.CS_set[i].update_statistics()
        for (s, t) in index_combination:
            s_centroid = self.CS_set[s].centroid
            distance = self.CS_set[t].mahalanobis_distance(s_centroid)
            index_combination_distance.append((s, t, distance))
        index_combination_distance = sorted(
            index_combination_distance, key=lambda x: (x[2], x[0], x[1])
        )

        for (s, t, dis) in index_combination_distance:
            if dis > THRESHOLD:
                break
            if exist[s] == False:
                continue
            if dis < THRESHOLD:
                exist[s] = False
                if t not in cs_batch_process_merge:
                    cs_batch_process_merge[t] = []
                cs_batch_process_merge[t].append(s)

        for i in range(len(self.CS_set)):
            if i not in cs_batch_process_merge:
                continue
            for j in range(len(self.CS_set)):
                if j in cs_batch_process_merge and i in cs_batch_process_merge[j]:
                    cs_batch_process_merge[j] += cs_batch_process_merge[i]
                    del cs_batch_process_merge[i]

        remove_index_list = set()
        for dest in cs_batch_process_merge:
            for source in cs_batch_process_merge[dest]:
                self.CS_set[dest].merge_Cluster(self.CS_set[source])
                remove_index_list.add(source)

        # print("[Before]==> The number of CS is : " + str(len(self.CS_set)))
        # print(cs_batch_process_merge)
        tmp = copy.deepcopy(self.CS_set)
        length = len(self.CS_set)
        del self.CS_set
        self.CS_set = []
        for i in range(length):
            if i not in remove_index_list:
                self.CS_set.append(tmp[i])
        del tmp
        # print("[After]==> The number of CS is : " + str(len(self.CS_set)))

    def information_summary(self):
        self.intermediate_info[self.round_no] = {}
        self.intermediate_info[self.round_no]["round_id"] = self.round_no
        self.intermediate_info[self.round_no]["nof_cluster_discard"] = len(self.DS_set)
        self.intermediate_info[self.round_no]["nof_point_discard"] = 0
        for label in self.DS_set:
            self.intermediate_info[self.round_no]["nof_point_discard"] += self.DS_set[
                label
            ].N
        self.intermediate_info[self.round_no]["nof_cluster_compression"] = len(
            self.CS_set
        )
        self.intermediate_info[self.round_no]["nof_point_compression"] = 0
        for index in range(len(self.CS_set)):
            self.intermediate_info[self.round_no][
                "nof_point_compression"
            ] += self.CS_set[index].N
        self.intermediate_info[self.round_no]["nof_point_retained"] = len(self.RS_set)

    def initialisation(self):
        # print("===> Now Operate on File: " + str(self.files[self.current_file_index]))
        self.round_no += 1
        train_data = load_data(self.files[self.current_file_index])
        self.current_file_index += 1
        train_data, sample = sample_data(train_data, self.sample_ratio, self.beta)
        ROUND_BATCH = int(len(train_data) * 0.1)
        DATA_LENGTH = len(train_data)
        X = sample
        keamns = _KMeans(K=self.K, tol = 1, compulsory=True, n_init=15)
        Y = keamns.fit(X)
        self.DS_set = initialise_DS(X, Y, self.K)
        # print("**** Initialisation Begin ****")
        displaySet(self.DS_set)

        next_line = 0
        while next_line < DATA_LENGTH:
            self.process_one_round(train_data, next_line, ROUND_BATCH)
            next_line += ROUND_BATCH  # Save

        
        self.make_CS()
        self.information_summary()
        # print("**** End of First Round  ****")

    def rounds(self):
        if self.current_file_index >= len(self.files):
            return
        # print("===> Now Operate on File: " + str(self.files[self.current_file_index]))
        self.round_no += 1
        train_data = load_data(self.files[self.current_file_index])
        self.current_file_index += 1
        ROUND_BATCH = int(len(train_data) * 0.1)
        DATA_LENGTH = len(train_data)
        next_line = 0
        while next_line < DATA_LENGTH:
            self.process_one_round(train_data, next_line, ROUND_BATCH)
            next_line += ROUND_BATCH

        self.merge_CS()
        self.make_CS()
        # self.merge_CS() # Change to Merge First Than Make
        self.information_summary()
        displaySet(self.DS_set)
        self.rounds()  # Trigger next round

    def teardown(self):
        ds_batch_process_add = {}
        for cs_index in range(len(self.CS_set)):
            self.CS_set[cs_index].update_statistics()
            centroid_point = self.CS_set[cs_index].centroid
            nearest_label = find_nearest_DS(self.DS_set, float("inf"), centroid_point)
            if nearest_label >= 0:
                if nearest_label not in ds_batch_process_add:
                    ds_batch_process_add[nearest_label] = []
                ds_batch_process_add[nearest_label].append(cs_index)

        # self.information_summary()
        for ds_label in ds_batch_process_add:
            for cs_index in ds_batch_process_add[ds_label]:
                self.DS_set[ds_label].merge_Cluster(self.CS_set[cs_index])

        output_cluster_result(self.DS_set, self.RS_set, self.cluster_result_filepath)
        output_intermediate_result(
            self.intermediate_info, self.intermediate_filepath
        )  # This is the last step

    def run(self):
        self.initialisation()
        self.rounds()
        self.teardown()

    def __init__(self, path, K, sample_ratio, cluster_result_filepath, intermediate_filepath):
        tmp = os.listdir(path)
        file_names = [file_name for file_name in tmp if not file_name.startswith(".")]
        file_names.sort()
        self.files = [os.path.join(path, file_name) for file_name in file_names]
        del tmp
        self.alpha = 4
        self.beta = 3 # beta * sigma
        self.K = K
        self.tol = 1e-4
        self.current_file_index = 0
        self.sample_ratio = sample_ratio
        self.round_no = 0
        self.intermediate_info = {}
        self.cluster_result_filepath = cluster_result_filepath
        self.intermediate_filepath = intermediate_filepath
        self.DS_set = []
        self.RS_set = []
        self.CS_set = []


if __name__ == "__main__":
    time_start = time.time()
    tol = 1e-4
    path = sys.argv[1]
    K = int(sys.argv[2])
    cluster_result_filepath = sys.argv[3]
    intermediate_filepath = sys.argv[4]
    bfr = BFR(path, K, 0.5, cluster_result_filepath, intermediate_filepath)
    bfr.run()
    time_end = time.time()
    print("Duration: ", time_end - time_start, "s")
