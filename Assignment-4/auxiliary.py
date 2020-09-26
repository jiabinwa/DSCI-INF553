"""
    USC Spring 2020
    INF 553 Foundations of Data Mining
    Assignment 4
    
    Student Name: Jiabin Wang
    Student ID: 4778-4151-95
"""
from pyspark import SparkConf, SparkContext, StorageLevel
from pyspark.sql import SparkSession, SQLContext
from collections import deque
import os
import re
import json
import time
import sys
import math
import random
import itertools
import copy


def makeCombinations(bid, uidList):
    answerSet = set()
    candidates = list(itertools.combinations(uidList, 2))
    for candidate in candidates:
        uid1 = candidate[0]
        uid2 = candidate[1]
        if uid1 > uid2:
            uid1, uid2 = uid2, uid1
        if (uid1, uid2) not in answerSet:
            answerSet.add((uid1, uid2))
    answer = []
    for uidPair in answerSet:
        answer.append((uidPair, [bid]))
    return answer


def makeListSorted(l):
    l.sort()
    return l


def buildGraph(spark, THRESHOLD, input_file_path):
    spark.sparkContext.setLogLevel("ERROR")
    csv_file = spark.sparkContext.textFile(input_file_path)
    header = csv_file.first()
    csv_file = csv_file.filter(lambda _: _ != header)
    rdd_file = (
        csv_file.map(lambda line: (line.split(",")[1], [line.split(",")[0]]))
        .reduceByKey(lambda a, b: a + b)
        .mapValues(lambda values: list(set(values)))
        .flatMap(lambda x: makeCombinations(x[0], x[1]))
        .reduceByKey(lambda a, b: a + b)
        .filter(lambda x: len(x[1]) >= THRESHOLD)
    )
    rdd_file.persist(storageLevel=StorageLevel.MEMORY_AND_DISK)
    user_vertices = rdd_file.flatMap(lambda x: [(x[0][0],), (x[0][1],)]).distinct()
    user_edges_si = rdd_file.map(lambda x: (x[0][0], x[0][1]))
    user_edges_bi = rdd_file.flatMap(lambda x: [(x[0][0], x[0][1]), (x[0][1], x[0][0])])
    return user_vertices, user_edges_si, user_edges_bi


class Graph:
    # const variables
    neighbour_permanent = {}
    vertices = {}
    edges = {}

    neighbour = {}

    # variables for computing
    vistit_order = []
    vertex_parent = {}
    shortest_path = {}
    betweeness = {}

    # Modularity
    Q_max_Modularity = float("-inf")
    best_partition = []

    def __init__(self, neighbour, vertices, edges):
        self.neighbour = neighbour
        self.vertices = vertices
        self.edges = edges

    def betweenessCalculation(self):
        for root in self.vertices:
            self.clear()
            self.bfs(root)
            self.bottomUp()
            localBetweeness = self.getLocalBetweeness()
            for edge in localBetweeness:
                self.edges[edge] = self.edges[edge] + localBetweeness[edge]
        for edge in self.edges:
            self.edges[edge] = self.edges[edge] / 2
        self.clear()
        self.betweeness = self.edges

    def calculate_modularity(self, current_partition):
        Q = 0
        m = len(self.edges)
        for clique in current_partition:
            if len(clique) == 1:
                continue
            for i in clique:
                for j in clique:
                    if i == j:
                        continue
                    A_ij = 0
                    if j in self.neighbour_permanent[i] and i in self.neighbour_permanent[j]:
                        A_ij = 1
                    k_i = len(self.neighbour_permanent[i])
                    k_j = len(self.neighbour_permanent[j])
                    Q = Q + (A_ij - (k_i * k_j) / (2 * m))
        Q = Q / (2 * m)
        return Q
    
    def has_edges(self):
        neighbour = self.neighbour
        for vertex in neighbour:
            if len(neighbour[vertex]) > 0:
                return True
        return False

    def cut_edge(self, pairList):
        for pair in pairList:
            u = pair[0]
            v = pair[1]
            if v in self.neighbour[u]:
                self.neighbour[u].remove(v)
            if u in self.neighbour[v]:
                self.neighbour[v].remove(u)
        
    def get_highest_betweeness_pair(self):
        highest_betweeness_pair_list = []
        betweeness_list = self.calculate_current_betweeness()
        highest_betweeness_score = max(betweeness_list, key=lambda x: x[2])[2]
        for triple in betweeness_list:
            if abs(triple[2] - highest_betweeness_score) <= 0.00000001:
                highest_betweeness_pair_list.append( (triple[0], triple[1]) )
                break
        # highest_betweeness_score = betweeness_list[0][2]
        # highest_betweeness_pair_list.append((betweeness_list[0][0],betweeness_list[0][1]))
        '''
        index = 1
        while index < len(betweeness_list) and abs(betweeness_list[index][2] - highest_betweeness_score) <= 0.000001:
            highest_betweeness_pair_list.append((betweeness_list[index][0],betweeness_list[index][1]))
            index = index + 1
        '''
        return highest_betweeness_pair_list


    def detect_community(self):
        self.neighbour_permanent = copy.deepcopy(self.neighbour)  # store
        while(self.has_edges()):
            self.detect_community_impl()
            highest_betweeness_pair_list = self.get_highest_betweeness_pair()
            self.cut_edge(highest_betweeness_pair_list)
            
        return self.best_partition

    def detect_community_impl(self):
        current_partition = []
        current_vertices = set(self.vertices)  # all vertices
        while len(current_vertices) > 0:
            visited_nodes, _, __ = self.bfs(list(current_vertices)[0])
            visited_nodes = list(visited_nodes)
            visited_nodes.sort()
            current_partition.append(list(visited_nodes))
            current_vertices = current_vertices - set(visited_nodes)
        
        current_modularity = self.calculate_modularity(current_partition)
        if current_modularity > self.Q_max_Modularity:
            self.Q_max_Modularity = current_modularity
            self.best_partition = current_partition

    def clear(self):
        self.vistit_order = []
        self.vertex_parent = {}
        self.shortest_path = {}
        self.betweeness = {}

    def getLocalBetweeness(self):
        return self.betweeness

    def getBetweeness(self):
        return self.edges

    def bfs(self, root, save=True):
        vistit_order = deque([])
        vertex_parent = {}
        vertex_parent[root] = []
        shortest_path = {}
        shortest_path[root] = {"distance": 0, "count": 1}
        queue = deque([root])
        level = -1
        while queue:
            level = level + 1
            size = len(queue)
            for i in range(size):
                i
                current_node = queue.popleft()
                vistit_order.append(current_node)

                neighbour_nodes = self.neighbour[current_node]
                for neighbour_node in neighbour_nodes:
                    # Calculate the short distance and count
                    if neighbour_node not in shortest_path:
                        shortest_path[neighbour_node] = {
                            "distance": level + 1,
                            "count": shortest_path[current_node]["count"],
                        }
                    elif level + 1 == shortest_path[neighbour_node]["distance"]:
                        shortest_path[neighbour_node]["count"] = (
                            shortest_path[neighbour_node]["count"]
                            + shortest_path[current_node]["count"]
                        )

                    if neighbour_node not in vertex_parent:
                        # not visted:
                        queue.append(neighbour_node)
                        vertex_parent[neighbour_node] = []
                    if (
                        neighbour_node not in vertex_parent[current_node]
                        and level < shortest_path[neighbour_node]["distance"]
                    ):
                        # this is parent child relationship:
                        vertex_parent[neighbour_node].append(current_node)

        self.vistit_order = vistit_order
        self.vertex_parent = vertex_parent
        self.shortest_path = shortest_path
        return vistit_order, vertex_parent, shortest_path

    def bottomUp(self):
        betweeness = {}
        vistit_order = self.vistit_order
        vertex_parent = self.vertex_parent
        shortest_path = self.shortest_path

        vertex_weight = {}
        for vertex in vistit_order:
            vertex_weight[vertex] = 1

        for vertex in list(reversed(vistit_order)):
            for parent in vertex_parent[vertex]:
                edgeWeight = vertex_weight[vertex] * (
                    shortest_path[parent]["count"] / shortest_path[vertex]["count"]
                )
                v1 = vertex
                v2 = parent
                if v1 > v2:
                    v1, v2 = v2, v1
                if (v1, v2) not in betweeness:
                    betweeness[(v1, v2)] = 0
                betweeness[(v1, v2)] = betweeness[(v1, v2)] + edgeWeight
                vertex_weight[parent] = vertex_weight[parent] + edgeWeight

        self.betweeness = betweeness
    
    def calculate_current_betweeness(self):
        for edge in self.edges:
            self.edges[edge] = 0
        self.betweenessCalculation()
        betweeness_set = self.getBetweeness()
        betweeness_list = []
        for b_pair in betweeness_set:
            betweeness_list.append((b_pair[0], b_pair[1], betweeness_set[b_pair]))
        betweeness_list = sorted(betweeness_list, key=lambda x: (-x[2], x[0], x[1]))
        return betweeness_list


if __name__ == "__main__":
    time_start = time.time()
    THRESHOLD = int(sys.argv[1])
    input_file_path = sys.argv[2]
    spark = (
        SparkSession.builder.master("local[3]")
        .appName("inf553")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )
    user_vertices, user_edges_si, user_edges_bi = buildGraph(
        spark, THRESHOLD, input_file_path
    )
    time_end = time.time()
    print("Duration: ", time_end - time_start, "s")
