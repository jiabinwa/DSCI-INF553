#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import time
import math

from pyspark import SparkConf, SparkContext


def parse_pred_rslt(pred_json):
    pred = json.loads(pred_json)
    return (pred["user_id"], pred["business_id"]), pred["stars"]


def parse_test_review(test_review_json):
    test_review = json.loads(test_review_json)
    return (test_review["user_id"], test_review["business_id"]), test_review["stars"]


def calc_rmse(cf_type, true_rating_map, pred_rating_map, avg_map):
    RSS = 0
    n = 0
    for uid_bid_pair, true_stars in true_rating_map.items():
        _id = uid_bid_pair[1] if cf_type == "item_based" else uid_bid_pair[0]
        pred_stars = pred_rating_map.get(
            uid_bid_pair, avg_map[_id] if _id in avg_map else None
        )

        if pred_stars is None:
            continue

        RSS += (pred_stars - true_stars) ** 2
        n += 1

    return math.sqrt((1 / n) * RSS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "true_rating_path",
        type=str,
        metavar="true_rating_path",
        help="the path for inputting the true rating file",
    )
    parser.add_argument(
        "pred_rating_path",
        type=str,
        metavar="pred_rating_path",
        help="the path for inputting the predicted rating file",
    )
    parser.add_argument(
        "avg_path",
        type=str,
        metavar="avg_path",
        help="the path for inputting the average rating file",
    )
    parser.add_argument(
        "cf_type",
        type=str,
        metavar="cf_type",
        choices=["item_based", "user_based"],
        help="the type of the collaborative filtering",
    )

    args = parser.parse_args()
    args = vars(args)
    true_rating_path = args["true_rating_path"]
    pred_rating_path = args["pred_rating_path"]
    avg_path = args["avg_path"]
    cf_type = args["cf_type"]

    with open(avg_path) as f:
        avg_map = json.load(f)

    try:
        conf = (
            SparkConf()
            .setAppName("inf553hw3-task3_rmse")
            .set("spark.driver.memory", "4g")
            .set("spark.executor.memory", "4g")
        )
        sc = SparkContext(conf=conf)
        sc.setLogLevel("WARN")

        start = time.time()

        true_rating_map = (
            sc.textFile(true_rating_path).map(parse_test_review).collectAsMap()
        )
        pred_rating_map = (
            sc.textFile(pred_rating_path).map(parse_pred_rslt).collectAsMap()
        )

        rmse = calc_rmse(cf_type, true_rating_map, pred_rating_map, avg_map)
        print("RMSE: {:4f}".format(rmse))

        end = time.time()
        print("Duration: {:.2f}s".format(end - start))
    finally:
        if sc:
            sc.stop()

    if cf_type == "item_based":
        pass
    else:
        pass


if __name__ == "__main__":
    main()
