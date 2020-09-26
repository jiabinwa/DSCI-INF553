#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.metrics import normalized_mutual_info_score
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "true_label_path",
        type=str,
        metavar="true_label_path",
        help="the path of the true cluster label",
    )
    parser.add_argument(
        "predict_label_path",
        type=str,
        metavar="predict_label_path",
        help="the path of the predictive cluster label",
    )

    args = parser.parse_args()
    args = vars(args)
    true_label_path = args["true_label_path"]
    predict_label_path = args["predict_label_path"]

    with open(true_label_path, "r") as f:
        true_label_dict = json.load(f)

    with open(predict_label_path, "r") as f:
        predict_label_dict = json.load(f)

    true_label_ls = [-1] * len(true_label_dict)
    for point_id, cluster_id in true_label_dict.items():
        true_label_ls[int(point_id)] = cluster_id

    predict_label_ls = [-1] * len(predict_label_dict)
    for point_id, cluster_id in predict_label_dict.items():
        predict_label_ls[int(point_id)] = cluster_id

    NMI = normalized_mutual_info_score(true_label_ls, predict_label_ls)
    print("NMI: %.5f" % NMI)


if __name__ == "__main__":
    main()
