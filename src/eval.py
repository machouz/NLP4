import numpy as np
from utils import *
import sys

def get_precision(gold, pred):
    TP = 0.0
    FP = 0.0

    for pr in pred:
        if pr in gold:
            TP += 1
        else:
            FP += 1

    return TP / len(pred)


def get_recall(gold, pred):
    TP = 0.0
    FN = 0.0

    for pr in gold:
        if pr in pred:
            TP += 1
        else:
            FN += 1

    return TP / (TP + FN)

def get_F1(recall, precision):
    return 2*(recall * precision) / (recall + precision)


if __name__ == '__main__':
    gold_file = sys.argv[1] if len(sys.argv) > 1 else 'data/Annotation/DEV.annotations.txt'
    pred_file = sys.argv[2] if len(sys.argv) > 2 else 'data/Annotation/DEV.annotations.txt'


    gold = read_annotations_file(gold_file)
    pred = read_annotations_file(pred_file)

    precision = get_precision(gold, pred)
    recall = get_recall(gold, pred)
    F1 = get_F1(recall, precision)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(F1))