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


def get_errors(gold, pred):
    FP = []
    FN = []

    for true in gold:
        if true not in pred:
            FN.append(true)

    for pr in pred:
        if pr not in gold:
            FP.append(pr)

    return FP, FN


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

    '''
    print
    print("False Positive ------")
    FP, FN = get_errors(gold, pred)
    for fp in FP:
        print fp

    
    print
    print("False Negative ------")
    FP, FN = get_errors(gold, pred)
    for fp in FP:
        print fp
    '''