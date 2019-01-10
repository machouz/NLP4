import numpy as np
from utils import *
import sys

def get_precision(gold, pred):
    TP = 0.0

    for num_pred, per_pred, loc_pred in gold:
        for num_gold, per_gold, loc_gold in pred:
            if num_pred == num_gold and per_pred == per_gold and loc_pred == loc_gold:
                TP += 1
                break

    return TP / len(pred)


def contains(gold, one_pred):
    num_p, per_p, loc_p = one_pred
    for num, per, loc in gold:
        if num == num_p and per == per_p and loc == loc_p:
            return True
    return False

def get_recall(gold, pred):
    TP = 0.0

    for num_pred, per_pred, loc_pred in gold:
        for num_gold, per_gold, loc_gold in pred:
            if num_pred == num_gold and per_pred == per_gold and loc_pred == loc_gold:
                TP += 1
                break

    return TP / len(gold)

def get_F1(recall, precision):
    return 2*(recall * precision) / (recall + precision)


def get_errors(gold, pred):
    FP = []
    FN = []

    flag = True
    for num_gold, per_gold, loc_gold in gold:
        for num_pred, per_pred, loc_pred in pred:
            if num_pred == num_gold and per_pred == per_gold and loc_pred == loc_gold:
                flag = False

        if flag:
            FN.append([num_gold, per_gold, loc_gold])

        flag = True

    flag = True

    for num_pred, per_pred, loc_pred in pred:
        for num_gold, per_gold, loc_gold in gold:
            if num_pred == num_gold and per_pred == per_gold and loc_pred == loc_gold:
                flag = False

        if flag:
            FP.append([num_pred, per_pred, loc_pred])

        flag = True

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
    FP, FN = get_errors(gold, pred)
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