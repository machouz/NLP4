import numpy as np
from utils import *
import sys

def get_precision(gold, pred):
    TP = 0

def get_recall(gold, pred):
    precision = 0.0

def get_F1(recall, precision):
    return 2*(recall * precision) / (recall + precision)


if __name__ == '__main__':
    gold_file = sys.argv[1] if len(sys.argv) > 1 else '../data/Annotation/DEV.annotations.txt'
    pred_file = sys.argv[2] if len(sys.argv) > 2 else '../data/Annotation/DEV.annotations.txt'

