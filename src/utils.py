import numpy as np


def read_annotations_file(fname):
    output = []
    with open(fname, 'r') as f:
        for line in f:
            sent, per, _, loc, _ = line.split('\t')
            output.append([sent, per, loc])

    return output


if __name__ == '__main__':
    fname = '../data/Annotation/DEV.annotations.txt'
    out = read_annoations_file(fname)
