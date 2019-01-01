import numpy as np


def read_annotations_file(fname):
    output = []
    with open(fname, 'r') as f:
        for line in f:
            sent, per, relation, loc, _ = line.split('\t')
            if relation == 'Live_In':
                output.append(np.array([sent, per, loc]))

    return np.array(output)


if __name__ == '__main__':
    fname = '../data/Annotation/DEV.annotations.txt'
    out = read_annotations_file(fname)
