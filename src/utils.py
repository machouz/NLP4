import numpy as np


def read_annotations_file(fname):
    output = []
    with open(fname, 'r') as f:
        for line in f:
            sent, per, relation, loc, _ = line.split('\t')
            if relation == 'Live_In':
                output.append(np.array([sent, per, loc]))

    return np.array(output)


def split_processed_file(fname):
    processed = []
    for line in file(fname):
        splitted = line.split()
        if len(splitted) > 1 and splitted[0] == '#id:':
            processed.append((splitted[1], []))
        processed[-1][-1].append(line[:-1])  # remove \n from line
    return processed


def sentence2dic(processed_sentence):
    sentence = []
    for word in processed_sentence:
        splitted = word.split("\t")
        if len(splitted) == 9:
            ID, TEXT, LEMMA, TAG, POS, HEAD, DEP, IOB, TYPE = splitted
        else:
            ID, TEXT, LEMMA, TAG, POS, HEAD, DEP, IOB = splitted
            TYPE = ''

        dict = {"ID": ID, "TEXT": TEXT, "LEMMA": LEMMA, "TAG": TAG, "POS": POS, "HEAD": HEAD, "DEP": DEP, "IOB": IOB,
                "TYPE": TYPE}
        sentence.append(dict)
    return sentence


def read_processed_file(fname):
    lines = split_processed_file(fname)
    lines = [(processed[0], processed[1][2:-1]) for processed in lines]  # remove two first comment and last \n
    lines = [(sentence[0], sentence2dic(sentence[1])) for sentence in lines]
    return lines


if __name__ == '__main__':
    fname = 'data/Processed_Corpus/Corpus.DEV.processed.txt'
    out = read_processed_file(fname)
