import numpy as np


def read_annotations_file(fname):
    output = []
    with open(fname, 'r') as f:
        for line in f:
            sent, per, _, loc, _ = line.split('\t')
            output.append([sent, per, loc])

    return output


def split_processed_file(fname):
    processed = []
    for line in file(fname):
        splitted = line.split()
        if len(splitted) > 1 and splitted[0] == '#id:':
            processed.append([])
        processed[-1].append(line[:-1])  # remove \n from line
    return processed


def sentence2dic(processed_sentence):
    sentence = []
    for word in processed_sentence:
        ID, TEXT, LEMMA, TAG, POS, HEAD, DEP, IOB, TYPE = word.split("\t")
        dict = {"ID": ID, "TEXT": TEXT, "LEMMA": LEMMA, "TAG": TAG, "POS": POS, "HEAD": HEAD, "DEP": DEP, "IOB": IOB,
                "TYPE": TYPE}
        sentence.append(dict)
    return sentence


def read_processed_file(fname):
    lines = split_processed_file(fname)
    lines = [processed[2:-1] for processed in lines]  # remove two first comment and last \n)
    lines = [sentence2dic(sentence) for sentence in lines]  # remove two first comment and last \n)
    return lines


if __name__ == '__main__':
    fname = 'data/Processed_Corpus/Corpus.DEV.processed.txt'
    out = read_processed_file(fname)
