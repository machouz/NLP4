import numpy as np


def read_annotations_file(fname):
    output = []
    with open(fname, 'r') as f:
        for line in f:
            sent, per, relation, loc, _ = line.split('\t')
            if relation == 'Live_In':
                output.append(np.array([sent, per, loc]))
    return np.array(output)


def dic_annotations_file(ann_name, pro_name):
    proc = read_processed_file(pro_name)
    proc = {key: arr for key, arr in proc}
    output = {}
    for line in file(ann_name):
        sent, per, relation, loc, _ = line.split('\t')
        per = per.split()[0]
        loc = loc.split()[0]
        if relation == 'Live_In':
            if sent not in output:
                output[sent] = []
            sentence = proc[sent]
            loc_word = filter(lambda x: x["TEXT"] == loc, sentence)
            per_word = filter(lambda x: x["TEXT"] == per, sentence)
            if len(loc_word) > 0 and len(per_word) > 0:
                output[sent].append((per_word[0], loc_word[0]))
    return output


def dim_dic_annotations_file(fname):
    output = {}
    with open(fname, 'r') as f:
        for line in f:
            sent, per, relation, loc, _ = line.split('\t')
            if relation == 'Live_In':
                if sent not in output:
                    output[sent] = {}
                if "per" not in output[sent]:
                    output[sent]["per"] = []
                if "loc" not in output[sent]:
                    output[sent]["loc"] = []
                output[sent]["per"].append(per)
                output[sent]["loc"].append(loc)
    return output


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


def get_words_id(word, words_id):
    if word not in words_id:
        return words_id["UUUNKKK"]
    return words_id[word]


def get_ids(data):
    ids = []
    lemmas = []
    tags = []
    poss = []
    heads = []
    deps = []
    iobs = []
    types = []
    for num, sentence in data:
        for word in sentence:
            ids.append(word["ID"])
            lemmas.append(word["LEMMA"])
            tags.append(word["TAG"])
            poss.append(word["POS"])
            heads.append(word["HEAD"])
            deps.append(word["DEP"])
            iobs.append(word["IOB"])
            types.append(word["TYPE"])
    envents = set(ids + lemmas + tags + poss + heads + deps + iobs + types)
    event_id = {word: i for i, word in enumerate(list(set(envents)) + ["UUUNKKK"])}
    id_event = {i: label for label, i in event_id.items()}
    return event_id, id_event


if __name__ == '__main__':
    fname = 'data/Annotation/DEV.annotations.txt'
    pro_name = 'data/Processed_Corpus/Corpus.DEV.processed.txt'
    out = dic_annotations_file(fname, pro_name)
