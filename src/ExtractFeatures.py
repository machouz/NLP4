import sys
from utils import *
import math


def extract_persons_location(num, sentence):
    persons = []
    locations = []
    new_sentence = []
    i = 0
    while i < len(sentence):
        if sentence[i]['TYPE'] == 'PERSON':
            pers = sentence[i]
            i += 1
            while i < len(sentence) and sentence[i]['IOB'] == 'I':
                pers['TEXT'] += ' ' + sentence[i]['TEXT']
                i += 1
            persons.append(pers)
        elif sentence[i]['TYPE'] == 'GPE' or sentence[i]['TYPE'] == 'NORP' or sentence[i]['TYPE'] == 'LOC':
            loca = sentence[i]
            i += 1
            while i < len(sentence) and sentence[i]['IOB'] == 'I':
                loca['TEXT'] += ' ' + sentence[i]['TEXT']
                i += 1
            locations.append(loca)

        else:
            i += 1

    for per in persons:
        for loc in locations:
            sent = [num, per, loc, sentence]
            new_sentence.append(sent)

    return new_sentence



def exists_mark(per, loc, sentence):
    start = int(per['ID']) - 1
    end = int(loc['ID']) - 1
    for i in range(start, end):
        if sentence[i]["DEP"] == "mark":
            return True
    return False


def filter_by_dependecies(data):
    output = []
    for num, per, loc, sentence in data:
        output.append([num, per, loc, sentence])
    return output


def exists_verb(per, loc, sentence):
    start = int(per['ID']) - 1
    end = int(loc['ID']) - 1
    for i in range(start, end):
        if sentence[i]["POS"] == "VERB":
            return True
    return False


def create_output(data):
    output = []
    for num, per, loc, sentence in data:
        person = per['TEXT']
        location = loc['TEXT']
        sent = map(lambda x: x['TEXT'], sentence)
        sent = " ".join(sent)
        output.append(num + '\t' + person + '\tLive_In\t' + location + '\t(' + sent + ')')

    return output


def verb_lemma(per, loc, sentence):
    start = int(per['ID']) - 1
    end = int(loc['ID']) - 1
    for i in range(start, end):
        if sentence[i]["POS"] == "VERB":
            return sentence[i]["LEMMA"]
    return None


def extract_features(num, person, location, sentence):
    features = {}
    features["distance"] = int(location['ID']) - int(person['ID'])
    features["mark"] = 1 if exists_mark(person, location, sentence) else 0
    features["exist_verb"] = 1 if exists_verb(person, location, sentence) else 0
    features["lemma_verb"] = verb_lemma(person, location, sentence)
    features["per_gazetter"] = 1 if person['LEMMA'] in gazetter else 0
    features["loc_gazetter"] = 1 if location['LEMMA'] in gazetter else 0
    features["tag_loc"] = location['TYPE']

    return features

def extract_features_with_label(num, person, location, sentence, label):
    features =extract_features(num, person, location, sentence)
    features["current_tag"] = label
    return features

def write_features_file(featured_data, output_file):
    data = []
    for featured_word in featured_data:
        label = str(featured_word['current_tag'])
        del featured_word['current_tag']
        for key, value in featured_word.items():
            label += " {}={}".format(key, value)
        data.append(label)
    write_to_file(output_file, data)


def get_label(num, per, loc, gold):
    target = gold.get(num, None)
    if target:
        for targ_per, targ_loc in target:
            if targ_per == per and targ_loc == loc:
                return 1
    return 0


if __name__ == '__main__':

    input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/Processed_Corpus/Corpus.TRAIN.processed.txt'
    gold_file = sys.argv[2] if len(sys.argv) > 2 else 'data/Annotation/TRAIN.annotations.txt'
    output_file = sys.argv[3] if len(sys.argv) > 1 else 'features_file'

    data = read_processed_file(input_file)
    gold = annotations_file_by_sent(gold_file)

    sentences = []

    for num, sentence in data:
        sent = extract_persons_location(num, sentence)
        sentences.extend(sent)

    featured_data = []
    for num, per, loc, sentence in sentences:
        label = get_label(num, per["TEXT"], loc["TEXT"], gold)
        a = extract_features_with_label(num, per, loc, sentence, label=label if label else 0)
        featured_data.append(a)

    write_features_file(featured_data, output_file)
