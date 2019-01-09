from __future__ import unicode_literals
import sys
from utils import *
import math
import spacy


def contains(gold, one_pred):
    num_p, per_p, loc_p = one_pred
    for num, per, loc, sentence in gold:
         if num == num_p and per["TEXT"] == per_p and loc["TEXT"] == loc_p:
             return True
    return False


def check_person(person):
    if person['TYPE'] == 'PERSON':
        return True
    return False

def check_location(location):
    if (location['TYPE'] == 'GPE' or location['TYPE'] == 'NORP' or location[
            'TYPE'] == 'LOC'):# or (gaz and  sentence[i]['LEMMA'] in gazetter)
        return True
    return False
nlp = spacy.load("en_core_web_sm")

def extract_persons_location(num, sentence, gaz=True):
    persons = []
    locations = []
    new_sentence = []
    i = 0
    while i < len(sentence):
        if check_person(sentence[i]):  # and sentence[i]['LEMMA'] not in gazetter:
            pers = sentence[i].copy()
            i += 1
            while i < len(sentence) and sentence[i]['IOB'] == 'I':
                pers['TEXT'] += ' ' + sentence[i]['TEXT']
                i += 1
            persons.append(pers)
        elif check_location(sentence[i]):
            loca = sentence[i].copy()
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


def extract_persons_location_new(num, sentence, gaz=True):
    persons = []
    locations = []
    new_sentence = []
    i = 0
    while i < len(sentence):
        if sentence[i]['TYPE'] == 'PERSON':  # and sentence[i]['LEMMA'] not in gazetter:
            pers = sentence[i].copy()
            i += 1
            while i < len(sentence) and sentence[i]['IOB'] == 'I':
                pers['TEXT'] += ' ' + sentence[i]['TEXT']
                i += 1
            persons.append(pers)
        elif (sentence[i]['TYPE'] == 'GPE' or sentence[i]['TYPE'] == 'NORP' or sentence[i][
            'TYPE'] == 'LOC'):  # or (gaz and  sentence[i]['LEMMA'] in gazetter):

            loca = sentence[i].copy()
            i += 1
            while i < len(sentence) and sentence[i]['IOB'] == 'I':
                loca['TEXT'] += ' ' + sentence[i]['TEXT']
                i += 1
            locations.append(loca)

        elif sentence[i]["POS"] == "PROPN":
            token = sentence[i].copy()
            i += 1
            while i < len(sentence) and sentence[i]["POS"] == "PROPN":
                token["TEXT"] += " " + sentence[i]['TEXT']
                i += 1
            persons.append(token)
            locations.append(token)

        else:
            i += 1

    for per in persons:
        for loc in locations:
            if per["TEXT"] != loc["TEXT"]:
                sent = [num, per, loc, sentence]
                new_sentence.append(sent)

    return new_sentence


def extract_chunk(num, sentence):
    new_sentence = []
    all_tokens = []
    sent = map(lambda x: x['TEXT'], sentence)
    sent = " ".join(sent)

    doc = nlp(sent)
    for chunk in doc.noun_chunks:
        tok = ""
        for token in chunk:
            if token.pos_ == "PROPN":
                tok += token.text + " "

        if len(tok) > 0:
            all_tokens.append(tok.rstrip())

    for token1 in all_tokens:
        for token2 in all_tokens:
            if token1 != token2:
                sent = [num, token1, token2, sentence]
                new_sentence.append(sent)

    return new_sentence


def exists_mark(per, loc, sentence):
    start = int(per['ID']) - 1
    end = int(loc['ID']) - 1
    for i in range(start, end):
        if sentence[i]["DEP"] == "mark":
            return True
    return False


def mark(per, loc, sentence):
    start = int(per['ID']) - 1
    end = int(loc['ID']) - 1
    counter = 0
    for i in range(start, end):
        if sentence[i]["DEP"] == "mark":
            counter += 1
    return counter


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


def exists_punk(person, location, sentence):
    start = int(person['ID']) - 1
    end = int(location['ID']) - 1
    for i in range(start, end):
        if sentence[i]["POS"] == "PUNCT":
            return True
    return False


def locations(sentence):
    counter = 0
    for word in sentence:
        if check_location(word):
            counter += 1
    if counter >= 3:
        return 3
    else:
        return counter

def persons(sentence):
    counter = 0
    for word in sentence:
        if check_person(word):
            counter += 1
    if counter >= 3:
        return 3
    else:
        return counter

def person_subject(person, sentence):
    sent = map(lambda x: x['TEXT'], sentence)
    sent = " ".join(sent)

    doc = nlp(sent)
    pers = person["ID"]


def extract_features(num, person, location, sentence):
    features = {}
    distance = int(location['ID']) - int(person['ID'])
    features["before"] = True if distance > 0 else False
    features["distance"] = distance
    features["mark"] = mark(person, location, sentence)
    features["num_of_locations"] = locations(sentence)
    features["num_of_persons"] = persons(sentence)
    features["exist_verb"] = True if exists_verb(person, location, sentence) else False
    features["per_gazetter"] = True if person['LEMMA'] in gazetter else False
    features["loc_gazetter"] = True if location['LEMMA'] in gazetter else False
    features["lemma_verb"] = verb_lemma(person, location, sentence)
    features["per_tag"] = person["TAG"]
    features["loc_tag"] = location["TAG"]
    features["per_pos"] = person["POS"]
    features["loc_pos"] = location["POS"]
    features["person_type"] = person["TYPE"] if len(person["TYPE"]) > 0 else None
    features["location_type"] = location["TYPE"] if len(location["TYPE"]) > 0 else None
    live = False
    for word in sentence:
        if word["LEMMA"] == 'live':
            live = True

    features["live"] = live

    return features


def extract_features_with_label(num, person, location, sentence, label):
    features = extract_features(num, person, location, sentence)
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
        sent = extract_persons_location_new(num, sentence)
        sentences.extend(sent)


    featured_data = []
    for num, per, loc, sentence in sentences:
        label = get_label(num, per["TEXT"], loc["TEXT"], gold)
        a = extract_features_with_label(num, per, loc, sentence, label=label if label else 0)
        featured_data.append(a)

    write_features_file(featured_data, output_file)
