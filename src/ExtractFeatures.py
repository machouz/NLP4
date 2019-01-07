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
        elif sentence[i]['TYPE'] == 'GPE' or sentence[i]['TYPE'] == 'NORP' or sentence[i]['TYPE'] == 'LOC' :
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


def filter_by_distance(data, distance=5):
    output = []
    for num, per, loc, sentence in data:
        if math.fabs(int(per['ID']) - int(loc['ID'])) < distance:
            output.append([num, per, loc, sentence])

    return output

def filter_by_order(data):
    output = []
    for num, per, loc, sentence in data:
        if int(per['ID']) - int(loc['ID']) < 0:
            output.append([num, per, loc, sentence])

    return output


def next_node(node, sentence):
    next_id = node['HEAD']
    if next_id == '0':
        return None
    return sentence[int(next_id) - 1]


def filter_by_subject(data): #bad recall, pred only 7
    '''
    Precision: 0.571428571429
    Recall: 0.0461538461538
    F1: 0.085409252669
    :param data:
    :return:
    '''
    output = []
    for num, per, loc, sentence in data:
        next = loc
        while next != None:
            if next['ID'] == per['ID']:
                output.append([num, per, loc, sentence])
                break

            next = next_node(next, sentence)
    return output

def filter_by_adj(data):
    output = []
    for num, per, loc, sentence in data:
        if loc['POS'] != 'ADJ':
            output.append([num, per, loc, sentence])

    return output

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


def extract_features(num, person, location, sentence, label):
    distance = int(location['ID']) - int(person['ID'])
    mark = exists_mark(person, location, sentence)
    exist_verb = exists_verb(person, location, sentence)
    lemma_verb = verb_lemma(person, location, sentence)
    per_gazetter = 1 if person['LEMMA'] in gazetter else 0
    loc_gazetter = 1 if location['LEMMA'] in gazetter else 0

    tag_loc = location['TYPE']




if __name__ == '__main__':

    input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/Processed_Corpus/Corpus.TRAIN.processed.txt'
    output_file = sys.argv[2] if len(sys.argv) > 1 else 'features_file'


    data = read_processed_file(input_file)
    sentences = []

    for num, sentence in data:
        sent = extract_persons_location(num, sentence)
        sentences.extend(sent)

    output = sentences
    #output = filter_by_subject(output)
    output = create_output(output)

    np.savetxt(output_file, output, fmt='%s')

