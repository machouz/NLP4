import sys
from utils import *
import math





def extract_persons_location(num, sentence, gaz=True):
    persons = []
    locations = []
    new_sentence = []
    i = 0
    while i < len(sentence):
        if sentence[i]['TYPE'] == 'PERSON' :#and sentence[i]['LEMMA'] not in gazetter:
            pers = sentence[i].copy()
            i += 1
            while i < len(sentence) and sentence[i]['IOB'] == 'I':
                pers['TEXT'] += ' ' + sentence[i]['TEXT']
                i += 1
            persons.append(pers)
        elif (sentence[i]['TYPE'] == 'GPE' or sentence[i]['TYPE'] == 'NORP' or sentence[i]['TYPE'] == 'LOC') :#or (gaz and  sentence[i]['LEMMA'] in gazetter):

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

def filter_by_dep(data):
    output = []
    for num, per, loc, sentence in data:
        start = int(per['ID']) - 1
        end = int(loc['ID']) - 1
        flag = True
        for i in range(start, end):
            if sentence[i]["DEP"] == "mark":
                flag = False
                break

        if flag:
            output.append([num, per, loc, sentence])


    return output

def filter_by_dependecies(data):
    output = []
    for num, per, loc, sentence in data:
        output.append([num, per, loc, sentence])
    return output

def filter_by_verb(data):
    output = []
    for num, per, loc, sentence in data:
        flag = False
        for word in sentence:
            if word['POS'] == 'VERB':
                flag = True

        if flag:
            output.append([num, per, loc, sentence])
    return output


def filter_by_from(data):
    output = []
    for num, per, loc, sentence in data:
        start = 0
        end = int(loc['ID']) - 1
        flag = False
        for i in range(start, end):
            if sentence[i]["LEMMA"] == "be":
                flag = True
                break

        if flag:
            output.append([num, per, loc, sentence])

    return output


def filter_person_by_gazeeter(data):
    output=[]
    for num, per, loc, sentence in data:
        if per['LEMMA'] not in gazetter:
            output.append([num, per, loc, sentence])

    return output


def filter_loc_by_gazeeter(data):
    output = []
    for num, per, loc, sentence in data:
        if loc['LEMMA'] in gazetter:
            output.append([num, per, loc, sentence])

    return output

def create_output(data):
    output = []
    for num, per, loc, sentence in data:
        person = per['TEXT']
        location = loc['TEXT']
        sent = map(lambda x: x['TEXT'], sentence)
        sent = " ".join(sent)
        output.append(num + '\t' + person + '\tLive_In\t' + location + '\t(' + sent + ')')

    return output

if __name__ == '__main__':

    input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/Processed_Corpus/Corpus.TRAIN.processed.txt'
    output_file = sys.argv[2] if len(sys.argv) > 1 else 'data/Annotation/output_greedy_train.txt'




    data = read_processed_file(input_file)
    sentences = []

    for num, sentence in data:
        sent = extract_persons_location(num, sentence)
        sentences.extend(sent)

    output = sentences
    #output = filter_person_by_gazeeter(output)
    output = create_output(output)

    np.savetxt(output_file, output, fmt='%s')


'''
Precision: 0.221105527638
Recall: 0.676923076923
F1: 0.333333333333
'''


'''
with gazetter in extract, increase the recall
Precision: 0.137085137085
Recall: 0.730769230769
F1: 0.230862697448
'''