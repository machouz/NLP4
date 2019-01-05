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
    #output = filter_by_adj(output)
    output = create_output(output)

    np.savetxt(output_file, output, fmt='%s')

